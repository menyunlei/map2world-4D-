"""Raw RGB map image -> map memory -> directly generated 3D Gaussian world.

Student inference inputs: the RGB map image, its world bounds, the region to
generate and a fixed world code. No semantic map, height field, building IDs,
seg volume or CityDreamer4D call appears anywhere in this file.

  MapMemory   E(I_map): frozen pretrained ViT (or a tiny CNN for CPU tests) plus
              a trainable high-resolution RGB stem and a relational transformer
              over map tokens with absolute 2D positions. Multiscale [/4,/8,/16].
  WorldDecoder G(Z_M, z): one query per map column (2D cell of the region).
              Queries read the memory with deformable attention around their
              own map location (local structure) on features already mixed by
              the relational module (wider context), then expand into a stack
              of height tokens with column self-attention (heights are
              predicted, never supplied). Each token emits K Gaussians.
  Sky         Direction -> colour, conditioned on the world code, so teacher
              frames with sky can be fitted without putting sky in the scene.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier(x, bands):
    """x [..., d] in [-1, 1] -> [..., d * 2 * bands]."""
    f = (2. ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def grid_pos(h, w, dim, device):
    """Absolute 2D sinusoidal position code [h*w, dim] over normalised map coords."""
    r = torch.linspace(-1, 1, h, device=device)
    c = torch.linspace(-1, 1, w, device=device)
    rr, cc = torch.meshgrid(r, c, indexing='ij')
    code = fourier(torch.stack([rr, cc], -1).reshape(-1, 2), dim // 4)
    return code[:, :dim]


# ----------------------------------------------------------------- map memory
class TinyBackbone(nn.Module):
    """Random-init stand-in with the ViT interface; CPU tests only, never a result."""
    patch = 16

    def __init__(self, dim=64):
        super().__init__()
        self.out_dim = dim
        self.net = nn.Sequential(nn.Conv2d(3, dim, 4, 4), nn.GELU(), nn.Conv2d(dim, dim, 4, 4))

    def forward(self, x):
        return self.net(x)


class HFViTBackbone(nn.Module):
    """Frozen Hugging Face ViT (DINOv2/DINOv3) returning its patch-token grid."""

    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(path).eval().requires_grad_(False)
        cfg = self.model.config
        self.patch = cfg.patch_size
        self.out_dim = cfg.hidden_size
        self.skip = 1 + getattr(cfg, 'num_register_tokens', 0)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[:, None, None], persistent=False)
        self.register_buffer('std', torch.tensor([.229, .224, .225])[:, None, None], persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        tokens = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, self.skip:]
        assert tokens.shape[1] == h * w, 'map size must be a multiple of the patch size'
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)


class MapMemory(nn.Module):
    def __init__(self, backbone, dim=128, relation_layers=2, heads=4):
        super().__init__()
        self.backbone = backbone
        self.adapt = nn.Conv2d(backbone.out_dim, dim, 1)
        # Trainable full-resolution path: thin roads and kerbs are below ViT patch size.
        self.stem4 = nn.Sequential(nn.Conv2d(3, dim // 2, 3, 2, 1), nn.GELU(), nn.Conv2d(dim // 2, dim, 3, 2, 1), nn.GELU())
        self.stem8 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.GELU())
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0., batch_first=True, norm_first=True)
        self.relation = nn.TransformerEncoder(layer, relation_layers, enable_nested_tensor=False)
        self.fuse8 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.fuse4 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.dim = dim

    def forward(self, image):
        """image [1, 3, H, W] in [0, 1]; H, W multiples of 16. Returns [/4, /8, /16] maps."""
        x16 = self.adapt(self.backbone(image))
        b, d, h, w = x16.shape
        tokens = x16.flatten(2).transpose(1, 2) + grid_pos(h, w, d, image.device)[None]
        x16 = self.relation(tokens).transpose(1, 2).reshape(b, d, h, w)
        x4 = self.stem4(image)
        x8 = self.stem8(x4)
        x8 = self.fuse8(x8 + F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False))
        x4 = self.fuse4(x4 + F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False))
        return [x4, x8, x16]


# --------------------------------------------------------------- map reading
class MapDeformableAttention(nn.Module):
    """Each query samples P learned points per level and head around its own map
    location (normalised coords, x = col, y = row) and mixes them by weights."""

    def __init__(self, dim, heads=4, levels=3, points=4, reach=(.04, .08, .16)):
        super().__init__()
        self.h, self.l, self.p, self.dh = heads, levels, points, dim // heads
        self.offsets = nn.Linear(dim, heads * levels * points * 2)
        self.weights = nn.Linear(dim, heads * levels * points)
        self.values = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(levels)])
        self.out = nn.Linear(dim, dim)
        self.register_buffer('reach', torch.tensor(reach), persistent=False)
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            # Start as a ring of points around the reference location.
            ang = torch.arange(points) * 2 * math.pi / points
            ring = torch.stack([ang.cos(), ang.sin()], -1)
            self.offsets.bias.copy_(ring[None, None].expand(heads, levels, -1, -1).reshape(-1))

    def forward(self, query, ref, memory):
        """query [N, D], ref [N, 2] in [-1, 1] (col, row), memory list of [1, D, h, w]."""
        n = query.shape[0]
        off = self.offsets(query).view(n, self.h, self.l, self.p, 2).tanh()
        off = off * self.reach[None, None, :, None, None]
        w = self.weights(query).view(n, self.h, self.l * self.p).softmax(-1).view(n, self.h, self.l, self.p)
        out = 0
        for li, level in enumerate(memory):
            v = self.values[li](level)[0].view(self.h, self.dh, *level.shape[-2:])
            loc = (ref[:, None, None] + off[:, :, li]).permute(1, 0, 2, 3)        # [h, N, P, 2]
            s = F.grid_sample(v, loc, mode='bilinear', padding_mode='zeros', align_corners=False)
            out = out + (s * w[:, :, li].permute(1, 0, 2)[:, None]).sum(-1)       # [h, dh, N]
        return self.out(out.permute(2, 0, 1).reshape(n, -1))


def sample_points(memory, ref):
    """Bilinear features at ref [N, 2] from the finest memory level."""
    s = F.grid_sample(memory[0], ref[None, :, None], mode='bilinear', align_corners=False)
    return s[0, :, :, 0].T


# ------------------------------------------------------------------- decoder
class WorldDecoder(nn.Module):
    def __init__(self, dim=128, style_dim=64, z_tokens=24, gaussians_per_token=4,
                 column_layers=2, heads=4, map_layers=2):
        super().__init__()
        self.dim, self.z_tokens, self.k = dim, z_tokens, gaussians_per_token
        self.pos = nn.Linear(4 * 8, dim)
        self.style = nn.Linear(style_dim, 2 * dim)
        self.read = nn.ModuleList([MapDeformableAttention(dim, heads) for _ in range(map_layers)])
        self.read_mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
                                       for _ in range(map_layers)])
        self.read_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(map_layers)])
        self.height = nn.Embedding(z_tokens, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0., batch_first=True, norm_first=True)
        self.column = nn.TransformerEncoder(layer, column_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.k * 14))
        with torch.no_grad():
            last = self.head[-1]
            last.weight.mul_(.01)
            b = last.bias.view(self.k, 14)
            b.zero_()
            b[:, 3:6] = -1.     # scale: 0.37 cell
            b[:, 6] = 1.        # identity quaternion
            b[:, 10] = -3.      # opacity: ~0.05, empty until supported by the views

    def columns(self, memory, ref, style, chunk=8192):
        """Column features [N, D] for map columns at ref [N, 2] (col, row in [-1, 1])."""
        film = self.style(style)
        gamma, beta = film[:self.dim], film[self.dim:]
        out = []
        for s in range(0, ref.shape[0], chunk):
            r = ref[s:s + chunk]
            q = self.pos(fourier(r, 8)) + sample_points(memory, r)
            q = q * (1 + gamma) + beta
            for attend, mlp, norm in zip(self.read, self.read_mlp, self.read_norm):
                q = q + attend(norm(q), r, memory)
                q = q + mlp(q)
            out.append(q)
        return torch.cat(out)

    def forward(self, memory, ref, style, chunk=4096):
        """Raw per-Gaussian parameters [N, z_tokens, K, 14] for columns at ref."""
        q = self.columns(memory, ref, style)
        out = []
        for s in range(0, q.shape[0], chunk):
            tokens = q[s:s + chunk, None] + self.height.weight[None]
            out.append(self.head(self.column(tokens)).view(tokens.shape[0], self.z_tokens, self.k, 14))
        return torch.cat(out)


class Sky(nn.Module):
    def __init__(self, style_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * 2 * 4 + style_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, dirs, style):
        x = fourier(dirs, 4)
        return torch.sigmoid(self.net(torch.cat([x, style.expand(*x.shape[:-1], -1)], -1)))


# --------------------------------------------------------------- full model
class RawMapWorld(nn.Module):
    """region: {'row': [r0, r1], 'col': [c0, c1], 'z': [z0, z1]} world units.
    map_bounds: {'row': [...], 'col': [...]} world extent of the map image."""

    def __init__(self, backbone, dim=128, style_dim=64, cell=4., z_tokens=24, gaussians_per_token=4, **kw):
        super().__init__()
        self.memory = MapMemory(backbone, dim)
        self.decoder = WorldDecoder(dim, style_dim, z_tokens, gaussians_per_token, **kw)
        self.sky = Sky(style_dim)
        self.cell, self.style_dim = float(cell), style_dim

    def encode(self, image):
        return self.memory(image)

    def anchors(self, region, map_bounds, device):
        r0, r1 = region['row']
        c0, c1 = region['col']
        rows = torch.arange(r0 + self.cell / 2, r1, self.cell, device=device)
        cols = torch.arange(c0 + self.cell / 2, c1, self.cell, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing='ij')
        centre = torch.stack([rr.reshape(-1), cc.reshape(-1)], -1)
        (mr0, mr1), (mc0, mc1) = map_bounds['row'], map_bounds['col']
        ref = torch.stack([(centre[:, 1] - mc0) / (mc1 - mc0) * 2 - 1, (centre[:, 0] - mr0) / (mr1 - mr0) * 2 - 1], -1)
        return centre, ref

    def generate(self, memory, region, map_bounds, style):
        """Gaussian world for region. Deterministic given (memory, style)."""
        device = memory[0].device
        centre, ref = self.anchors(region, map_bounds, device)
        raw = self.decoder(memory, ref, style)                                # [N, Z, K, 14]
        z0, z1 = region['z']
        dz = (z1 - z0) / self.decoder.z_tokens
        zc = z0 + dz * (torch.arange(self.decoder.z_tokens, device=device) + .5)
        cell = torch.tensor([self.cell, self.cell, dz], device=device)
        anchor = torch.cat([centre[:, None, None].expand(-1, raw.shape[1], raw.shape[2], -1),
                            zc[None, :, None, None].expand(raw.shape[0], -1, raw.shape[2], 1)], -1)
        raw = raw.reshape(-1, 14)
        anchor = anchor.reshape(-1, 3)
        return {'means': anchor + torch.tanh(raw[:, :3]) * cell,
                'scales': torch.exp(raw[:, 3:6].clamp(-5, 2)) * cell,
                'quats': F.normalize(raw[:, 6:10], dim=-1),
                'opacities': torch.sigmoid(raw[:, 10]),
                'colors': torch.sigmoid(raw[:, 11:14])}

    def forward(self, image, region, map_bounds, style):
        return self.generate(self.encode(image), region, map_bounds, style)


def build_backbone(spec):
    if spec == 'tiny':
        return TinyBackbone()
    if spec.startswith('hf:'):
        return HFViTBackbone(spec[3:])
    raise ValueError('backbone must be "tiny" or "hf:<local path>"')


def world_code(seed, dim, device='cpu'):
    """Fixed world-level code: seed -1 is the prior mean (zeros)."""
    if seed < 0:
        return torch.zeros(dim, device=device)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g).to(device)


def composite(rgb, alpha, sky_rgb):
    return rgb + (1 - alpha)[..., None] * sky_rgb

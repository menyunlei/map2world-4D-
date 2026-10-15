"""One map-injection intervention; matched capacity and unchanged native geometry."""
import torch
from torch import nn
import torch.nn.functional as F


class SubcellRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(48, 128), nn.SiLU())
        self.write = nn.ModuleList([nn.Linear(128, 3072) for _ in range(6)])
        for layer in self.write:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, mask, site, dtype):
        return (self.write[site](self.encode(features.float())) * mask[:, None]).to(dtype)


def spatial_features(channels, xyz, mode, gather_grid):
    # Pixel-unshuffle retains channel-major [dy,dx] ordering within each 4x4 cell.
    grid = F.pixel_unshuffle(channels.float(), 4)
    if mode == 'control':
        grid = grid.reshape(1, 3, 16, 16, 16)
        grid = grid.mean(2, keepdim=True).expand_as(grid).reshape(1, 48, 16, 16)
    elif mode == 'shuffled':
        grid = grid.reshape(1, 3, 16, 16, 16)
        # Fixed permutation of only subcell addresses; per-role values/counts unchanged.
        order = torch.tensor([5, 12, 1, 8, 15, 2, 11, 4, 9, 0, 7, 14, 3, 10, 13, 6], device=grid.device)
        grid = grid[:, :, order].reshape(1, 48, 16, 16)
    else:
        assert mode == 'correct'
    return gather_grid(grid, xyz).float()


def attach(model, mode):
    from typed_write import _bounding_boxes
    from hierarchy_model_v2 import gather_grid
    branch = model.branch
    original_prepare, original_read = branch.prepare, branch.read
    branch.subcell = SubcellRead().to(next(model.parameters()).device)

    def prepare(encoder, raster, xyz, old, relation):
        prepared = original_prepare(encoder, raster, xyz, old, relation)
        m = relation['masks'].float() * relation['valid'][:, None, None].float()
        footprint = relation['roles'][:, 1].float() * relation['valid'].float()
        fill = (m * footprint[:, None, None]).sum(0).clamp(0, 1)
        void = (m * (relation['roles'][:, 3].float() * relation['valid'].float())[:, None, None]).sum(0).clamp(0, 1)
        bbox = _bounding_boxes(m, footprint).sum(0).clamp(0, 1)
        channels = torch.stack([(fill - void).clamp_min(0), void, (bbox - fill).clamp_min(0)])[None]
        features = spatial_features(channels, xyz, mode, gather_grid)
        # Existing gates enforce the exact same map validity and support region.
        support = (prepared[5]['gates'].sum(-1) > 0).float()
        assert torch.allclose(features.reshape(-1, 3, 16).mean(-1) * support[:, None],
                              prepared[5]['gates'], atol=1e-6), 'SUBCELL_ADDRESS_OR_VALIDITY_MISMATCH'
        return (*prepared, features, support)

    def read(hidden, prepared, site):
        previous = original_read(hidden, prepared[:6], site)
        residual = branch.subcell(prepared[6], prepared[7], site, hidden.dtype)
        return previous + residual.reshape_as(hidden)

    branch.prepare, branch.read = prepare, read
    return original_read

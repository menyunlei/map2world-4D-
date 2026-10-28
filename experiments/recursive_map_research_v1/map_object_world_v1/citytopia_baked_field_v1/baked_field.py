"""Baked view-independent field for the CityDreamer4D volumetric generators.

DRAFT: untested. Not validated against GanCraftGenerator or on GPU.

Root cause addressed here. The native generators re-run a per-sample MLP at
render time: 24 samples per ray, one full pass for the background and one
additional pass per visible building instance, for every frame. Yet the field
being evaluated does not depend on the camera: RenderMLP receives only the
sample position (hash-grid / sin-cos positional features plus layout features
sampled at that position), the containing voxel's class one-hot and the
instance style. View dependence enters only through the sky MLP (ray direction)
and the 2D RenderCNN.

This module evaluates the field once per lattice point on the shell the native
sampler can reach and renders new cameras by trilinear gathering, keeping the
native ray caster, sampler, volume-rendering weights, sky and RenderCNN
unchanged. Baking is chunked in the map plane.
"""
import itertools

import torch
import torch.nn.functional as F

MISSING = -1
SAMPLE_DEPTH = 3          # native _get_sampled_coordinates(sample_depth=3)
CORNERS = torch.tensor(list(itertools.product([0, 1], repeat=3)), dtype=torch.long)


# ---------------------------------------------------------------- label helpers
def background_labels(voxel_id, inst):
    """Mirror of scripts.inference.get_background_voxel_ids for a fixed contract."""
    lo, hi = inst['BLDG_INST_RANGE']
    clo, chi = inst['CAR_INST_RANGE']
    out = voxel_id.clone()
    out[(voxel_id >= lo) & (voxel_id < hi)] = inst['BG_FACADE_CID']
    out[(voxel_id >= clo) & (voxel_id < chi)] = inst['BG_CAR_CID']
    return out


def visible_buildings(first_hit, inst):
    """Mirror of scripts.inference.get_visible_buildings."""
    lo, hi = inst['BLDG_INST_RANGE']
    m, off = inst['BLDG_INST_MULTIPLIER'], inst['ROOF_OFFSET']
    ids = torch.unique(first_hit[(first_hit >= lo) & (first_hit < hi)])
    facades = ids[ids % m == 0]
    roofs = ids[ids % m == off % m] - off
    parents = torch.unique(torch.cat([facades, roofs]))
    return parents[(parents >= lo) & (parents < hi)]


def building_of(ids, inst):
    """(building id, is_roof) per voxel id; building 0 where the id is not a building."""
    lo, hi = inst['BLDG_INST_RANGE']
    m, off = inst['BLDG_INST_MULTIPLIER'], inst['ROOF_OFFSET']
    ids = ids.long()
    inside = (ids >= lo) & (ids < hi)
    facade = inside & (ids % m == 0)
    roof = inside & (ids % m == off % m) & ~facade
    building = torch.where(facade, ids, torch.where(roof, ids - off, torch.zeros_like(ids)))
    building = torch.where((building >= lo) & (building < hi), building, torch.zeros_like(building))
    return building, roof & (building > 0)


def building_label(building, is_roof, n_bg):
    return n_bg + 2 * building.long() + is_roof.long()


# ------------------------------------------------------------------- geometry
def shell_voxels(solid, thickness):
    """Solid voxels within L-inf distance `thickness` of a non-solid voxel.

    The native sampler places all samples within SAMPLE_DEPTH units of in-solid
    path from the first intersection, so `thickness` = SAMPLE_DEPTH + 1 covers
    every sample position the native renderer can evaluate.
    """
    dtype = torch.float16 if solid.device.type == 'cuda' else torch.float32
    free = (~solid).to(dtype)[None, None]
    for _ in range(thickness):
        free = F.max_pool3d(free, 3, stride=1, padding=1)
    return solid & (free[0, 0] > 0)


def lattice_points(voxels, labels, s):
    """All (lattice point, label) pairs covering the voxels: (s+1)^3 points each."""
    axis = torch.arange(s + 1, device=voxels.device)
    offs = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    points = (voxels[:, None, :] * s + offs[None]).reshape(-1, 3)
    return points, labels[:, None].expand(-1, offs.shape[0]).reshape(-1)


def calibrate(volume, world, ids, tries=None):
    """Find the axis permutation and half-voxel offset under which floor(world)
    indexes `volume` consistently with the ray caster's voxel ids.

    Returns (perm, offset, agreement). The native path is exact only under the
    caster's own convention, so the renderer should refuse to run below 0.98.
    """
    valid = ids != 0
    world, ids = world[valid], ids[valid]
    best = (None, None, -1.)
    offsets = tries or [-.5, 0., .5]
    for perm in itertools.permutations(range(3)):
        for off in itertools.product(offsets, repeat=3):
            off_t = torch.tensor(off, device=world.device)
            idx = torch.floor(world[:, list(perm)] + off_t).long()
            ok = torch.ones(len(idx), dtype=torch.bool, device=idx.device)
            for a in range(3):
                ok &= (idx[:, a] >= 0) & (idx[:, a] < volume.shape[a])
            if ok.sum() == 0:
                continue
            got = volume[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
            agreement = (got.long() == ids[ok].long()).float().mean().item() * ok.float().mean().item()
            if agreement > best[2]:
                best = (tuple(perm), off_t, agreement)
    return best


# --------------------------------------------------------------- lattice table
class LatticeTable:
    """Sparse (lattice point, label) -> channel vector storage.

    Entries are indexed per lattice column by up to `ranges_per_column` runs of
    consecutive Z for one label: (z0, length, base slot, label).
    """

    def __init__(self, columns, z_count, channels, ranges_per_column=8, capacity=1 << 16,
                 device='cpu', dtype=torch.float16, label_space=1 << 17):
        nx, ny = columns
        self.shape = (nx, ny, z_count)
        self.R = ranges_per_column
        self.label_space = label_space
        self.ranges = torch.zeros((nx, ny, ranges_per_column, 4), dtype=torch.int32, device=device)
        self.values = torch.zeros((capacity, channels), dtype=dtype, device=device)
        self.size = 0

    @property
    def device(self):
        return self.values.device

    @property
    def channels(self):
        return self.values.shape[1]

    def _key(self, X, Y, Z, label):
        nx, ny, nz = self.shape
        return ((X.long() * ny + Y.long()) * self.label_space + label.long()) * nz + Z.long()

    def _decode(self, key):
        nx, ny, nz = self.shape
        Z = key % nz
        rest = key // nz
        label = rest % self.label_space
        rest = rest // self.label_space
        return rest // ny, rest % ny, Z, label

    def lookup(self, X, Y, Z, label, piece=1 << 21):
        """Slot per query, MISSING where absent. Flat long tensors of equal length."""
        nx, ny, nz = self.shape
        out = torch.full(X.shape, MISSING, dtype=torch.long, device=self.device)
        for start in range(0, X.numel(), piece):
            sl = slice(start, start + piece)
            x, y, z, l = X[sl], Y[sl], Z[sl], label[sl]
            ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz) & (l >= 0)
            r = self.ranges[x.clamp(0, nx - 1), y.clamp(0, ny - 1)].long()      # [m, R, 4]
            z0, length, base, lab = r.unbind(-1)
            hit = (length > 0) & (lab == l[:, None]) & (z[:, None] >= z0) & (z[:, None] < z0 + length)
            slot = torch.where(hit, base + z[:, None] - z0, torch.full_like(base, MISSING))
            slot = slot.max(dim=1).values
            out[sl] = torch.where(ok, slot, torch.full_like(slot, MISSING))
        return out

    def gather(self, slot):
        v = self.values[slot.clamp(min=0)].float()
        v[slot < 0] = 0
        return v

    def plan(self, X, Y, Z, label):
        """Decide which new entries to store. Returns (entries [n,4] long, runs [k,5] long
        (X, Y, label, z0, length)). Entries already present are dropped.

        UNTESTED: the overflow / hull-merge path has not been exercised.
        """
        key = torch.unique(self._key(X, Y, Z, label))
        X, Y, Z, label = self._decode(key)
        present = self.lookup(X, Y, Z, label) >= 0
        X, Y, Z, label = X[~present], Y[~present], Z[~present], label[~present]
        for _ in range(2):
            runs = self._runs(X, Y, Z, label)
            over = self._overflow(runs)
            if not over.any():
                return torch.stack([X, Y, Z, label], 1), runs
            cols = runs[over][:, :2]
            key_col = cols[:, 0] * self.shape[1] + cols[:, 1]
            in_over = torch.isin(X * self.shape[1] + Y, key_col)
            keep = ~in_over
            merged = self._hull(X[in_over], Y[in_over], Z[in_over], label[in_over])
            X = torch.cat([X[keep], merged[0]]); Y = torch.cat([Y[keep], merged[1]])
            Z = torch.cat([Z[keep], merged[2]]); label = torch.cat([label[keep], merged[3]])
            key = torch.unique(self._key(X, Y, Z, label))
            X, Y, Z, label = self._decode(key)
            present = self.lookup(X, Y, Z, label) >= 0
            X, Y, Z, label = X[~present], Y[~present], Z[~present], label[~present]
        raise RuntimeError('column needs more than ranges_per_column=%d runs; raise it' % self.R)

    def _runs(self, X, Y, Z, label):
        if X.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.long, device=self.device)
        group = (X * self.shape[1] + Y) * self.label_space + label
        start = torch.ones_like(X, dtype=torch.bool)
        start[1:] = (group[1:] != group[:-1]) | (Z[1:] != Z[:-1] + 1)
        idx = torch.nonzero(start)[:, 0]
        length = torch.diff(torch.cat([idx, torch.tensor([X.numel()], device=self.device)]))
        return torch.stack([X[idx], Y[idx], label[idx], Z[idx], length], 1)

    def _overflow(self, runs):
        if runs.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        col = runs[:, 0] * self.shape[1] + runs[:, 1]
        uniq, inv, counts = torch.unique(col, return_inverse=True, return_counts=True)
        used = (self.ranges[runs[:, 0], runs[:, 1], :, 1] > 0).sum(1)
        return counts[inv] + used > self.R

    def _hull(self, X, Y, Z, label):
        group = (X * self.shape[1] + Y) * self.label_space + label
        uniq, inv = torch.unique(group, return_inverse=True)
        zmin = torch.full((len(uniq),), 1 << 30, dtype=torch.long, device=self.device).scatter_reduce(0, inv, Z, 'amin')
        zmax = torch.full((len(uniq),), -1, dtype=torch.long, device=self.device).scatter_reduce(0, inv, Z, 'amax')
        counts = zmax - zmin + 1
        rep = torch.repeat_interleave(torch.arange(len(uniq), device=self.device), counts)
        offs = torch.arange(int(counts.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        gX, gY, gZ, gl = self._decode(uniq * self.shape[2])
        return gX[rep], gY[rep], zmin[rep] + offs, gl[rep]

    def commit(self, runs, values):
        """Store `values` (one row per planned entry, in plan order) and index them."""
        n = values.shape[0]
        if n == 0:
            return
        if self.size + n > self.values.shape[0]:
            new = torch.zeros((max(2 * self.values.shape[0], self.size + n), self.channels),
                              dtype=self.values.dtype, device=self.device)
            new[:self.size] = self.values[:self.size]
            self.values = new
        self.values[self.size:self.size + n] = values.to(self.values.dtype)
        first = torch.cumsum(runs[:, 4], 0) - runs[:, 4]
        base = self.size + first
        col = runs[:, 0] * self.shape[1] + runs[:, 1]
        order = torch.argsort(col, stable=True)
        col, runs_s, base = col[order], runs[order], base[order]
        start = torch.ones_like(col, dtype=torch.bool)
        start[1:] = col[1:] != col[:-1]
        group_first = torch.cummax(torch.where(start, torch.arange(len(col), device=self.device), torch.zeros_like(col)), 0).values
        rank = torch.arange(len(col), device=self.device) - group_first
        used = (self.ranges[runs_s[:, 0], runs_s[:, 1], :, 1] > 0).sum(1)
        slot = used + rank
        if (slot >= self.R).any():
            raise RuntimeError('ranges_per_column exceeded at commit')
        entry = torch.stack([runs_s[:, 3], runs_s[:, 4], base, runs_s[:, 2]], 1).to(torch.int32)
        self.ranges[runs_s[:, 0], runs_s[:, 1], slot] = entry
        self.size += n

    def state_dict(self):
        return {'shape': self.shape, 'R': self.R, 'label_space': self.label_space,
                'ranges': self.ranges, 'values': self.values[:self.size].clone(), 'size': self.size}


# ------------------------------------------------------------ field sampling
def sample_field(table, world, labels, offset, s, piece=1 << 20):
    """Trilinear gather of the baked field at `world` [M,3] for `labels` [M].

    Returns ([M, C] float32, fraction of labelled samples with a missing corner).
    """
    nx, ny, nz = table.shape
    M = world.shape[0]
    u = (world.float() + offset) * s
    l0 = torch.floor(u)
    t = u - l0
    l0 = l0.long()
    ok = labels >= 0
    for a, n in enumerate((nx, ny, nz)):
        ok &= (l0[:, a] >= 0) & (l0[:, a] < n - 1)
    l0c = torch.stack([l0[:, a].clamp(0, n - 2) for a, n in enumerate((nx, ny, nz))], 1)
    key = ((l0c[:, 0] * ny + l0c[:, 1]) * nz + l0c[:, 2]) * table.label_space + labels.clamp(min=0)
    key = torch.where(ok, key, torch.full_like(key, -1))
    uniq, inv = torch.unique(key, return_inverse=True)
    lab = uniq % table.label_space
    rest = uniq // table.label_space
    cz, rest = rest % nz, rest // nz
    cy, cx = rest % ny, rest // ny
    corners = CORNERS.to(world.device)
    slots = table.lookup((cx[:, None] + corners[None, :, 0]).reshape(-1), (cy[:, None] + corners[None, :, 1]).reshape(-1),
                         (cz[:, None] + corners[None, :, 2]).reshape(-1), lab[:, None].expand(-1, 8).reshape(-1)).reshape(-1, 8)
    slots[uniq < 0] = MISSING
    out = torch.zeros((M, table.channels), dtype=torch.float32, device=world.device)
    missing = torch.zeros((), dtype=torch.long, device=world.device)
    for start in range(0, M, piece):
        sl = slice(start, start + piece)
        cs = slots[inv[sl]]                                              # [m, 8]
        v = table.gather(cs.reshape(-1)).reshape(cs.shape[0], 8, -1)     # [m, 8, C]
        tt = t[sl]
        w = torch.stack([
            (tt[:, 0] if c[0] else 1 - tt[:, 0]) * (tt[:, 1] if c[1] else 1 - tt[:, 1]) * (tt[:, 2] if c[2] else 1 - tt[:, 2])
            for c in corners.tolist()], 1)
        out[sl] = (v * w[..., None]).sum(1)
        missing += ((cs < 0).any(1) & ok[sl]).sum()
    return out, missing.float() / max(int(ok.sum()), 1)


def perpix_baked(model, table, label_fn, voxel_id, depth2, raydirs, cam_origin, offset, s, z=None):
    """Mirror of GanCraftGenerator._forward_perpix with the MLP replaced by the
    baked lattice. Sampler, weights, sky and output convention are the model's own."""
    cfg = model.cfg
    sky_mask = voxel_id[:, :, :, [-1], :] == 0
    sky_only_mask = voxel_id[:, :, :, [0], :] == 0
    rand_depth, new_dists, new_idx = model._sample_depth_batched(
        depth2, cfg.N_SAMPLE_POINTS_PER_RAY + 1, deterministic=True, use_box_boundaries=False, sample_depth=SAMPLE_DEPTH)
    rand_depth = torch.nan_to_num(rand_depth, nan=0., posinf=0., neginf=0.)
    world = raydirs * rand_depth + cam_origin[:, None, None, None, :]
    seg = torch.gather(voxel_id, -2, new_idx)[..., 0]
    labels = label_fn(seg)
    values, missing = sample_field(table, world.reshape(-1, 3), labels.reshape(-1), offset, s)
    values = values.reshape(*world.shape[:-1], -1)
    net_out_s, net_out_c = values[..., :1], values[..., 1:]
    weights = model._volum_rendering_relu(net_out_s, new_dists * cfg.DIST_SCALE, dim=-2)
    weights = weights * torch.logical_not(sky_only_mask).float()
    sky_weight, rgbs_sky = 0, 0
    if cfg.SKY_ENABLED:
        non_sky_mask = torch.logical_not(sky_mask).float()
        sky_weight = 1.0 - torch.sum(weights, dim=-2, keepdim=True)
        skynet_out_c = model.get_sky(raydirs)
        if cfg.SKY_GLOBAL_AVGPOOL:
            if model.sky_avg is not None:
                sky_avg = model.sky_avg
            else:
                sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                model.sky_avg = sky_avg
            skynet_out_c = skynet_out_c * (1.0 - non_sky_mask) + sky_avg * non_sky_mask
        else:
            sky_weight = sky_weight * (1.0 - non_sky_mask)
        rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
    rgbs = torch.clamp(net_out_c, -1, 1) + 1
    net_out = torch.sum(weights * rgbs, dim=-2, keepdim=True) + sky_weight * rgbs_sky
    net_out = net_out.squeeze(-2) - 1
    return net_out, torch.sum(weights, dim=-2).squeeze(dim=-1), missing


# -------------------------------------------------------------------- baker
class Baker:
    """Chunked baking of the background and building fields into one table.

    bg_field(points [n,3] world, onehot [n, n_bg]) -> (sigma [n,1], colour [n,C])
    building_field(b, points [n,3] world, onehot [n, n_bg]) -> same, evaluated in
    the building's own frame with its style (the caller owns that transform).
    """

    def __init__(self, volume, table, bg_field, building_field, inst, n_bg, s=1, offset=None,
                 thickness=SAMPLE_DEPTH + 1, chunk=256, batch=1 << 19, bg_on_buildings=True):
        self.volume, self.table = volume, table
        self.bg_field, self.building_field = bg_field, building_field
        self.inst, self.n_bg, self.s = inst, n_bg, s
        self.offset = torch.zeros(3, device=volume.device) if offset is None else offset.to(volume.device)
        self.thickness, self.chunk, self.batch = thickness, chunk, batch
        self.bg_on_buildings = bg_on_buildings
        self.baked = set()
        self.entries = 0
        nx, ny, nz = volume.shape
        self.grid = ((nx + chunk - 1) // chunk, (ny + chunk - 1) // chunk)

    def chunks_of_voxels(self, voxels):
        nx, ny, nz = self.volume.shape
        v = voxels[(voxels[:, 0] >= 0) & (voxels[:, 0] < nx) & (voxels[:, 1] >= 0) & (voxels[:, 1] < ny)]
        c = torch.unique(torch.div(v[:, :2], self.chunk, rounding_mode='floor'), dim=0)
        return {(int(a), int(b)) for a, b in c.tolist()}

    def ensure(self, voxels):
        """Bake every chunk a frame's samples fall in. Returns the chunks baked now."""
        needed = sorted(self.chunks_of_voxels(voxels) - self.baked)
        for c in needed:
            self.bake_chunk(*c)
        return needed

    def bake_chunk(self, cx, cy):
        if (cx, cy) in self.baked:
            return 0
        nx, ny, nz = self.volume.shape
        x0, x1 = cx * self.chunk, min((cx + 1) * self.chunk, nx)
        y0, y1 = cy * self.chunk, min((cy + 1) * self.chunk, ny)
        h = self.thickness
        hx0, hx1, hy0, hy1 = max(0, x0 - h), min(nx, x1 + h), max(0, y0 - h), min(ny, y1 + h)
        vol = self.volume[hx0:hx1, hy0:hy1]
        solid = vol != 0
        shell = shell_voxels(solid, self.thickness)
        own = torch.zeros_like(shell)
        own[x0 - hx0:x1 - hx0, y0 - hy0:y1 - hy0] = True
        shell &= own
        vox = torch.nonzero(shell)
        ids = vol[shell].long()
        vox = vox + torch.tensor([hx0, hy0, 0], device=vox.device)
        labels, points = [], []
        bg_class = background_labels(ids, self.inst)
        building, is_roof = building_of(ids, self.inst)
        keep_bg = torch.ones_like(ids, dtype=torch.bool) if self.bg_on_buildings else building == 0
        p, l = lattice_points(vox[keep_bg], bg_class[keep_bg], self.s)
        points.append(p); labels.append(l)
        has_b = building > 0
        if has_b.any():
            p, l = lattice_points(vox[has_b], building_label(building[has_b], is_roof[has_b], self.n_bg), self.s)
            points.append(p); labels.append(l)
        points, labels = torch.cat(points), torch.cat(labels)
        entries, runs = self.table.plan(points[:, 0], points[:, 1], points[:, 2], labels)
        values = self._evaluate(entries)
        self.table.commit(runs, values)
        self.baked.add((cx, cy))
        self.entries += entries.shape[0]
        return entries.shape[0]

    def _evaluate(self, entries):
        n = entries.shape[0]
        out = torch.zeros((n, self.table.channels), dtype=torch.float32, device=entries.device)
        if n == 0:
            return out
        world = entries[:, :3].float() / self.s - self.offset
        label = entries[:, 3]
        is_bg = label < self.n_bg
        idx = torch.nonzero(is_bg)[:, 0]
        for start in range(0, idx.numel(), self.batch):
            i = idx[start:start + self.batch]
            onehot = F.one_hot(label[i], self.n_bg).float()
            sigma, colour = self.bg_field(world[i], onehot)
            out[i] = torch.cat([sigma, colour], -1)
        b_idx = torch.nonzero(~is_bg)[:, 0]
        if b_idx.numel():
            code = label[b_idx] - self.n_bg
            for b in torch.unique(code // 2).tolist():
                sel = b_idx[code // 2 == b]
                is_roof = (label[sel] - self.n_bg) % 2 == 1
                cls = torch.where(is_roof, torch.full_like(sel, self.inst['ROOF_CID']),
                                  torch.full_like(sel, self.inst['FACADE_CID']))
                for start in range(0, sel.numel(), self.batch):
                    i = sel[start:start + self.batch]
                    onehot = F.one_hot(cls[start:start + self.batch], self.n_bg).float()
                    sigma, colour = self.building_field(b, world[i], onehot)
                    out[i] = torch.cat([sigma, colour], -1)
        return out


# ------------------------------------------------------------------ renderer
def building_label_fn(b, inst, n_bg):
    facade, roof = inst['FACADE_CID'], inst['ROOF_CID']
    def fn(seg):
        lab = torch.full_like(seg, MISSING, dtype=torch.long)
        lab[seg == facade] = n_bg + 2 * b
        lab[seg == roof] = n_bg + 2 * b + 1
        return lab
    return fn


def render_frame(bg, bldg, table, rays, inst, styles, offset, s, n_bg, padding=8, blurrer=None, stats=None):
    """Full-frame render from the baked field: background pass, native road blur,
    then per visible building the ROI pass with its style, composited on first-hit
    masks as native_balanced_v4 does. Returns ([1,3,H,W], info)."""
    voxel_id, depth2, raydirs, origin = rays
    H, W = voxel_id.shape[1:3]
    info = {'missing_bg': 0., 'missing_bldg': [], 'buildings': 0}
    bg_ids = background_labels(voxel_id, inst)
    feat, _, miss = perpix_baked(bg, table, lambda seg: seg.long(), bg_ids, depth2, raydirs, origin, offset, s)
    info['missing_bg'] = float(miss)
    rgb = bg._forward_global(feat, None)
    if blurrer is not None:
        road = (bg_ids[:, None, :, :, 0, 0] == inst['ROAD_CID']).float()
        rgb = blurrer(rgb) * road + rgb * (1 - road)
    rgb = rgb.clone()
    first = voxel_id[0, :, :, 0, 0]
    for b in visible_buildings(first, inst).tolist():
        if b not in styles or (stats is not None and b not in stats):
            continue
        roof = b + inst['ROOF_OFFSET']
        hit = (first == b) | (first == roof)
        ys, xs = torch.nonzero(hit, as_tuple=True)
        if ys.numel() == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        ax, bx, ay, by = max(0, x0 - padding), min(W, x1 + padding), max(0, y0 - padding), min(H, y1 + padding)
        ids = voxel_id[:, ay:by, ax:bx]
        bid = torch.zeros_like(ids)
        bid[ids == b] = inst['FACADE_CID']
        bid[ids == roof] = inst['ROOF_CID']
        dirs = raydirs[:, ay:by, ax:bx].clone()
        dirs[bid[..., 0, 0] == 0] = 0
        feat_b, _, miss = perpix_baked(bldg, table, building_label_fn(b, inst, n_bg), bid, depth2[:, ay:by, ax:bx],
                                       dirs, origin, offset, s, z=styles[b])
        out = bldg._forward_global(feat_b, styles[b])
        mask = hit[ay:by, ax:bx][None, None].float()
        rgb[:, :, ay:by, ax:bx] = rgb[:, :, ay:by, ax:bx] * (1 - mask) + out * mask
        info['missing_bldg'].append(float(miss))
        info['buildings'] += 1
    return rgb, info

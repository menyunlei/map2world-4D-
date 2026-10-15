"""One candidate: bind preserved fine map boundaries to the current local state.

Both arms receive IDENTICAL ordered map features and have identical parameters.
The control receives a shared, map-valid global state query; the candidate uses
the corresponding local hidden state. Original G2-B computations are untouched.
This is not local-Q attention, output carving, or a decoder modification.
"""
import torch
from torch import nn
from subcell_model_v1 import spatial_features


class QueryBoundaryRead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ('global', 'local', 'misaligned')
        self.mode = mode
        self.map_encode = nn.Sequential(nn.Linear(48, 128), nn.SiLU())
        self.query_encode = nn.Sequential(nn.Linear(3072, 128), nn.SiLU())
        self.interact = nn.Sequential(nn.Linear(384, 128), nn.SiLU())
        self.write = nn.ModuleList([nn.Linear(128, 3072) for _ in range(6)])
        for layer in self.write:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, hidden, mask, site, dtype):
        flat = hidden.reshape(-1, 3072).float()
        valid = mask.float()[:, None]
        assert features.shape == (len(flat), 48) and valid.shape == (len(flat), 1)
        if self.mode == 'global':
            query = (flat * valid).sum(0, keepdim=True) / valid.sum().clamp_min(1)
            query = query.expand_as(flat)
        elif self.mode == 'misaligned':
            # Fixed inference-only spatial reassociation. Native hidden tokens,
            # old G2-B paths, features, gates, weights and noise are not modified.
            query = flat.roll(17, 0)
        else:
            query = flat
        mapped = self.map_encode(features.float())
        state = self.query_encode(query)
        joint = self.interact(torch.cat((mapped, state, mapped * state), -1))
        return (self.write[site](joint) * valid).to(dtype).reshape_as(hidden)


def attach(model, mode):
    from typed_write import _bounding_boxes
    from hierarchy_model_v2 import gather_grid
    branch = model.branch
    old_prepare, old_read = branch.prepare, branch.read
    branch.query_boundary = QueryBoundaryRead(mode).to(next(model.parameters()).device)

    def prepare(encoder, raster, xyz, old, relation):
        result = old_prepare(encoder, raster, xyz, old, relation)
        masks = relation['masks'].float() * relation['valid'][:, None, None].float()
        footprints = relation['roles'][:, 1].float() * relation['valid'].float()
        fill = (masks * footprints[:, None, None]).sum(0).clamp(0, 1)
        void = (masks * (relation['roles'][:, 3].float() * relation['valid'].float())[:, None, None]).sum(0).clamp(0, 1)
        boxes = _bounding_boxes(masks, footprints).sum(0).clamp(0, 1)
        channels = torch.stack(((fill - void).clamp_min(0), void, (boxes - fill).clamp_min(0)))[None]
        features = spatial_features(channels, xyz, 'correct', gather_grid)
        support = (result[5]['gates'].sum(-1) > 0).float()
        assert torch.allclose(features.reshape(-1, 3, 16).mean(-1) * support[:, None],
                              result[5]['gates'], atol=1e-6), 'FINE_MAP_ADDRESS_MISMATCH'
        return (*result, features, support)

    def read(hidden, result, site):
        return old_read(hidden, result[:6], site) + branch.query_boundary(
            result[6], hidden, result[7], site, hidden.dtype)

    branch.prepare, branch.read = prepare, read


"""Additive map conditioning for the audited dense TRELLIS.2 structure flow.

The native contract is x [B, 8, 16, 16, 16], flattened with
``x.view(B, 8, -1).permute(0, 2, 1)``: X outermost, then Y, Z fastest.
Each of the 30 native blocks returns h [B, 4096, 1536]. The caller freezes
the native model and adds this branch *after* blocks 4, 14, and 24:

    context = conditioner.prepare(mapbatch, native_world_grid())
    # Within the caller's existing native block loop:
    h = block(h, ...)  # Existing native arguments and behavior.
    h = conditioner.add_to(h, context, block_index)

No native parameters, hooks, normalization, output placement, or pruning
are owned by this module. Z is a query coordinate, never an observed height.

Required mapbatch tensors (support and confidence values lie in [0, 1]):
  footprints, footprint_known: [B, O, 64, 64]
  roads, roads_known, open_space, open_space_known: [B, 1, 64, 64]
  object_summary: [B, O, 8], ordered as centroidX/Y in [-.5, .5],
      extentX/Y in [0, 1], area, knownXYconfidence, heightKnown=0,
      classBuilding=1.
  object_valid: bool [B, O] (the external objectValid mask).
Optional mapYrow is 'increasing' (default) or 'decreasing'; columns always
increase with world X. IDs and other extra fields are external metadata
and are ignored. The world cube is fixed at [-.5, .5], without scene scaling.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


GRID_SIZE = 16
TOKEN_COUNT = GRID_SIZE ** 3
MODEL_CHANNELS = 1536
BLOCK_COUNT = 30
INJECTION_BLOCKS = (4, 14, 24)
OBJECT_CHANNELS = 128


def native_world_grid(*, device=None, dtype=torch.float32) -> Tensor:
    """Cell centers in the native ij meshgrid / dense flatten order [4096, 3]."""
    axis = (torch.arange(GRID_SIZE, device=device, dtype=dtype) + 0.5) / GRID_SIZE - 0.5
    xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack(xyz, dim=-1).reshape(TOKEN_COUNT, 3)


@dataclass(frozen=True)
class MapObjectContext:
    """Per-forward context; recompute after changing inputs or branch weights.

    token_features: [B, 4096, 128]; object_tokens: [B, O, 128];
    routing_weights: [B, 4096, O], confidence-weighted average coverage.
    The context retains autograd history; it is not a persistent cache.
    """

    token_features: Tensor
    object_tokens: Tensor
    routing_weights: Tensor


class Trellis2MapObjectConditioner(nn.Module):
    """One shared object branch and one shared zero-initialized output Wo."""

    def __init__(self, *, mapYrow: str = "increasing"):
        super().__init__()
        self._check_rows(mapYrow)
        self.mapYrow = mapYrow
        self.object_mlp = nn.Sequential(
            nn.Linear(8, OBJECT_CHANNELS), nn.SiLU(),
            nn.Linear(OBJECT_CHANNELS, OBJECT_CHANNELS),
        )
        self.object_attention = nn.MultiheadAttention(
            OBJECT_CHANNELS, num_heads=4, dropout=0.0, batch_first=True,
        )
        # Routed object (128), relative geometry (7), query XYZ (3),
        # footprint support/known, road support/known, open support/known (6).
        self.token_mlp = nn.Sequential(
            nn.Linear(144, OBJECT_CHANNELS), nn.SiLU(),
            nn.Linear(OBJECT_CHANNELS, OBJECT_CHANNELS),
        )
        self.Wo = nn.Linear(OBJECT_CHANNELS, MODEL_CHANNELS, bias=False)
        nn.init.zeros_(self.Wo.weight)

    @staticmethod
    def _check_rows(convention):
        if convention not in ("increasing", "decreasing"):
            raise ValueError("mapYrow must be 'increasing' or 'decreasing'")

    def _tensor(self, batch, key, shape, *, unit_interval=False):
        value = batch[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must have shape {shape}")
        value = value.to(device=self.Wo.weight.device, dtype=self.Wo.weight.dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} must be finite")
        if unit_interval and ((value < 0) | (value > 1)).any():
            raise ValueError(f"{key} must lie in [0, 1]")
        return value

    def prepare(self, mapbatch: Mapping, grid: Tensor) -> MapObjectContext:
        """Route whole-cell map coverage to native tokens without reading h.

        grid is [4096, 3] or [16, 16, 16, 3] in the caller's token order,
        already expressed in the fixed world cube. No per-object Z extent
        is inferred. All masks are required; absent masks never mean known.
        """
        footprints = mapbatch["footprints"]
        if not isinstance(footprints, Tensor) or footprints.ndim != 4:
            raise ValueError("footprints must have shape [B, O, 64, 64]")
        batch_size, objects = footprints.shape[:2]
        if batch_size == 0:
            raise ValueError("batch size must be positive")
        object_shape = (batch_size, objects, 64, 64)
        map_shape = (batch_size, 1, 64, 64)
        rasters = {
            key: self._tensor(mapbatch, key, shape, unit_interval=True)
            for key, shape in (
                ("footprints", object_shape), ("footprint_known", object_shape),
                ("roads", map_shape), ("roads_known", map_shape),
                ("open_space", map_shape), ("open_space_known", map_shape),
            )
        }
        rows = mapbatch.get("mapYrow", self.mapYrow)
        self._check_rows(rows)
        if rows == "decreasing":
            rasters = {key: value.flip(-2) for key, value in rasters.items()}

        valid = mapbatch["object_valid"]
        if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
                or tuple(valid.shape) != (batch_size, objects)):
            raise ValueError("object_valid must be bool [B, O]")
        valid = valid.to(device=self.Wo.weight.device)
        summary = mapbatch["object_summary"]
        if not isinstance(summary, Tensor) or tuple(summary.shape) != (batch_size, objects, 8):
            raise ValueError("object_summary must have shape [B, O, 8]")
        summary = summary.to(device=self.Wo.weight.device, dtype=self.Wo.weight.dtype)
        # Padded summaries may contain sentinels; exclude them before any MLP.
        summary = torch.where(valid[..., None], summary, torch.zeros_like(summary))
        if not torch.isfinite(summary).all():
            raise ValueError("valid object summaries must be finite")
        observed = summary[valid]
        if ((observed[:, :2] < -0.5) | (observed[:, :2] > 0.5)).any():
            raise ValueError("XY centroids must be world-normalized to [-.5, .5]")
        if ((observed[:, 2:6] < 0) | (observed[:, 2:6] > 1)).any():
            raise ValueError("extents, area, and knownXYconfidence must lie in [0, 1]")
        if (observed[:, 6] != 0).any() or (observed[:, 7] != 1).any():
            raise ValueError("valid summaries require heightKnown=0 and classBuilding=1")

        if not isinstance(grid, Tensor) or tuple(grid.shape) not in (
            (TOKEN_COUNT, 3), (GRID_SIZE, GRID_SIZE, GRID_SIZE, 3),
        ):
            raise ValueError("grid must be [4096, 3] or [16, 16, 16, 3]")
        grid = grid.reshape(TOKEN_COUNT, 3).to(
            device=self.Wo.weight.device, dtype=self.Wo.weight.dtype,
        )
        if not torch.isfinite(grid).all() or ((grid < -0.5) | (grid > 0.5)).any():
            raise ValueError("grid coordinates must lie in the fixed [-.5, .5] cube")
        cell_xy = ((grid[:, :2] + 0.5) * GRID_SIZE).floor().long().clamp(0, GRID_SIZE - 1)
        # Pooled maps are [row=Y, col=X]; native tensors are [X, Y, Z].
        map_index = cell_xy[:, 1] * GRID_SIZE + cell_xy[:, 0]

        def pool_to_tokens(value):
            if value.shape[1] == 0:
                return value.new_zeros(batch_size, TOKEN_COUNT, 0)
            # Every one of the 16 source pixels contributes, including narrow
            # support that misses a coarse-cell center. Unknown pixels contribute 0.
            pooled = F.avg_pool2d(value, kernel_size=4, stride=4)
            return pooled.flatten(2).index_select(2, map_index).transpose(1, 2)

        coverage = pool_to_tokens(rasters["footprints"] * rasters["footprint_known"])
        footprint_known = pool_to_tokens(rasters["footprint_known"])
        confidence = summary[..., 5] * valid
        weights = coverage * confidence[:, None, :]
        denominator = weights.sum(-1, keepdim=True).clamp_min(1.0)
        # Keeping denominator >= 1 retains fractional coverage for small objects.
        if objects:
            active = valid & (confidence > 0)
            object_tokens = self.object_mlp(summary) * confidence[..., None]
            safe_active = active.clone()
            # MHA cannot softmax an all-masked row. A zero dummy key is used
            # only for empty rows, and its output is masked away immediately.
            safe_active[:, 0] |= ~active.any(-1)
            interaction, _ = self.object_attention(
                object_tokens, object_tokens, object_tokens,
                key_padding_mask=~safe_active, need_weights=False,
            )
            object_tokens = (object_tokens + interaction) * active[..., None]
            routed_objects = torch.bmm(weights, object_tokens) / denominator
            relative_xy = grid[None, :, None, :2] - summary[:, None, :, :2]
            scaled_xy = relative_xy / summary[:, None, :, 2:4].clamp_min(1.0 / GRID_SIZE)
            distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
            relative_features = torch.cat((
                relative_xy, scaled_xy, distance, coverage[..., None],
                footprint_known[..., None],
            ), dim=-1)
            routed_geometry = (relative_features * weights[..., None]).sum(2) / denominator
            object_known = (footprint_known * confidence[:, None, :]).amax(-1, keepdim=True)
        else:
            object_tokens = grid.new_zeros(batch_size, 0, OBJECT_CHANNELS)
            routed_objects = grid.new_zeros(batch_size, TOKEN_COUNT, OBJECT_CHANNELS)
            routed_geometry = grid.new_zeros(batch_size, TOKEN_COUNT, 7)
            object_known = grid.new_zeros(batch_size, TOKEN_COUNT, 1)

        road = pool_to_tokens(rasters["roads"] * rasters["roads_known"])
        road_known = pool_to_tokens(rasters["roads_known"])
        open_space = pool_to_tokens(rasters["open_space"] * rasters["open_space_known"])
        open_known = pool_to_tokens(rasters["open_space_known"])
        local_maps = torch.cat((
            weights.sum(-1, keepdim=True).clamp_max(1.0), object_known,
            road, road_known, open_space, open_known,
        ), dim=-1)
        features = torch.cat((
            routed_objects, routed_geometry,
            grid[None].expand(batch_size, -1, -1), local_maps,
        ), dim=-1)
        evidence = torch.maximum(object_known, torch.maximum(road_known, open_known))
        token_features = self.token_mlp(features) * evidence
        return MapObjectContext(token_features, object_tokens, weights)

    def residual(self, context: MapObjectContext, block_index: int):
        """Return [B, 4096, 1536] at the three taps, or None elsewhere."""
        if not isinstance(block_index, int) or not 0 <= block_index < BLOCK_COUNT:
            raise ValueError("block_index must be a 0-based integer in [0, 29]")
        if block_index not in INJECTION_BLOCKS:
            return None
        return self.Wo(context.token_features)

    def add_to(self, h: Tensor, context: MapObjectContext, block_index: int) -> Tensor:
        """Add to a block's output without modifying h or native parameters."""
        expected = (context.token_features.shape[0], TOKEN_COUNT, MODEL_CHANNELS)
        if tuple(h.shape) != expected or not h.is_floating_point():
            raise ValueError(f"native h must be floating point with shape {expected}")
        if h.device != context.token_features.device:
            raise ValueError("native h and context must share a device")
        delta = self.residual(context, block_index)
        return h if delta is None else h + delta.to(dtype=h.dtype)

"""Two object-generation contracts using exactly the v3 parameters.

Main prepares ONE complete map square64 per output object, then batches these
maps as B. In each square, footprints/known are [B, O, 64, 64], including the
same neighbors in BOTH modes; roads/open-space and their required known masks
are [B, 1, 64, 64]. object_summary [B, O, 8] is expressed in that output
object's canonical frame. Main's object_valid mask excludes summaries whose
XY leaves [-.5, .5]; this module never clamps or recenters summaries. The
caller supplies the corresponding canonical native XYZ grid, in v3 order.
Z remains a query, never observed geometry. External numeric IDs are ignored.

    conditioner = ObjectContractConditioner(reading_mode="object_grounded")
    # canonical_mapbatch['target_object'] is int64 [B]; ordinary uses it too.
    context = conditioner.prepare(canonical_mapbatch, canonical_grid)
    condition = conditioner(context)  # [B, 256, 1024], for BOTH native flows
    # Within the existing SS block loop, at the unchanged taps 4/14/24:
    if condition.context is not None:
        h = conditioner.add_to(h, condition.context, block_index)

Both modes have the same input/output contract and preserve v3's object
MLP/MHA, token MLP, map head, Wo, state keys, and 611584 parameter count.
Ordinary keeps v3's all-object routing and local map features. Grounded uses
only the selected object's coverage, descriptor, and relative geometry for
the local object channels. Its descriptor still contains the existing MHA
summary over all active object summaries; neighbors' spatial rasters cannot
contribute to the target's object channels. Shared road/open channels remain.

The fixed contract in BOTH modes keeps the PRE-evidence-gate token-MLP output:
unknown content does not invalidate a canonical query. Known masks still
exclude hidden raster values. No image geometry, new trainable modules,
normalization, losses, native model changes, or canonicalization is added.
Negative CFG retains v3's exact zeros/context=None (skip SS residuals).
"""

from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from trellis2_map_geometry_token_module_v3 import (
    MapGeometryTokenCondition, Trellis2MapGeometryTokenConditioner,
)
from trellis2_map_object_module_v1 import (
    GRID_SIZE, OBJECT_CHANNELS, MapObjectContext,
)


class ObjectContractConditioner(Trellis2MapGeometryTokenConditioner):
    """Add only routing behavior; all trainable modules/parameters are v3's."""

    def __init__(
        self,
        reading_mode: Literal["ordinary", "object_grounded"] = "ordinary",
        *,
        mapYrow: str = "increasing",
    ):
        if reading_mode not in ("ordinary", "object_grounded"):
            raise ValueError("reading_mode must be 'ordinary' or 'object_grounded'")
        super().__init__(mapYrow=mapYrow)
        self.reading_mode = reading_mode

    def _target_indices(self, mapbatch: Mapping, target_object: int | Tensor) -> Tensor:
        footprints = mapbatch["footprints"]
        if not isinstance(footprints, Tensor) or footprints.ndim != 4:
            raise ValueError("footprints must have shape [B, O, 64, 64]")
        batch_size, objects = footprints.shape[:2]
        valid = mapbatch["object_valid"]
        if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
                or tuple(valid.shape) != (batch_size, objects)):
            raise ValueError("object_valid must be bool [B, O]")
        device = self.Wo.weight.device
        if (isinstance(target_object, Tensor)
              and target_object.dtype == torch.int64
              and tuple(target_object.shape) == (batch_size,)):
            indices = target_object.to(device=device, dtype=torch.long)
        else:
            raise ValueError("target_object must be an int64 [B] tensor of object slots")
        if batch_size == 0 or objects == 0 or ((indices < 0) | (indices >= objects)).any():
            raise ValueError("target_object must index an existing object in each batch row")
        rows = torch.arange(batch_size, device=device)
        if not valid.to(device=device)[rows, indices].all():
            raise ValueError("target_object must select an object_valid slot")
        return indices

    def prepare(
        self,
        mapbatch: Mapping,
        grid: Tensor,
    ) -> MapObjectContext:
        """Prepare identical canonical inputs with either local object binding.

        mapbatch['target_object'] is int64 [B], selecting one slot per row.
        The index is a slot, not an external object ID. A valid zero-confidence
        target is allowed. All v3 raster/summary/grid checks still apply.
        grid is [4096, 3] or [16, 16, 16, 3], as in v3: canonical maps across
        B use the same normalized query lattice, each in its own local frame.
        """
        if self.reading_mode not in ("ordinary", "object_grounded"):
            raise ValueError("reading_mode must be 'ordinary' or 'object_grounded'")
        targets = self._target_indices(mapbatch, mapbatch["target_object"])
        captured = []

        def capture_pre_gate(_module, args, output):
            # Retain autograd, without modifying either the input or output.
            captured.append((args[0], output))

        hook = self.token_mlp.register_forward_hook(capture_pre_gate)
        try:
            original = super().prepare(mapbatch, grid)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("v3 prepare must evaluate the existing token MLP exactly once")
        ordinary_input, pre_gate = captured[0]
        if self.reading_mode == "ordinary":
            return MapObjectContext(pre_gate, original.object_tokens, original.routing_weights)

        rows = torch.arange(targets.shape[0], device=targets.device)
        summary = mapbatch["object_summary"].to(
            device=pre_gate.device, dtype=pre_gate.dtype,
        )[rows, targets]
        # Reuse v3's validated queries and already masked shared map channels.
        query_xyz = ordinary_input[..., OBJECT_CHANNELS + 7:OBJECT_CHANNELS + 10]
        cell_xy = ((query_xyz[0, :, :2] + .5) * GRID_SIZE).floor().long().clamp(
            0, GRID_SIZE - 1,
        )
        map_index = cell_xy[:, 1] * GRID_SIZE + cell_xy[:, 0]

        def target_raster(key):
            value = mapbatch[key].to(device=pre_gate.device, dtype=pre_gate.dtype)[rows, targets]
            if mapbatch.get("mapYrow", self.mapYrow) == "decreasing":
                value = value.flip(-2)
            return value

        def pool_to_queries(value):
            pooled = F.avg_pool2d(value[:, None], kernel_size=4, stride=4)
            return pooled.flatten(1).index_select(1, map_index)[..., None]

        known_raster = target_raster("footprint_known")
        coverage = pool_to_queries(target_raster("footprints") * known_raster)
        footprint_known = pool_to_queries(known_raster)
        weight = original.routing_weights[rows, :, targets][..., None]
        denominator = weight.clamp_min(1.0)  # Same fractional-coverage rule as v3.
        # The selected token already includes v3's cross-object MHA summary.
        routed_object = original.object_tokens[rows, targets][:, None] * weight / denominator
        relative_xy = query_xyz[..., :2] - summary[:, None, :2]
        scaled_xy = relative_xy / summary[:, None, 2:4].clamp_min(1.0 / GRID_SIZE)
        relative = torch.cat((
            relative_xy, scaled_xy,
            torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True),
            coverage, footprint_known,
        ), dim=-1)
        routed_geometry = relative * weight / denominator
        local_maps = torch.cat((
            weight.clamp_max(1.0), footprint_known * summary[:, None, 5:6],
            ordinary_input[..., -4:],  # Road/known and open/known, unchanged.
        ), dim=-1)
        target_input = torch.cat((routed_object, routed_geometry, query_xyz, local_maps), dim=-1)
        # Apply the SAME pre-gate computation as ordinary, with target bindings.
        token_features = self.token_mlp(target_input)
        slots = torch.arange(original.object_tokens.shape[1], device=targets.device)
        routing_weights = original.routing_weights * (slots[None, None] == targets[:, None, None])
        return MapObjectContext(token_features, original.object_tokens, routing_weights)

    def forward(
        self,
        context: MapObjectContext,
        image_tokens: Tensor | None = None,
        *,
        arm: Literal["separated"] = "separated",
        negative_cfg: bool = False,
    ) -> MapGeometryTokenCondition:
        """Keep the v3 separated interface; image_tokens is ignored entirely."""
        if arm != "separated":
            raise ValueError("v6 object geometry requires arm='separated'")
        return super().forward(context, image_tokens, arm=arm, negative_cfg=negative_cfg)

"""Shared map input tokens for native SS and fine ShapeFlow cross-attention.

Only the condition features are extended. Native blocks, normalization, and
the inherited prepare/add_to/Wo path (SS taps 4, 14, 24) are unchanged.
Contexts must be prepared with native_world_grid(): X outermost, Y next,
Z fastest. Pooling Z does not observe or infer object heights. External IDs
and the base class's required known/unknown masks retain their semantics.

    context = conditioner.prepare(mapbatch, native_world_grid())
    condition = conditioner(context, arm="separated")
    # Pass condition.features to BOTH native SS and fine ShapeFlow attention.
    # Only SS uses the existing post-block residual:
    if condition.context is not None:
        h = conditioner.add_to(h, condition.context, block_index)

The reference arm concatenates unmodified image tokens before map tokens.
The separated arm never reads image_tokens, including their shape or values.
For negative CFG, features are zeros and context is None: skip map residuals.
This module does not own native model integration or training target encoders;
using it does not establish availability of legitimate fine TRAIN targets.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from trellis2_map_object_module_v1 import (
    GRID_SIZE, OBJECT_CHANNELS, TOKEN_COUNT,
    MapObjectContext, Trellis2MapObjectConditioner,
)


MAP_CONDITION_TOKEN_COUNT = GRID_SIZE ** 2
CONDITION_CHANNELS = 1024


@dataclass(frozen=True)
class MapGeometryTokenCondition:
    """Features for both flows; optional context for inherited SS residuals.

    features is [B, 256, 1024] for separated, [B, L_image+256, 1024]
    for reference. context=None explicitly disables map residuals for CFG.
    Recompute this per forward after changing inputs or branch weights.
    """

    features: Tensor
    context: MapObjectContext | None


class Trellis2MapGeometryTokenConditioner(Trellis2MapObjectConditioner):
    """One trainable 128->256->1024 head shared by SS and fine ShapeFlow.

    Both linear biases start at zero. Xavier uniform gain=1 for the first
    weight, gain=0.1 for the second keeps initial condition amplitudes modest
    while allowing gradients through both layers immediately. The head is
    nonzero; although inherited Wo starts at zero, cross-attention with these
    extra input tokens is NOT guaranteed to match an unconditioned backbone.
    No normalization or modification of native hidden states is introduced.
    """

    def __init__(self, *, mapYrow: str = "increasing"):
        super().__init__(mapYrow=mapYrow)
        self.map_condition_head = nn.Sequential(
            nn.Linear(OBJECT_CHANNELS, 256),
            nn.SiLU(),
            nn.Linear(256, CONDITION_CHANNELS),
        )
        nn.init.xavier_uniform_(self.map_condition_head[0].weight, gain=1.0)
        nn.init.xavier_uniform_(self.map_condition_head[2].weight, gain=0.1)
        nn.init.zeros_(self.map_condition_head[0].bias)
        nn.init.zeros_(self.map_condition_head[2].bias)

    def _check_context(self, context: MapObjectContext) -> Tensor:
        if not isinstance(context, MapObjectContext):
            raise ValueError("context must be a prepared MapObjectContext")
        features = context.token_features
        if (features.ndim != 3 or features.shape[0] == 0
                or tuple(features.shape[1:]) != (TOKEN_COUNT, OBJECT_CHANNELS)
                or not features.is_floating_point()):
            raise ValueError("context.token_features must be floating [B, 4096, 128]")
        weight = self.map_condition_head[0].weight
        if features.device != weight.device or features.dtype != weight.dtype:
            raise ValueError("context and map head must share device and dtype")
        return features

    def map_tokens(self, context: MapObjectContext) -> Tensor:
        """Mean Z in native XYZ order, then head -> [B, X*16+Y, 1024]."""
        features = self._check_context(context)
        batch_size = features.shape[0]
        pooled = features.reshape(
            batch_size, GRID_SIZE, GRID_SIZE, GRID_SIZE, OBJECT_CHANNELS,
        ).mean(dim=3).reshape(batch_size, MAP_CONDITION_TOKEN_COUNT, OBJECT_CHANNELS)
        return self.map_condition_head(pooled)

    def forward(
        self,
        context: MapObjectContext,
        image_tokens: Tensor | None = None,
        *,
        arm: Literal["reference", "separated"] = "separated",
        negative_cfg: bool = False,
    ) -> MapGeometryTokenCondition:
        """Build native condition inputs without touching either backbone.

        For reference, image_tokens is floating [B, L, 1024] on the context
        device. Only map tokens are cast to its dtype; image values are kept
        exactly as supplied. Separated ignores image_tokens entirely.
        Negative CFG bypasses the head and returns zeros even after training
        its biases; the returned context=None must be used to skip SS add_to.
        """
        if arm not in ("reference", "separated"):
            raise ValueError("arm must be 'reference' or 'separated'")
        if not isinstance(negative_cfg, bool):
            raise ValueError("negative_cfg must be bool")
        features = self._check_context(context)
        batch_size = features.shape[0]
        if arm == "reference":
            if (not isinstance(image_tokens, Tensor) or image_tokens.ndim != 3
                    or image_tokens.shape[0] != batch_size
                    or image_tokens.shape[2] != CONDITION_CHANNELS
                    or not image_tokens.is_floating_point()):
                raise ValueError("reference image_tokens must be floating [B, L, 1024]")
            if image_tokens.device != features.device:
                raise ValueError("reference image_tokens and context must share a device")
            if negative_cfg:
                return MapGeometryTokenCondition(image_tokens.new_zeros(
                    batch_size, image_tokens.shape[1] + MAP_CONDITION_TOKEN_COUNT,
                    CONDITION_CHANNELS,
                ), None)
            tokens = self.map_tokens(context).to(dtype=image_tokens.dtype)
            return MapGeometryTokenCondition(torch.cat((image_tokens, tokens), dim=1), context)
        if negative_cfg:
            return MapGeometryTokenCondition(features.new_zeros(
                batch_size, MAP_CONDITION_TOKEN_COUNT, CONDITION_CHANNELS,
            ), None)
        return MapGeometryTokenCondition(self.map_tokens(context), context)

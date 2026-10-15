"""Fixed, additive LoRA for TRELLIS.2 structure cross-attention.

Load the native checkpoint, place/cast the model, and freeze it BEFORE calling
``attach_lora(native)``. Only blocks 4, 14, 24 and their to_q/to_kv/to_out
projections are wrapped. Existing projection parameters move under ``.base``;
their objects and values are preserved. No native forward method is patched.

``trainable_parameters(native)`` supplies the optimizer parameters.
``lora_state_dict(native)`` exports detached CPU copies of only A/B, and
``load_lora_state_dict(native, state)`` strictly restores those 18 tensors.
The native model's ordinary state_dict still includes its frozen base weights.
"""

import math
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LORA_BLOCKS = (4, 14, 24)
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
PROJECTION_SHAPES = {
    "to_q": (1536, 1536),
    "to_kv": (1024, 3072),
    "to_out": (1536, 1536),
}


class LoRALinear(nn.Module):
    """Frozen Linear plus (alpha / rank) * B(A(x)), computed in FP32."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires a torch.nn.Linear base")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = LORA_RANK
        self.alpha = LORA_ALPHA
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(LORA_DROPOUT)
        self.A = nn.Parameter(torch.empty(
            self.rank, base.in_features, device=base.weight.device,
            dtype=torch.float32,
        ), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(
            base.out_features, self.rank, device=base.weight.device,
            dtype=torch.float32,
        ), requires_grad=True)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.train(base.training)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        # Explicitly disable ambient autocast for both residual matmuls. Keep
        # base autograd enabled so gradients can reach upstream conditioners.
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(
                F.linear(self.dropout(x.float()), self.A.float()), self.B.float(),
            ) * self.scaling
        return output + residual.to(dtype=output.dtype)


def _projections(native: nn.Module):
    blocks = getattr(native, "blocks", None)
    if blocks is None or len(blocks) <= max(LORA_BLOCKS):
        raise ValueError("native.blocks must contain indices 4, 14, and 24")
    for index in LORA_BLOCKS:
        attention = getattr(blocks[index], "cross_attn", None)
        if attention is None:
            raise ValueError(f"blocks.{index} must have cross_attn")
        for name, shape in PROJECTION_SHAPES.items():
            path = f"blocks.{index}.cross_attn.{name}"
            projection = getattr(attention, name, None)
            if not isinstance(projection, (nn.Linear, LoRALinear)):
                raise TypeError(f"{path} must be a torch.nn.Linear projection")
            if (projection.in_features, projection.out_features) != shape:
                raise ValueError(f"{path} must have in/out features {shape}")
            yield path, attention, name, projection


def attach_lora(native: nn.Module) -> nn.Module:
    """Attach exactly nine wrappers in place; return native for convenience.

    Validate all targets and the caller's freeze before mutating any module.
    Repeated attachment is rejected rather than stacking residuals.
    """
    projections = list(_projections(native))
    if any(isinstance(projection, LoRALinear)
           for _, _, _, projection in projections):
        raise ValueError("LoRA is already attached")
    if any(parameter.requires_grad for parameter in native.parameters()):
        raise ValueError("Caller must freeze native BEFORE attach_lora")
    for _, attention, name, projection in projections:
        setattr(attention, name, LoRALinear(projection))
    return native


def _lora_parameters(native: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    projections = list(_projections(native))
    if any(not isinstance(projection, LoRALinear)
           for _, _, _, projection in projections):
        raise ValueError("Call attach_lora before accessing LoRA parameters")
    for path, _, _, projection in projections:
        yield f"{path}.A", projection.A
        yield f"{path}.B", projection.B


def trainable_parameters(native: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only trainable A/B tensors from the nine fixed projections."""
    for _, parameter in _lora_parameters(native):
        if parameter.requires_grad:
            yield parameter


def lora_state_dict(native: nn.Module) -> dict[str, Tensor]:
    """Export independent CPU A/B tensors, with no native weights or metadata."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _lora_parameters(native)
    }


def load_lora_state_dict(native: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Strictly load an A/B-only state; validate every entry before copying."""
    parameters = dict(_lora_parameters(native))
    missing = parameters.keys() - state.keys()
    unexpected = state.keys() - parameters.keys()
    if missing or unexpected:
        raise ValueError(
            f"Invalid LoRA state: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    prepared = {}
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, Tensor) or value.shape != parameter.shape:
            raise ValueError(f"{name} must be a tensor with shape {tuple(parameter.shape)}")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        prepared[name] = value.detach().to(device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(prepared[name])

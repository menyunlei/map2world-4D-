"""Compare a loaded SupportSceneFlow with its actual native parent in place.

Call check_native_parent_identity(model, noisy, t, text, map_tensor) inside
the SAME autocast context used for Detail inference. It returns a JSON-safe
proof; callers should require proof['exact_identity'] before a fresh run.
A mismatch returns status FAIL. Forward/contract errors propagate after
restoring the method and RNG state. No models, weights, or data are loaded
here; dreamx_scene_flow is imported lazily to obtain the parent method.

Both passes use model(...) and its inherited native forward dispatch. Only
the reference pass temporarily binds DreamXSceneFlow._flow to that same
model. The candidate branch remains enabled for the candidate pass. No
second model is allocated; sparse outputs are copied to CPU before the next
pass to avoid output aliasing and retaining both GPU outputs.

Global Python, NumPy, Torch CPU, and already-initialized Torch CUDA RNGs are
replayed and restored. Custom generators are not covered. Training modes,
parameter values, requires_grad flags, and existing gradients are not
changed by this helper; forwards run under no_grad. Like the native Detail
forward, it assumes forwards do not mutate model weights/buffers or inputs.
Use only with exclusive access to the model and process-global RNGs.
"""

import random
from types import MethodType

import numpy as np
import torch


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        # Do not initialize CUDA when checking a CPU model.
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None
        ),
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _same_rng(a, b):
    an, bn = a["numpy"], b["numpy"]
    numpy_equal = (
        an[0] == bn[0] and np.array_equal(an[1], bn[1])
        and an[2:] == bn[2:]
    )
    ac, bc = a["torch_cuda"], b["torch_cuda"]
    cuda_equal = (
        ac is None and bc is None
        or ac is not None and bc is not None and len(ac) == len(bc)
        and all(torch.equal(x, y) for x, y in zip(ac, bc))
    )
    return bool(
        a["python"] == b["python"] and numpy_equal
        and torch.equal(a["torch_cpu"], b["torch_cpu"]) and cuda_equal
    )


def _sparse_snapshot(output, label):
    coords, feats = getattr(output, "coords", None), getattr(output, "feats", None)
    if not isinstance(coords, torch.Tensor) or not isinstance(feats, torch.Tensor):
        raise TypeError(f"{label} forward must return sparse .coords and .feats tensors")
    if coords.ndim != 2 or feats.ndim != 2 or len(coords) != len(feats):
        raise ValueError(f"{label} sparse coordinates/features must be aligned 2-D tensors")
    if not feats.is_floating_point():
        raise TypeError(f"{label} sparse features must be floating point")
    metadata = {
        "type": f"{type(output).__module__}.{type(output).__qualname__}",
        "coords_shape": list(coords.shape),
        "features_shape": list(feats.shape),
        "coords_dtype": str(coords.dtype),
        "features_dtype": str(feats.dtype),
        "coords_device": str(coords.device),
        "features_device": str(feats.device),
    }
    return (
        coords.detach().to(device="cpu", copy=True),
        feats.detach().to(device="cpu", copy=True),
        metadata,
    )


@torch.no_grad()
def check_native_parent_identity(model, noisy, t, text, map_tensor):
    """Return exact sparse-coordinate/feature equality, with no optimizer step.

    ``model`` must be the unwrapped loaded candidate, inheriting the native
    DreamXSceneFlow.forward. Pass t in the units expected by that forward
    (the trainer currently passes t * 1000); this helper does not rescale it.
    Equality is row-for-row, with matching dtypes/devices and finite features.
    No coordinate sorting, tolerance, target lookup, or branch disabling is
    used. A nonzero/trained candidate can legitimately return status FAIL.
    """
    entry_rng = _rng_state()
    missing = object()
    original_instance_flow = model.__dict__.get("_flow", missing)
    swapped = False
    try:
        from dreamx_scene_flow import DreamXSceneFlow

        if not isinstance(model, DreamXSceneFlow):
            raise TypeError("model must be an unwrapped DreamXSceneFlow subclass")
        if getattr(model.forward, "__func__", None) is not DreamXSceneFlow.forward:
            raise TypeError("identity check requires inherited native forward dispatch")
        if not callable(getattr(model, "_flow", None)):
            raise TypeError("candidate must expose a callable _flow")
        if getattr(model._flow, "__func__", None) is DreamXSceneFlow._flow:
            raise TypeError("candidate _flow already points to the native parent")

        # Replay after the lazy import too, so both forwards start at entry RNG.
        _restore_rng(entry_rng)
        candidate_coords, candidate_feats, candidate_meta = _sparse_snapshot(
            model(noisy, t, text, map_tensor), "candidate"
        )
        candidate_rng_after = _rng_state()

        model._flow = MethodType(DreamXSceneFlow._flow, model)
        swapped = True
        _restore_rng(entry_rng)
        parent_coords, parent_feats, parent_meta = _sparse_snapshot(
            model(noisy, t, text, map_tensor), "native parent"
        )
        parent_rng_after = _rng_state()

        coords_equal = bool(torch.equal(candidate_coords, parent_coords))
        features_equal = bool(torch.equal(candidate_feats, parent_feats))
        candidate_finite = bool(torch.isfinite(candidate_feats).all())
        parent_finite = bool(torch.isfinite(parent_feats).all())
        contract_equal = candidate_meta == parent_meta
        exact = coords_equal and features_equal and contract_equal and candidate_finite and parent_finite
        max_delta = None
        if candidate_feats.shape == parent_feats.shape and candidate_finite and parent_finite:
            max_delta = (
                float((candidate_feats.double() - parent_feats.double()).abs().max())
                if candidate_feats.numel() else 0.0
            )
        proof = {
            "status": "PASS" if exact else "FAIL",
            "exact_identity": exact,
            "coords_equal": coords_equal,
            "features_equal": features_equal,
            "sparse_contract_equal": contract_equal,
            "candidate_features_finite": candidate_finite,
            "parent_features_finite": parent_finite,
            "max_abs_feature_delta": max_delta,
            "candidate_output": candidate_meta,
            "parent_output": parent_meta,
            "reference_flow": "dreamx_scene_flow.DreamXSceneFlow._flow",
            "forward_dispatch": "model(...) with inherited native forward",
            "same_model_and_weights": True,
            "same_initial_rng": True,
            "rng_consumption_equal": _same_rng(candidate_rng_after, parent_rng_after),
        }
    finally:
        try:
            if swapped:
                if original_instance_flow is missing:
                    delattr(model, "_flow")
                else:
                    model._flow = original_instance_flow
        finally:
            _restore_rng(entry_rng)

    proof["flow_binding_restored"] = (
        "_flow" not in model.__dict__ if original_instance_flow is missing
        else model.__dict__.get("_flow") is original_instance_flow
    )
    proof["rng_restored"] = _same_rng(entry_rng, _rng_state())
    if not proof["flow_binding_restored"] or not proof["rng_restored"]:
        raise RuntimeError("identity check did not restore the method/RNG state")
    return proof

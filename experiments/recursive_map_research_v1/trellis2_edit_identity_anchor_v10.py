"""One frozen-weight edit mechanism: anchor original-map object descriptors.

Only the object-MLP output is reused from the ORIGINAL INPUT MAP. The edited
map's coverage, boundary, relative position/extent, road/open masks, and query
XYZ continue through the existing prepare() unchanged. No latent, target,
mesh or reference-image features are copied. This is an edit-context mechanism,
not a newly trained model or a claim that the old descriptor is pure identity.
"""
from contextlib import contextmanager
import torch

@contextmanager
def anchor_original_descriptor(branch, original_map):
    summary=original_map['object_summary'].to(branch.Wo.weight)
    valid=original_map['object_valid'].to(device=summary.device)
    summary=torch.where(valid[...,None],summary,torch.zeros_like(summary))
    with torch.no_grad():descriptor=branch.object_mlp(summary).detach().clone()
    def frozen(_module,_args,output):
        if output.shape!=descriptor.shape:raise ValueError('Object slot contract changed')
        return descriptor
    hook=branch.object_mlp.register_forward_hook(frozen)
    try:yield descriptor
    finally:hook.remove()


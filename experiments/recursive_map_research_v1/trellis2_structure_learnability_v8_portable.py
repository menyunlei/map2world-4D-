"""Reload the delivered small trained weights; does not generate or use targets.

Caller loads the ORIGINAL frozen TRELLIS.2 Structure Flow first. These NPZs
contain only the existing map adapter and, for the candidate, 18 LoRA tensors.
They are model checkpoints, not 3D world assets.
"""
import numpy as np,torch
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_structure_lora_v8 import attach_lora,load_lora_state_dict

def load_portable(native,checkpoint_path):
    if any(p.requires_grad for p in native.parameters()):
        raise ValueError('Freeze original native weights before loading the adapter/LoRA')
    with np.load(checkpoint_path,allow_pickle=False) as z:
        branch_state={k.split('::',1)[1]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('branch::')}
        lora_state={k.split('::',1)[1]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('lora::')}
        assert len(branch_state)+len(lora_state)==len(z.files)
    branch=ObjectContractConditioner(reading_mode='object_grounded')
    branch.load_state_dict(branch_state,strict=True)
    branch.to(next(native.parameters()).device).eval().requires_grad_(False)
    if lora_state:
        attach_lora(native);load_lora_state_dict(native,lora_state)
    native.eval().requires_grad_(False)
    return native,branch

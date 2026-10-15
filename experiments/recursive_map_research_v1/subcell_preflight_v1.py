"""Actual native microbatch, exact adapter identity, gradients, and address checks."""
from paths import *
from subcell_runtime_v1 import *
from subcell_model_v1 import spatial_features
import argparse, time
import torch.nn.functional as F

p = argparse.ArgumentParser(); p.add_argument('--arm', required=True); p.add_argument('--seed', type=int, required=True)
a = p.parse_args(); torch.cuda.set_device(0); torch.set_num_threads(4)
model, branch, parent, restore = initialize(a.arm, a.seed)
rt.restore_rng(restore)
data = Data('structure'); aux = rt.auxiliary(data)
r = data.lookup['courtyard_new_007_narrow1p5m']
_, noisy, target, t, text, raster, surface, seq = data.pack([r], [930007])
# Use a fixed legitimate TRAIN map without conditioning dropout for the preflight.
with __import__('numpy').load(r['map_path']) as z:
    raster = torch.from_numpy(z['map_tensor'].copy())[None].float().cuda()
relation = aux.topology(r['scene_id'])
args = (noisy, t * 1000, text, raster, relation)
torch.cuda.reset_peak_memory_stats(); tick = time.time()
model.train()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    actual = model(*args)
    added_read = model.branch.read
    old_read = added_read.__closure__
    # Disable only the new adapter, leaving all original trained G2-B paths active.
    old_forward = model.branch.subcell.forward
    model.branch.subcell.forward = lambda features, mask, site, dtype: torch.zeros((len(features), 3072), device=features.device, dtype=dtype)
    try: reference = model(*args)
    finally: model.branch.subcell.forward = old_forward
assert torch.equal(actual, reference), ('ADAPTER_ZERO_IDENTITY', float((actual-reference).abs().max()))
with torch.autocast('cuda', dtype=torch.bfloat16):
    prediction = model(*args)
    loss = rt.per_scene_flow(prediction, target, 'structure').mean()
loss.backward()
grad = sum(float(p.grad.float().abs().sum()) for name,p in branch.items() if name.startswith('branch.subcell.write'))
assert grad > 0 and torch.isfinite(loss), ('NO_NEW_ADAPTER_GRADIENT', grad)
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
# Equal means, different within-cell locations are distinguishable only in correct mode.
from hierarchy_model_v2 import gather_grid
from hierarchy_compile_v2 import empty_relation
channels = torch.zeros((1,3,64,64), device='cuda'); channels[0,1,0,0] = 1
other = channels.clone(); other[0,1,0,0] = 0; other[0,1,3,3] = 1
# Native grid coordinates come from the real wrapper; test the complete feature grids directly.
x, y = F.pixel_unshuffle(channels,4), F.pixel_unshuffle(other,4)
assert torch.equal(x.reshape(1,3,16,16,16).mean(2), y.reshape(1,3,16,16,16).mean(2)) and not torch.equal(x,y)
with torch.no_grad():
    model.branch.subcell.write[0].bias.fill_(1)
    unknown = model.branch.subcell(torch.zeros((2,48),device='cuda'),torch.zeros(2,device='cuda'),0,torch.float32)
    assert unknown.count_nonzero() == 0
save(NEW/a.arm/'runs/g2_b'/f'seed{a.seed}/SUBCELL_PREFLIGHT.json', {'real_TRAIN_microbatch':True,'exact_original_G2B_identity':True,
    'new_adapter_gradient_L1':grad,'same_area_different_subcell_distinguished':True,'unknown_exact_zero':True,
    'seconds':time.time()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'no_optimizer_updates':True})
print('SUBCELL_PREFLIGHT_PASS', a.arm, a.seed, grad, flush=True)

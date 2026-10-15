"""Reuse the working trainer; only geometry-image routing differs between arms."""
import argparse
import sys
import time
import trellis2_map_learning_train_fix1 as fixed
from paths import read, save, sha
from trellis2_map_geometry_entry_v2_prepare import OUT

base = fixed.base
torch = base.torch


def setup(arm, seed):
    directory = OUT / arm
    plan = read(directory / 'PLAN.json')
    parent = plan['parent_checkpoints'][str(seed)]
    assert sha(parent['path']) == parent['sha256']
    checkpoint = torch.load(parent['path'], map_location='cpu', weights_only=False)
    assert checkpoint['updates'] == 100 and checkpoint['map10_TRAIN'] is False
    native_branch = base.Trellis2MapObjectConditioner

    class ResumedBranch(native_branch):
        def __init__(self):
            super().__init__()
            self.load_state_dict(checkpoint['branch_state'], strict=True)

    def geometry_cond(current_plan):
        cond = fixed.image_cond(current_plan)
        return cond if arm == 'original' else torch.zeros_like(cond)

    base.OUT = directory
    base.Trellis2MapObjectConditioner = ResumedBranch
    base.image_cond = geometry_cond
    return plan, directory


def preflight(plan, directory, seed):
    start = time.time()
    torch.manual_seed(seed)
    cond = base.image_cond(plan)
    model = base.native_model()
    branch = base.Trellis2MapObjectConditioner().cuda()
    reader = base.BoundMapReader(model, branch)
    record = next(r for r in plan['records'] if r['split'] == 'adapter_TRAIN')
    x0, _ = base.clean(record)
    noise = torch.randn_like(x0)
    t = torch.tensor([.6], device='cuda')
    x = .4*x0 + (1e-5+(1-1e-5)*.6)*noise
    with torch.no_grad():
        unmapped = reader(x, t*1000, cond, None)
    predicted = reader(x, t*1000, cond, base.load_map(record))
    loss = (predicted.float() - ((1-1e-5)*noise-x0)).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradient = float(branch.Wo.weight.grad.float().norm())
    assert gradient > 0 and all(p.grad is None for p in model.parameters())
    if plan['arm'] == 'separated':
        assert torch.count_nonzero(cond) == 0, 'Image leaked into geometry'
    target = directory / f'seed{seed}'
    target.mkdir(exist_ok=False)
    save(target / 'PREFLIGHT.json', {'status': 'ACTUAL_NATIVE_FORWARD_BACKWARD_PASS',
         'arm': plan['arm'], 'seed': seed, 'source_TRAIN': record['scene_id'],
         'parent_checkpoint': plan['parent_checkpoints'][str(seed)],
         'geometry_cond_shape': list(cond.shape), 'geometry_cond_nonzero': int(torch.count_nonzero(cond)),
         'map_response_maxdiff': float((predicted.detach()-unmapped).abs().max()),
         'Wo_gradient_norm': gradient, 'native_gradients_none': True, 'optimizer_updates': 0,
         'trainable_parameters': sum(p.numel() for p in branch.parameters()),
         'allocated_GiB': torch.cuda.max_memory_allocated()/2**30, 'seconds': time.time()-start})
    print('REAL_PREFLIGHT_PASS', plan['arm'], gradient, tuple(cond.shape), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['original', 'separated'], required=True)
    parser.add_argument('--seed', type=int, choices=[953101, 953102], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan, directory = setup(args.arm, args.seed)
    if args.preflight:
        preflight(plan, directory, args.seed)
    else:
        sys.argv = [sys.argv[0], '--seed', str(args.seed)]
        base.main()

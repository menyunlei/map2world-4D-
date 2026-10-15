"""TRAIN-only generated-state Dice inputs; original flow targets are untouched."""
from paths import *
import numpy as np
import torch
import runtime as rt
from infer_structure import TopologyCondition
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

DEST = JOB / 'dice_generated_state_v1'


def dice_input(model, data, arm, slot, noisy, timestep, text, raster, topology):
    """Both arms pay identical rollout NFE; only registered Dice inputs differ."""
    selected = not slot['drop'] and slot['t'] < .5
    evidence = {'scene_id': slot['scene_id'], 'sample_id': slot['sample_id'],
        'noise_seed': slot['noise_seed'], 't': slot['t'], 'drop': slot['drop'],
        'eligible': selected, 'input_kind': 'target_noised', 'rollout_NFE': 0}
    if not selected:
        return noisy, evidence
    rng = rt.rng_state()
    was_training = model.training
    sampler = FlowEulerGuidanceIntervalSampler(1e-5)
    positive = TopologyCondition(text, raster, topology)
    unknown = torch.zeros_like(raster)
    unknown[:, 5] = 1
    negative = TopologyCondition(data.empty[None].cuda(), unknown,
        rt.relation_to_device(rt.empty_relation(), 'cuda'))
    times = np.linspace(1, 0, 26)
    times = 3 * times / (1 + 2 * times)
    goal = float(timestep[0])
    assert abs(goal - slot['t']) < 1e-6 and 0 < goal < .5
    calls = 0

    def prediction(x, t, condition):
        nonlocal calls
        calls += 1
        return model(x, t, condition.text, condition.map, condition.topology)

    try:
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = torch.randn(noisy.shape, device=noisy.device, dtype=torch.float32,
                generator=torch.Generator(device='cuda').manual_seed(slot['noise_seed']))
            for t, next_t in zip(times[:-1], times[1:]):
                if t <= goal:
                    break
                end = max(float(next_t), goal)
                generated = sampler.sample_once(prediction, generated, float(t), end, positive,
                    neg_cond=negative, cfg_strength=7.5, cfg_interval=[.5, .95]).pred_x_prev
                if end == goal:
                    break
            assert torch.isfinite(generated).all()
    finally:
        model.train(was_training)
        rt.restore_rng(rng)
    evidence.update({'rollout_NFE': calls, 'generated_state_std': float(generated.std()),
        'state_MSE_from_target_noised': float((generated - noisy).square().mean()),
        'rollout_uses_clean_target': False, 'rollout_detached': not generated.requires_grad})
    if arm == 'generated':
        evidence['input_kind'] = 'generated'
        return generated.detach(), evidence
    assert arm == 'baseline'
    return noisy, evidence


def main():
    base = JOB / 'dice_schedule_train_v1.py'
    source = base.read_text()

    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, ('TRAIN_WRAPPER_SOURCE_CHANGED', old, source.count(old))
        source = source.replace(old, new)

    replace("DEST = JOB / 'dice_schedule_probe_v1'", "DEST = JOB / 'dice_generated_state_v1'")
    replace("choices=['old', 'corrected']", "choices=['baseline', 'generated']")
    replace("starting_meta = read(JOB / 'runs/g2_b' / f'seed{args.seed}/CHECKPOINT_000300.json')",
        "starting_meta = read(plan['starts'][str(args.seed)])")
    replace("queue = plan['dice_queues'][args.arm]", "queue = plan['dice_queue']\n    rollout_audit = []")
    replace("_, noisy, _, timestep, text, raster, surface, seq = data.pack([record], [slot['sample_id']])",
        "_, noisy, _, timestep, text, raster, surface, seq = data.pack([record], [slot['sample_id']])", count=2)
    probe_anchor = "        topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])\n        with torch.autocast"
    replace(probe_anchor,
        "        topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])\n"
        "        noisy, evidence = dice_input(model, data, args.arm, slot, noisy, timestep, text, raster, topology)\n"
        "        with torch.autocast")
    replace("slot = queue[0]", "slot = next(row for row in queue if not row['drop'] and row['t'] < .5)")
    replace("probe_loss.backward()", "probe_loss.backward()\n        assert np.isfinite(grad_norm(parameters)) and grad_norm(parameters) > 0\n        assert evidence['rollout_NFE'] > 0 and evidence['rollout_detached']")
    replace("'peak_allocated_bytes': torch.cuda.max_memory_allocated()", "'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'generated_state_preflight': evidence")
    anchor = "                topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])\n                with torch.autocast"
    replace(anchor,
        "                topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])\n"
        "                noisy, evidence = dice_input(model, data, args.arm, slot, noisy, timestep, text, raster, topology)\n"
        "                rollout_audit.append(evidence)\n                with torch.autocast")
    replace("flow_losses, dice_losses, aux_losses, flow_audit, dice_audit = [], [], [], [], []",
        "flow_losses, dice_losses, aux_losses, flow_audit, dice_audit = [], [], [], [], []\n        rollout_audit = []")
    replace("'flow_audit': flow_audit, 'dice_audit': dice_audit,", "'flow_audit': flow_audit, 'dice_audit': dice_audit, 'rollout_audit': rollout_audit,")
    scope = {'__name__': '__main__', '__file__': str(base), 'dice_input': dice_input}
    exec(compile(source, str(DEST / 'EFFECTIVE_TRAIN.py'), 'exec'), scope)


if __name__ == '__main__':
    main()

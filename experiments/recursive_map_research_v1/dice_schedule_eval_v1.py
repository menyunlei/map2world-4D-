"""Fixed seen-TRAIN native Structure checks at 0/20/100 updates."""
from paths import *
import argparse
import time
import numpy as np
import torch
import runtime as rt
from infer_structure import TopologyCondition
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler, FlowEulerSampler
import verifier_core as verifier

DEST = JOB / 'dice_schedule_probe_v1'
OLD = JOB / 'query_boundary_structure_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['old', 'corrected'], required=True)
    parser.add_argument('--seed', type=int, choices=[1701, 1702], required=True)
    parser.add_argument('--update', type=int, choices=[0, 20, 100], required=True)
    args = parser.parse_args()
    plan = read(DEST / 'PLAN.json')
    for path, digest in plan['frozen_source_and_inputs'].items():
        assert sha(path) == digest, ('FROZEN_CHANGED', path)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, parameters, _, _ = rt.initialize('g2_b', args.seed)
    meta = read(JOB / 'runs/g2_b' / f'seed{args.seed}/CHECKPOINT_000300.json') if args.update == 0 else read(
        DEST / args.arm / f'seed{args.seed}' / f'CHECKPOINT_{args.update:06d}.json')
    assert sha(meta['path']) == meta['sha256']
    saved = torch.load(meta['path'], map_location='cpu', weights_only=False)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved['branch_state'][name])
    model.eval().requires_grad_(False)
    decoder = rt.structure_decoder().eval().requires_grad_(False)
    guided = FlowEulerGuidanceIntervalSampler(1e-5)
    plain = FlowEulerSampler(1e-5)
    schedule = np.linspace(1, 0, 26)
    schedule = 3 * schedule / (1 + 2 * schedule)
    zero = rt.relation_to_device(rt.empty_relation(), 'cuda')
    packet = read(OLD / 'train_probe/baseline/packets/g2_b_300.json')
    units = read(OLD / 'clean_trace_v1/PLAN.json')['units']
    eval_seeds = {row['noise_seed'] for row in packet['rows']}
    train_noise_seeds = {slot['noise_seed'] for batch in plan['flow_updates'] for slot in batch}
    assert eval_seeds.isdisjoint(train_noise_seeds)
    out = DEST / 'eval' / args.arm / f'seed{args.seed}' / f'plus{args.update:03d}'
    out.mkdir(parents=True, exist_ok=False)
    results = []

    def decode(value):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = decoder(value.float())
        return torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].int().cpu()

    for row in packet['rows']:
        sid = row['scene_id']
        assert sid in plan['units'] and sid in units
        tick = time.time()
        with np.load(row['map_path']) as archive:
            map_tensor = archive['map_tensor'].copy()
        raster = torch.from_numpy(map_tensor)[None].float().cuda()
        unknown = torch.zeros_like(raster)
        unknown[:, 5] = 1
        relation = rt.relation_to_device(rt.compile_relation(map_tensor), 'cuda')
        text = torch.load(row['text']['path'], map_location='cpu', weights_only=True)['context'][None].cuda()
        negative = torch.load(row['negative']['path'], map_location='cpu', weights_only=True)['context'][None].cuda()
        positive = TopologyCondition(text, raster, relation)
        neg = TopologyCondition(negative, unknown, zero)
        noise = torch.randn((1, 8, 16, 16, 16), device='cuda',
            generator=torch.Generator(device='cuda').manual_seed(row['noise_seed']))
        def predict(x, t, cond):
            return model(x, t, cond.text, cond.map, cond.topology)
        guidance = dict(neg_cond=neg, cfg_strength=7.5, cfg_interval=[.5, .95])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sample = guided.sample(predict, noise, positive, steps=25, rescale_t=3,
                                   verbose=False, **guidance)
            final_q = decode(sample.samples)
        # Training target and masks enter only after free generation is finished.
        target = torch.load(units[sid]['structure'], map_location='cpu', weights_only=True)
        clean = target['S0'].cuda()
        target_q = verifier.dense(target['Q0'].numpy())
        with np.load(units[sid]['labels']) as archive:
            masks = {key: archive[key].astype(bool) for key in archive.files}
        def score(q):
            return verifier.score(verifier.dense(q.numpy()), target_q, masks, row['variant'])
        assert score(decode(clean))['overall_native_IoU'] == 1.0
        teacher = {}
        for index in plan['evaluation']['teacher_step_indices']:
            t = float(schedule[index])
            x_t = (1 - t) * clean + (1e-5 + (1 - 1e-5) * t) * noise
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cond, _, _ = plain._get_model_prediction(predict, x_t, t, positive)
                cfg, _, _ = guided._get_model_prediction(predict, x_t, t, positive, **guidance)
            teacher[str(index + 1)] = {'t': t, 'conditional': score(decode(cond)),
                                        'native_CFG': score(decode(cfg))}
        result = {'arm': args.arm, 'seed': args.seed, 'update': args.update,
                  'scene': sid, 'family': row['family'], 'variant': row['variant'],
                  'eval_noise_seed': row['noise_seed'], 'free': score(final_q),
                  'teacher': teacher, 'seconds': time.time() - tick,
                  'teacher_uses_target_noised_TRAIN_structure': True,
                  'free_generation_never_uses_target': True}
        save(out / f'{sid}.json', result)
        torch.save({'predicted_S': sample.samples.cpu(), 'predicted_Q': final_q,
                    'structure_noise': noise.cpu(), 'no_target_input_to_free_generation': True},
                   out / f'{sid}.pt')
        results.append(result)
        print('EVAL', args.arm, args.seed, args.update, sid, round(result['seconds'], 2), flush=True)
        del sample, target, clean
    save(out / 'RESULTS.json', {'records': results, 'checkpoint_sha256': meta['sha256'],
         'all_eight_TRAIN_scenes': True, 'eval_noise_never_training': True,
         'scope': 'TRAIN learning ability only'})


if __name__ == '__main__':
    main()

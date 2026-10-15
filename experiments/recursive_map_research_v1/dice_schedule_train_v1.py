"""G2-B fixed TRAIN capability probe; only the Dice scene queue differs."""
from paths import *
import argparse
import json
import time
import torch
import numpy as np
import runtime as rt
from subcell_runtime_v1 import Data

DEST = JOB / 'dice_schedule_probe_v1'


def grad_norm(parameters, before=None):
    values = []
    for name, parameter in parameters.items():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.float()
        if before is not None:
            gradient = gradient - before[name].float()
        values.append(gradient.norm())
    return float(torch.stack(values).norm()) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['old', 'corrected'], required=True)
    parser.add_argument('--seed', type=int, choices=[1701, 1702], required=True)
    parser.add_argument('--until', type=int, choices=[20, 100], required=True)
    parser.add_argument('--resume', type=int, choices=[20])
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    plan_path = DEST / 'PLAN.json'
    plan = read(plan_path)
    for path, digest in plan['frozen_source_and_inputs'].items():
        assert sha(path) == digest, ('FROZEN_CHANGED', path)
    assert plan['dice_coefficient'] == 0.1 and plan['loss_divisor'] == 16
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, parameters, parent, _ = rt.initialize('g2_b', args.seed)
    starting_meta = read(JOB / 'runs/g2_b' / f'seed{args.seed}/CHECKPOINT_000300.json')
    assert sha(starting_meta['path']) == starting_meta['sha256']
    starting = torch.load(starting_meta['path'], map_location='cpu', weights_only=False)
    assert set(parameters) == set(starting['branch_state'])
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(starting['branch_state'][name])
            assert torch.equal(parameter.cpu(), starting['branch_state'][name])
    model.train()
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, weight_decay=0, fused=True)
    ema = {name: p.detach().clone() for name, p in parameters.items()}
    output = DEST / args.arm / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        meta = read(output / 'CHECKPOINT_000020.json')
        assert sha(meta['path']) == meta['sha256']
        saved = torch.load(meta['path'], map_location='cpu', weights_only=False)
        assert saved['update'] == args.resume and saved['queue_cursor'] == 80
        assert saved['plan_sha256'] == sha(plan_path) and set(parameters) == set(saved['branch_state'])
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved['branch_state'][name])
        optimizer.load_state_dict(saved['optimizer'])
        ema = {name: value.cuda() for name, value in saved['EMA'].items()}
        rt.restore_rng(saved['rng'])
        start = args.resume
    else:
        assert not (output / 'TRAIN_CURVE.jsonl').exists(), 'duplicate training start'
        rt.restore_rng(starting['rng'])
        start = 0
    assert start < args.until
    data = Data('structure')
    decoder = rt.structure_decoder()
    auxiliary = rt.auxiliary(data)
    aux_schedule = {g['added_update']: g for g in read(C4 / 'AUXILIARY_SCHEDULE.json')['groups']}
    aux_coeff = read(C4 / 'AUXILIARY_CONTRACT.json')['coefficient']
    frozen = [p for p in model.parameters() if not p.requires_grad]
    queue = plan['dice_queues'][args.arm]
    torch.cuda.reset_peak_memory_stats()
    if args.probe_only:
        slot = queue[0]
        record = data.lookup[slot['scene_id']]
        _, noisy, _, timestep, text, raster, surface, seq = data.pack([record], [slot['sample_id']])
        assert abs(float(timestep[0]) - slot['t']) < 1e-6 and seq[0][3] == slot['drop']
        topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(noisy, timestep * 1000, text, raster, topology)
            dice = rt.decoded_dice(decoder, noisy, timestep, prediction, surface, True)
            probe_loss = 0.1 * dice.sum() / 16
        assert torch.isfinite(probe_loss)
        probe_loss.backward()
        assert all(p.grad is None for p in (p for p in model.parameters() if not p.requires_grad))
        save(output / 'PREFLIGHT.json', {'TRAIN_Dice_microbatch': True, 'no_optimizer_update': True,
            'dice_grad_norm': grad_norm(parameters), 'start_sha256': starting_meta['sha256'],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        print('DICE_SCHEDULE_PREFLIGHT_PASS', args.arm, args.seed, flush=True)
        return
    for update in range(start + 1, args.until + 1):
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        flow_losses, dice_losses, aux_losses, flow_audit, dice_audit = [], [], [], [], []
        for slot in plan['flow_updates'][update - 1]:
            record = data.lookup[slot['scene_id']]
            assert record['split'] == 'SUPPLEMENTARY_TRAIN'
            _, noisy, target, timestep, text, raster, _, seq = data.pack([record], [slot['sample_id']])
            assert abs(float(timestep[0]) - slot['t']) < 1e-6 and seq[0][3] == slot['drop']
            topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(noisy, timestep * 1000, text, raster, topology)
                flow = rt.per_scene_flow(prediction, target, 'structure')
                loss = flow.sum() / 16
            assert torch.isfinite(loss)
            loss.backward()
            flow_losses.append(float(flow.mean()))
            flow_audit.append([slot['scene_id'], slot['sample_id'], slot['t'], slot['drop']])
            del noisy, target, timestep, text, raster, prediction, flow, loss
        flow_grad = grad_norm(parameters)
        if update % 4 == 0:
            prior = {name: p.grad.detach().clone() for name, p in parameters.items()}
            for slot in queue[(update // 4 - 1) * 16:(update // 4) * 16]:
                record = data.lookup[slot['scene_id']]
                assert record['split'] == 'SUPPLEMENTARY_TRAIN'
                _, noisy, _, timestep, text, raster, surface, seq = data.pack([record], [slot['sample_id']])
                assert abs(float(timestep[0]) - slot['t']) < 1e-6 and seq[0][3] == slot['drop']
                topology = auxiliary.zero if slot['drop'] else auxiliary.topology(record['scene_id'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(noisy, timestep * 1000, text, raster, topology)
                    dice = rt.decoded_dice(decoder, noisy, timestep, prediction, surface, True)
                    loss = 0.1 * dice.sum() / 16
                assert torch.isfinite(loss)
                loss.backward()
                dice_losses.append(float(dice.mean()))
                dice_audit.append([slot['scene_id'], slot['sample_id'], slot['t'], slot['drop']])
                del noisy, timestep, text, raster, surface, prediction, dice, loss
            dice_grad = grad_norm(parameters, prior)
            del prior
            after_dice = {name: p.grad.detach().clone() for name, p in parameters.items()}
            auxiliary_rng = rt.rng_state()
            group = aux_schedule[update]
            for entry in group['entries']:
                batch, topology, masks = auxiliary.pack_auxiliary(entry)
                _, noisy, _, timestep, text, raster, surface, _ = batch
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(noisy, timestep * 1000, text, raster, topology)
                    logits = rt.decode(decoder, noisy, timestep, prediction)
                    _, positive, negatives = rt.terms(logits, surface, masks, auxiliary.groups[(update // 4) - 1])
                    loss = aux_coeff * (positive + negatives['O1'])
                assert torch.isfinite(loss)
                loss.backward()
                aux_losses.append(float(loss))
                del batch, noisy, timestep, text, raster, surface, prediction, logits, loss, masks
            rt.restore_rng(auxiliary_rng)
            aux_grad = grad_norm(parameters, after_dice)
            del after_dice
        else:
            dice_grad = aux_grad = 0.0
        assert all(p.grad is None for p in frozen), 'frozen parent gradient'
        assert all(p.grad is not None for p in parameters.values())
        preclip = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0))
        assert np.isfinite(preclip)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in parameters.items():
                ema[name].lerp_(parameter, 1 - .9999)
        torch.cuda.synchronize()
        row = {'update': update, 'arm': args.arm, 'seed': args.seed,
               'flow': float(np.mean(flow_losses)), 'dice': float(np.mean(dice_losses)) if dice_losses else 0.0,
               'auxiliary_loss': sum(aux_losses), 'flow_grad_norm': flow_grad,
               'dice_grad_norm': dice_grad, 'auxiliary_grad_norm': aux_grad,
               'preclip_grad_norm': preclip, 'seconds': time.time() - tick,
               'flow_audit': flow_audit, 'dice_audit': dice_audit,
               'optimizer_update_completed': True}
        with (output / 'TRAIN_CURVE.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print('UPDATE', args.arm, args.seed, update, round(row['seconds'], 2), flush=True)
        if update in (20, 100):
            state = output / f'checkpoint_{update:06d}.pt'
            assert not state.exists()
            torch.save({'arm': args.arm, 'seed': args.seed, 'update': update, 'queue_cursor': update // 4 * 16,
                'plan_sha256': sha(plan_path), 'starting_checkpoint_sha256': starting_meta['sha256'],
                'branch_state': {name: p.detach().cpu() for name, p in parameters.items()},
                'optimizer': optimizer.state_dict(), 'EMA': {name: value.cpu() for name, value in ema.items()},
                'rng': rt.rng_state()}, state)
            save(output / f'CHECKPOINT_{update:06d}.json', {'path': str(state), 'sha256': sha(state),
                'update': update, 'arm': args.arm, 'seed': args.seed})
    save(output / f'COMPLETE_{start:03d}_{args.until:03d}.json', {'update': args.until,
         'hard_stop_at100': args.until == 100, 'no_automatic_extension': True})


if __name__ == '__main__':
    main()

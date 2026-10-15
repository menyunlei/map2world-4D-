"""Paired per-family TRAIN recovery and pure-noise readout, no new gate."""
from paths import *
from collections import Counter
from statistics import mean
from verifier_core import THRESHOLDS

DEST = JOB / 'dice_schedule_probe_v1'


def opening(metrics):
    return (metrics['free_air_wrong_fraction'] == 0 and
            metrics['overhead_column_fraction'] == 0 and
            metrics['required_surface_recall'] >= THRESHOLDS['required_surface_recall'])


def closed(metrics):
    return metrics['closed_closure_recall'] >= THRESHOLDS['closed_closure_recall']


def main():
    plan = read(DEST / 'PLAN.json')
    results = {}
    for seed in plan['seeds']:
        for arm in plan['arms']:
            for update in plan['evaluation']['checkpoints']:
                actual_arm = 'old' if update == 0 else arm
                packet = read(DEST / 'eval' / actual_arm / f'seed{seed}' / f'plus{update:03d}/RESULTS.json')
                rows = {r['scene']: r for r in packet['records']}
                assert set(rows) == set(plan['units'])
                family_rows = []
                for family in plan['family_ids']:
                    prefix = f'courtyard_new_{family:03d}_'
                    opened, roof = rows[prefix + 'narrow1p5m'], rows[prefix + 'closed']
                    variants = {}
                    for name, part in [('free', 'free'), ('mid_conditional', ('teacher', '17', 'conditional')),
                                       ('mid_native_CFG', ('teacher', '17', 'native_CFG')),
                                       ('low_conditional', ('teacher', '25', 'conditional')),
                                       ('low_native_CFG', ('teacher', '25', 'native_CFG'))]:
                        def metric(row):
                            return row[part] if isinstance(part, str) else row[part[0]][part[1]][part[2]]
                        a, b = metric(opened), metric(roof)
                        variants[name] = {'open_air_wrong': a['free_air_wrong_fraction'],
                            'open_overhead_wrong': a['overhead_column_fraction'],
                            'open_required_recall': a['required_surface_recall'],
                            'closed_roof_recall': b['closed_closure_recall'],
                            'open_and_walls': opening(a), 'paired_open_wall_roof': opening(a) and closed(b)}
                    family_rows.append({'family': family, 'metrics': variants})
                summary = {kind: {'opening_and_walls': sum(r['metrics'][kind]['open_and_walls'] for r in family_rows),
                    'paired_open_wall_roof': sum(r['metrics'][kind]['paired_open_wall_roof'] for r in family_rows),
                    'mean_open_air_wrong': mean(r['metrics'][kind]['open_air_wrong'] for r in family_rows),
                    'mean_open_overhead_wrong': mean(r['metrics'][kind]['open_overhead_wrong'] for r in family_rows),
                    'mean_open_required_recall': mean(r['metrics'][kind]['open_required_recall'] for r in family_rows),
                    'mean_closed_roof_recall': mean(r['metrics'][kind]['closed_roof_recall'] for r in family_rows)}
                    for kind in family_rows[0]['metrics']}
                results[(arm, seed, update)] = {'arm': arm, 'seed': seed, 'update': update,
                    'four_families': family_rows, 'summary': summary,
                    'identical_start_from_old_record': update == 0}
    coverage, gradients = {}, {}
    for arm in plan['arms']:
        for seed in plan['seeds']:
            root = DEST / arm / f'seed{seed}'
            rows = [read_line for read_line in (json.loads(line) for line in (root / 'TRAIN_CURVE.jsonl').read_text().splitlines())]
            assert [r['update'] for r in rows] == list(range(1, 101))
            flow = Counter(sid for r in rows for sid, _, _, drop in r['flow_audit'] if not drop)
            dice = Counter(sid for r in rows for sid, _, _, drop in r['dice_audit'] if not drop)
            assert sum(len(r['dice_audit']) for r in rows) == 400
            assert sum(len(r['flow_audit']) for r in rows) == 1600
            coverage[f'{arm}_{seed}'] = {'conditioned_flow': dict(flow), 'conditioned_dice': dict(dice),
                'dice_microbatch_count': 400, 'dice_optimizer_update_count': 25}
            gradients[f'{arm}_{seed}'] = {'flow_mean': mean(r['flow_grad_norm'] for r in rows),
                'dice_mean_on_active_updates': mean(r['dice_grad_norm'] for r in rows if r['update'] % 4 == 0),
                'existing_O1_mean_on_active_updates': mean(r['auxiliary_grad_norm'] for r in rows if r['update'] % 4 == 0),
                'preclip_mean': mean(r['preclip_grad_norm'] for r in rows),
                'per_update': [{'update': r['update'], 'flow': r['flow_grad_norm'],
                                'dice': r['dice_grad_norm'], 'existing_O1': r['auxiliary_grad_norm'],
                                'preclip': r['preclip_grad_norm']} for r in rows]}
    paired = []
    for seed in plan['seeds']:
        a, b = results[('old', seed, 100)], results[('corrected', seed, 100)]
        paired.append({'seed': seed, 'old': a['summary'], 'corrected': b['summary'],
            'mid_conditional_joint_delta': b['summary']['mid_conditional']['paired_open_wall_roof'] - a['summary']['mid_conditional']['paired_open_wall_roof'],
            'free_joint_delta': b['summary']['free']['paired_open_wall_roof'] - a['summary']['free']['paired_open_wall_roof'],
            'low_joint_delta': b['summary']['low_conditional']['paired_open_wall_roof'] - a['summary']['low_conditional']['paired_open_wall_roof']})
    costs = [read(p) for p in (DEST / 'gpu_cost').glob('*.json')]
    assert costs and all(r['returncode'] == 0 for r in costs)
    result = {'status': 'MATCHED_100_COMPLETE', 'scope': 'fixed seen-TRAIN learning ability, not generalization',
        'starting_checkpoint': 'original G2-B@300',
        'results': [results[key] for key in sorted(results)],
        'paired_100': paired, 'actual_conditioned_coverage': coverage,
        'gradient_components': gradients,
        'gpu_seconds_including_load_idle': sum(r['gpu_seconds'] for r in costs),
        'gpu_cost_receipts': costs, 'no_300_extension': True,
        'human_review': 'PENDING',
        'decision': ('SCHEDULE_REPAIR_SHOWS_PAIRED_TRAIN_JOINT_GAIN' if all(
            r['mid_conditional_joint_delta'] > 0 and r['free_joint_delta'] > 0 and r['low_joint_delta'] >= 0 for r in paired)
            else 'SCHEDULE_REPAIR_NOT_ESTABLISHED_AS_FULL_GENERATION_FIX'),
        'note': 'Opening and wall/roof gates reuse frozen thresholds; teacher-forced target-noise recovery is separate from pure-noise generation.'}
    save(DEST / 'RESULTS.json', result)
    save(DEST / 'COMPLETE.json', {'status': result['status'], 'decision': result['decision'],
        'gpu_seconds': result['gpu_seconds_including_load_idle'], 'hard_stop100': True})
    print(result['decision'], flush=True)


if __name__ == '__main__':
    import json
    main()

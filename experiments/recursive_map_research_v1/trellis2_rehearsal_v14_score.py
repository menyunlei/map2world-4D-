"""Matched per-family comparisons and exact training-slot audit, CPU only."""
from paths import JOB, read, save, sha
from trellis2_rehearsal_v14_train import OUT, ARM, SEEDS, replay_order
from trellis2_coverage_v12_prepare import OUT as V12
import collections, json, statistics

def avg(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float, bool))]
    return statistics.mean(values) if values else None

def main():
    plan = read(OUT/'PLAN.json'); raw = []; family = []; summaries = []; exposure = []; mesh = []
    for seed in SEEDS:
        baseline_log = [json.loads(x) for x in (V12/'small/bounded_coverage'/f'seed{seed}/train/train_log.jsonl').read_text().splitlines()]
        new_log = [json.loads(x) for x in (OUT/'small'/ARM/f'seed{seed}/train/train_log.jsonl').read_text().splitlines()]
        assert len(baseline_log) == len(new_log) == 1000
        counts = collections.Counter(); conditional = collections.Counter()
        order = replay_order(plan['order_seed'])
        expected = [plan['selected_records'][i]['scene_id']+':'+str(plan['selected_records'][i]['object_id']) for i in order]
        for u, (old, new) in enumerate(zip(baseline_log, new_log)):
            assert len(new['slots']) == 2
            for m, (a, b) in enumerate(zip(old['slots'], new['slots'])):
                assert all(a[k] == b[k] for k in ['noise_key', 't', 'drop']), ('SLOT_POLICY_CHANGED', u, m)
                assert b['object'] == expected[(u*2+m) % len(expected)]
                counts[b['object']] += 1; conditional[b['object']] += not b['drop']
        exposure.append({'seed': seed, 'slots': sum(counts.values()), 'per_object': dict(counts),
                         'conditioned_per_object': dict(conditional), 'noise_t_dropout_exactly_matched': True,
                         'old_new_per_update': [1, 1], 'updates': 1000})
        for arm, root in [('bounded_coverage', V12), (ARM, OUT)]:
            for update in [100, 300, 1000]:
                path = root/f'evaluation/seed{seed}/{arm}/update{update:06d}/RESULTS.json'
                if not path.exists(): continue
                result = read(path); raw.append({'path': str(path), 'sha256': sha(path)})
                for group in ['original_six', 'added_TRAIN', 'heldout']:
                    rows = [r for r in result['rows'] if r['mode'] == 'correct' and r['group'] == group]
                    edits = [r for r in result['edit_rows'] if r['group'] == group]
                    fs = []
                    for fam in sorted({r['family'] for r in rows}):
                        rr = [r for r in rows if r['family'] == fam]; er = [r for r in edits if r['family'] == fam]
                        row = {'seed': seed, 'arm': arm, 'update': update, 'group': group, 'family': fam,
                            'map_IoU': avg(rr, 'map_projection_IoU'), 'footprint_coverage': avg(rr, 'required_footprint_coverage'),
                            'outside_fraction': avg(rr, 'outside_footprint_fraction'), 'empty_fraction': avg(rr, 'empty'),
                            'thin_axes_mean': avg(rr, 'axes_below_three_cells'),
                            'largest_component_fraction': avg(rr, 'largest_component_fraction'),
                            'removed_occupancy_after': avg(er, 'removed_occupancy_after'),
                            'retained_coverage_after': avg(er, 'retained_coverage_after'),
                            'protected_Q_change_fraction': avg(er, 'protected_Q_change_fraction')}
                        family.append(row); fs.append(row)
                    summaries.append({'seed': seed, 'arm': arm, 'update': update, 'group': group,
                                      'families': len(fs), **{k: avg(fs, k) for k in list(fs[0])[5:]}})
            path = root/f'evaluation/seed{seed}/RENDER_COMPLETE.json'
            if path.exists(): mesh.extend(r for r in read(path)['rows'] if r['arm'] == arm)
    deltas = []
    for seed in SEEDS:
        for group in ['original_six', 'added_TRAIN', 'heldout']:
            pair = [next((r for r in summaries if (r['seed'], r['arm'], r['update'], r['group']) ==
                         (seed, arm, 1000, group)), None) for arm in ['bounded_coverage', ARM]]
            if not all(pair): continue
            deltas.append({'seed': seed, 'group': group,
                **{k+'_delta': pair[1][k]-pair[0][k] for k in ['map_IoU', 'footprint_coverage', 'outside_fraction',
                      'empty_fraction', 'thin_axes_mean', 'removed_occupancy_after', 'retained_coverage_after']}})
    payload = {'status': 'NUMERIC_COMPLETE_AI_VISUAL_REVIEW_PENDING', 'summaries': summaries, 'per_family': family,
        'final_candidate_minus_control': deltas, 'exposure': exposure, 'mesh_rows': mesh, 'source_results': raw,
        'GPU_cost': [read(p) for p in (OUT/'gpu_cost').glob('*.json')], 'checkpoint_selection': '1000 only',
        'not_a_real_Map10_result': True, 'human_review_not_claimed': True, 'no_more_training_queued': True,
        'limitations': ['Original six span only two independent map families.', 'Authored source assets overlap across layouts.',
                       'Unknown height/roof changes alone are not new map-violation gates.', 'Full neighbor mesh regeneration was not measured.']}
    save(OUT/'RESULTS.json', payload)
    for row in deltas: print('FINAL_DELTA', row, flush=True)

if __name__ == '__main__': main()

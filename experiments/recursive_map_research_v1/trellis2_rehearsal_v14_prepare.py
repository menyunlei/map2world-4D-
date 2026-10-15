"""Register one fixed replay comparison. Reuse targets and historical controls."""
from paths import JOB, read, save, sha
from trellis2_coverage_v12_prepare import OUT as V12
from trellis2_rehearsal_v14_train import OUT, ARM, SEEDS, replay_order
from collections import Counter
import copy

def prepare():
    assert not OUT.exists(), 'Preserve previous output; no duplicate experiment'
    previous = read(V12 / 'PLAN.json')
    assert read(V12 / 'EVALUATION_QUEUE_COMPLETE.json')['errors'] == []
    records = previous['selected_records']
    assert len(records) == 24 and records[:6] == previous['six_records']
    assert all(r['split'] == 'adapter_TRAIN' for r in records)
    assert not ({r['family_id'] for r in records} & {r['family_id'] for r in previous['heldout_records']})
    order = replay_order(previous['order_seed'])
    assert len(order) == 36 and all(order[i] < 6 <= order[i+1] for i in range(0, 36, 2))
    counts = {}
    for updates in [100, 300, 1000]:
        counts[str(updates)] = dict(Counter(order[s % len(order)] for s in range(updates * 2)))
    frozen = {**read(V12 / 'SOURCE_FREEZE.json'), **read(V12 / 'EVALUATION_FREEZE.json')}
    for seed in SEEDS:
        for update in [100, 300, 1000]:
            ck = read(V12 / 'small/bounded_coverage' / f'seed{seed}/train/CHECKPOINT_{update:06d}.json')
            frozen[ck['path']] = ck['sha256']
            path = V12 / f'evaluation/seed{seed}/bounded_coverage/update{update:06d}/RESULTS.json'
            frozen[str(path)] = sha(path)
        for name in ['RENDER_COMPLETE.json', 'MESH_COMPLETE.json']:
            path = V12 / f'evaluation/seed{seed}' / name
            frozen[str(path)] = sha(path)
    plan = {**copy.deepcopy(previous),
        'version': 'v14_fixed_old_new_replay', 'arms': [ARM],
        'authorization': 'User: ok, do it, following the fixed old-object rehearsal proposal',
        'only_variable': 'TRAIN object allocation: one old6 and one added18 object per effective batch2',
        'small': {'updates': 1000, 'checkpoints': [100, 300, 1000], 'effective_batch': 2, 'primary_checkpoint': 1000},
        'control': {'experiment': str(V12), 'arm': 'bounded_coverage', 'reuse_trained_checkpoints_and_outputs': True},
        'replay': {'old_indices': list(range(6)), 'new_indices': list(range(6, 24)),
                   'cycle': order, 'cycle_slots': 36, 'planned_counts_by_index': counts,
                   'no_extra_samples_or_updates': True, 'same_slot_noise_t_dropout_policy': True},
        'GPU_cap_seconds': 5400, 'training_seconds_each': 1500,
        'scheduling': 'Two registered training seeds concurrently on free verified GPU0/1, then fixed evaluation',
        'milestone_evaluation': [100, 300, 1000], 'no_automatic_extension': True,
        'large_conditional': {'authorized': False, 'updates': 0},
        'success': 'Both seeds must retain old-object generated geometry and edit response while preserving added TRAIN/held-out layout and nondegenerate final meshes. Report regressions; do not choose an intermediate winner.',
    }
    for path, h in frozen.items():
        assert sha(path) == h, ('PREVIOUS_SOURCE_CHANGED', path)
    save(OUT / 'PLAN.json', plan)
    save(OUT / 'SCHEDULE_AUDIT.json', {'order': order, 'counts': counts, 'effective_batch': 2,
          'old_new_slots_per_update': [1, 1], 'inference_contract_unchanged': True})
    names = ['trellis2_rehearsal_v14_train.py', 'trellis2_rehearsal_v14_prepare.py',
             'trellis2_rehearsal_v14_eval.py', 'trellis2_rehearsal_v14_owner.py', 'trellis2_rehearsal_v14_score.py']
    frozen.update({str(JOB/n): sha(JOB/n) for n in names})
    frozen[str(OUT / 'PLAN.json')] = sha(OUT / 'PLAN.json')
    save(OUT / 'SOURCE_FREEZE.json', frozen)
    print('REPLAY_REGISTERED', counts, flush=True)

if __name__ == '__main__':
    prepare()

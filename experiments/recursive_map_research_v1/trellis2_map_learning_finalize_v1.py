"""Archive the bounded learned-map trial, including its negative real-map results."""
from paths import JOB, read, save, sha
from trellis2_map_learning_prepare_fix1 import OUT
from collections import defaultdict
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not (OUT / 'FINAL_RESULTS.json').exists(), 'Never overwrite an archived result'
    plan = read(OUT / 'PLAN.json')
    training = read(OUT / 'TRAINING_COMPLETE.json')
    assets = read(OUT / 'ASSETS_COMPLETE.json')
    views = read(OUT / 'NATIVE_RELOADS_COMPLETE.json')
    assert training['updates_per_seed'] == 100 and assets['native_assets'] == 8
    assert views['view_count'] == 72

    per_family = []
    summaries = []
    for seed in plan['seeds']:
        units = read(OUT / f'seed{seed}/HELDOUT_000100.json')['units']
        grouped = defaultdict(list)
        for unit in units:
            grouped[unit['family_id']].append(unit)
        for family, cases in sorted(grouped.items()):
            row = {'training_seed': seed, 'family': family, 'paired_scenes': len(cases)}
            for mode in ['baseline', 'correct', 'shuffled']:
                row[f'{mode}_flow_MSE'] = sum(c[f'{mode}_flow_MSE'] for c in cases) / len(cases)
            row['diagnostic_only_not_final3D_success'] = True
            per_family.append(row)
        summary = {'training_seed': seed, 'assessed_scenes': len(units),
                   'independent_families': len(grouped)}
        for mode in ['baseline', 'correct', 'shuffled']:
            summary[f'{mode}_flow_MSE'] = sum(u[f'{mode}_flow_MSE'] for u in units) / len(units)
        summary['relative_correct_improvement_vs_baseline'] = (
            1 - summary['correct_flow_MSE'] / summary['baseline_flow_MSE'])
        summaries.append(summary)
    write_csv(OUT / 'FINAL_FLOW_PER_FAMILY.csv', per_family)

    cost_rows = []
    for receipt in sorted((OUT / 'gpu_cost').glob('*.json')):
        record = read(receipt)
        cost_rows.append({'receipt': str(receipt), 'receipt_sha256': sha(receipt),
                          'actual_GPU_seconds_including_load_idle': record['gpu_seconds'],
                          'returncode': record.get('returncode', '')})
    assert cost_rows
    write_csv(OUT / 'FINAL_GPU_COST.csv', cost_rows)
    seconds = sum(r['actual_GPU_seconds_including_load_idle'] for r in cost_rows)

    evidence = []
    for seed in plan['seeds']:
        for mode in ['baseline', 'correct', 'shuffled', 'edit']:
            root = OUT / f'seed{seed}/assets/{mode}'
            evidence.append({'training_seed': seed, 'mode': mode,
                             'asset': read(root / 'ASSET.json'),
                             'reloaded_view_record': read(root / 'views/RELOAD.json'),
                             'agent_visual_findings': {
                                 'source': 'actual independently reloaded contact sheet; AI review, not human acceptance',
                                 'multi_building_road_layout': 'FAIL: one stylized reference-like building, not the annotated block',
                                 'realistic_street_world': 'FAIL: no demonstrated navigable mapped street scene',
                                 'local_building_edit': 'FAIL' if mode == 'edit' else 'NOT_THIS_CONDITION',
                                 'street_camera_limitation': 'Fixed input-based street cameras rendered blank; not alone proof of absent geometry'},
                             'motion': read(root / 'MOTION.json') if (root / 'MOTION.json').exists() else None})

    result = {
        'verdict': 'TRAINED_MAP_ADAPTER_REAL_BLOCK_FAIL',
        'goal_achieved': False,
        'actual_training_completed': True,
        'seeds': plan['seeds'], 'optimizer_updates_each': 100,
        'trainable_parameters': 315392, 'frozen_SS_flow_parameters': 1292179976,
        'TRAIN': plan['TRAIN_counts'], 'adapter_heldout_available': plan['HELDOUT_counts'],
        'teacher_forced_assessed': summaries,
        'verified_real_map_mesh_training_pairs': 0,
        'map10_in_training': False, 'historical_DEV_or_sealed_test_used': False,
        'real_map_input_limitations': '10.jpg oblique screenshot; four manually fixed roof proxies and selected asphalt patches; no surveyed footprints, real object IDs, heights or geographic scale',
        'native_assets': assets['native_assets'], 'independently_reloaded_views': views['view_count'],
        'evidence': evidence,
        'interpretation_correction': {
            'original_PLAN_preserved': True,
            'misleading_original_field': 'appearance_condition',
            'actual_contract': 'The same reference building image conditions native SS geometry, fine ShapeFlow geometry and texture; it is not appearance-only',
            'diagnosed_failure_location': 'Map-controlled block layout already fails at coarse SS generation; not merely GLB export',
            'support_attribution': read(OUT / 'SPARSE_SUPPORT_ATTRIBUTION.json'),
            'inference': 'Small learned SS residual did not overcome the single-building image geometry condition within this bounded run; not proof of model impossibility or that longer training will fix it'},
        'next_mechanism_recommendation_not_launched': 'Within the same TRELLIS.2, make native SS and fine ShapeFlow geometry map-led; restrict the single-building image reference to appearance. Verify native shape training targets and real paired-data availability before another optimizer run.',
        'quality_limitations': 'Stylized building reference and authored synthetic TRAIN; preview GLB has model-generated basecolor/alpha but not the complete spatial PBR material; raw native PBR remains saved on server',
        'cost': {'actual_GPU_seconds_including_load_idle_failed_preflight': seconds,
                 'GPU_hours': seconds / 3600, 'receipt_count': len(cost_rows),
                 'physical_occupancy_not_process_doublecounted': True},
        'human_review': 'PENDING_NOT_A_BLOCKER',
        'historical_results_and_thresholds_unchanged': True,
        'automatic_training_extension': False,
        'scope': 'One bounded map-conditioning mechanism pilot; not Map-to-World success'}
    save(OUT / 'FINAL_RESULTS.json', result)
    save(OUT / 'FINAL_SESSION_STATE.json', {
        'status': 'BOUNDED_RUN_COMPLETED_GOAL_NOT_ACHIEVED',
        'verdict': result['verdict'], 'active_GPU_jobs': [],
        'next_optimizer_launch': 'NONE', 'historical_state_preserved': True,
        'result_path': str(OUT / 'FINAL_RESULTS.json'), 'result_sha256': sha(OUT / 'FINAL_RESULTS.json')})
    print(json.dumps({'verdict': result['verdict'], 'GPU_seconds': seconds,
                      'GPU_hours': seconds / 3600, 'assessed': summaries,
                      'result_sha256': sha(OUT / 'FINAL_RESULTS.json')}), flush=True)


if __name__ == '__main__':
    main()

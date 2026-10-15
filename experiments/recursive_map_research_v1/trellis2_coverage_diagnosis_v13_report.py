"""Summarize saved evidence only; no training, sampling, checkpoint selection or repair."""
from pathlib import Path
import json, hashlib, statistics, datetime

JOB = Path(__file__).resolve().parent
V12 = JOB / 'map_object_world_v1/trellis2_coverage_v12'
OUT = V12 / 'diagnosis_v1'
SEEDS = [953101, 953102]
COMBOS = ['coverage_map_parent_lora', 'parent_map_coverage_lora']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mean(rows, field):
    return statistics.mean(r[field] for r in rows)

def main():
    evidence = read(OUT / 'CPU_EVIDENCE.json')
    summary_path = V12 / 'summary/RESULTS_1791983621702023294.json'
    summary = read(summary_path)
    curves = summary['summaries_family_balanced']
    report = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'DIAGNOSIS_COMPLETE_NO_VALIDATED_FIX_NO_NEW_TRAINING',
        'historical_v12_verdict_unchanged': True,
        'sources': {
            str(summary_path.relative_to(JOB)): digest(summary_path),
            str((OUT / 'CPU_EVIDENCE.json').relative_to(JOB)): digest(OUT / 'CPU_EVIDENCE.json'),
        },
        'actual_training_updates_this_diagnosis': 0,
        'crossover_completion': read(OUT / 'crossover/COMPLETE.json'),
        'exposure': [], 'curves': [], 'crossover': [], 'first_error_layer': [],
    }
    for seed in SEEDS:
        for arm in ['six_object_rehearsal', 'bounded_coverage']:
            row = next(r for r in evidence['exposures_and_gradients']
                       if (r['seed'], r['arm'], r['update']) == (seed, arm, 1000))
            report['exposure'].append({
                'seed': seed, 'arm': arm,
                'total_slots_per_object_min_max': [min(row['counts'].values()), max(row['counts'].values())],
                'conditioned_slots_min_max': [min(row['conditioned'].values()), max(row['conditioned'].values())],
                'clipped_updates': row['clipped_updates'],
            })
        for group in ['original_six', 'added_TRAIN', 'heldout']:
            result = {'seed': seed, 'group': group}
            for arm in ['parent', 'six_object_rehearsal', 'bounded_coverage']:
                result[arm] = {str(r['update']): r['map_IoU'] for r in curves
                               if r['seed'] == seed and r['arm'] == arm and r['group'] == group}
            report['curves'].append(result)
        for arm in ['parent', 'six_object_rehearsal', 'bounded_coverage', *COMBOS]:
            update = 0 if arm == 'parent' else 1000
            base = OUT / 'crossover' if arm in COMBOS else V12
            path = base / f'evaluation/seed{seed}/{arm}/update{update:06d}/RESULTS.json'
            rows = [r for r in read(path)['rows'] if r['index'] in [0, 1, 6, 7]]
            mesh = read(base / f'evaluation/seed{seed}/RENDER_COMPLETE.json')['rows']
            entry = {'seed': seed, 'arm': arm, 'TRAIN_indices': [0, 1, 6, 7],
                     'generation_seeds': [970201, 970202],
                     'aggregation': 'fixed four-object diagnostic mean, NOT independent held-out evidence',
                     'mesh_indices': [0, 1], 'mesh_generation_seed': 970201}
            for mode in ['correct', 'edit']:
                entry[mode] = {
                    'SS_map_IoU': mean([r for r in rows if r['mode'] == mode], 'map_projection_IoU'),
                    'mesh_map_IoU': mean([r for r in mesh if r['arm'] == arm and r['mode'] == mode and r['index'] in [0, 1]], 'mesh_map_IoU'),
                }
            report['crossover'].append(entry)
        for arm in ['parent', 'six_object_rehearsal', 'bounded_coverage']:
            update = 0 if arm == 'parent' else 1000
            q = next(r for r in evidence['saved_native_Q']
                     if (r['seed'], r['arm'], r['update'], r['index'], r['noise']) ==
                     (seed, arm, update, 0, 970201))
            mesh = next(r for r in read(V12 / f'evaluation/seed{seed}/RENDER_COMPLETE.json')['rows']
                        if (r['arm'], r['mode'], r['index']) == (arm, 'correct', 0))
            report['first_error_layer'].append({
                'seed': seed, 'arm': arm, 'object_index': 0, 'noise': 970201,
                'SS_central90_Z_span_cells': q['central90_z_span'],
                'SS_Z_min_max_cells': q['z_min_max'], 'mesh_native_Z_extent': mesh['extent'][2],
                'interpretation': 'Shape drift is already present in Structure, not created solely by export/materials. Height change alone is NOT a map violation or a new failure threshold.',
            })
    report['conclusions'] = {
        'exposure_schedule_bug': False,
        'slot_noise_t_dropout_matched': evidence['matched_slot_noise_time_dropout_exact'],
        'per_object_exposure_is_reduced': True,
        'exposure_insufficiency_as_only_explanation': 'NOT_SUPPORTED: added exposure from +300 to +1000 does not monotonically improve pure-noise generation; seed953101 regresses substantially.',
        'expanded_training_stability_regression': 'OBSERVED on old TRAIN objects and in saved Structure/mesh pairs, seed dependent; also not proof that every new object is harmed.',
        'relative_weight_drift_is_not_causal_proof': True,
        'unique_bad_module_identified': False,
        'restore_parent_LoRA_as_fix': 'REJECTED_AS_VALIDATED_FIX: seed953101 improves, but seed953102 edit scores and visible mesh integrity worsen.',
        'more_exposure_might_still_help_some_objects': 'NOT_EXCLUDED: exact exposure-matched checkpoints are unavailable and object diversity differs.',
        'no_checkpoint_promoted_from_intermediate_results': True,
        'no_new_Map10_or_materials_or_mesh_repair': True,
    }
    report['AI_visual_review'] = {
        'reviewer': 'AI assistant, not human acceptance',
        'files_inspected': [f'crossover/evaluation/seed{s}/object{i:03d}_MATCHED_MESH.jpg'
                            for s in SEEDS for i in [0, 1]],
        '953101': 'Both crossed models remain recognizable building massing on the two probes. Footprint edit also changes roof/height; neither establishes general stability.',
        '953102': 'coverage_map_parent_lora shows detached floating fragments on object0 edit and object1 original/edit. parent_map_coverage_lora looks more coherent here but does not consistently improve layout/edit scores.',
        'blank_bottom_rows': 'The reused v12 sheet template has six rows; crossover has four populated rows. Two blank rows are not missing generated cases.',
        'raw_outputs_preserved': True,
    }
    report['single_proposed_intervention_NOT_LAUNCHED'] = {
        'name': 'explicit_old_object_rehearsal_in_each_batch',
        'hypothesis_not_proven_fix': 'Counteract dilution of already learned object behavior during coverage expansion, without assuming that either adapter or LoRA is independently defective.',
        'change_only': 'At effective batch2, one slot cycles the original6 TRAIN objects and one cycles the added18 TRAIN objects; no extra slots or optimizer updates.',
        'why_this_one': 'Old-object share falls from 100% to25% in v12; old-object pure-noise/mesh retention worsens. Module swapping did not yield a robust fix, so do not amputate either module or change the backbone.',
        'same': ['TRELLIS.2', 'Map Adapter + Structure LoRA', 'registered parent checkpoints',
                 'native flow objective', 'optimizer', 'slot-indexed noise/time/dropout', '24 TRAIN targets',
                 'coordinate contract', 'frozen Shape/decoders', 'evaluation seeds and old thresholds'],
        'suggested_bounded_test': {'training_seeds': SEEDS, 'max_updates_each': 1000,
                                  'matched_control': 'Existing uniform24 v12 checkpoints at +100/+300/+1000; retain its original update budget, not fourfold exposure compensation.',
                                  'new_control_training_needed': False,
                                  'eval_needed': 'Pure-noise all40 records; original/edit final mesh with same fixed v12 objects. Reuse matched final control assets; do not select an intermediate winner.',
                                  'no_automatic_extension': True},
        'risk': 'New objects receive fewer slots; all added TRAIN and held-out results must be reported, not only old-object retention.',
        'training_launch_status': 'NOT_LAUNCHED; this report does not establish that replay will fix the problem.',
    }
    p = OUT / 'DIAGNOSIS_DECISION.json'
    with p.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'report': str(p), 'sha256': digest(p), 'GPU_seconds': report['crossover_completion']['GPU_seconds'],
                      'training_updates': 0, 'status': report['status']}))
    for r in report['crossover']:
        print(r['seed'], r['arm'], r['correct'], r['edit'])

if __name__ == '__main__':
    main()

"""Record completed measurements and actual AI visual inspection; no model operations."""
from pathlib import Path
import json, statistics, hashlib

JOB = Path(__file__).resolve().parent
OUT = JOB/'map_object_world_v1/trellis2_rehearsal_v14'

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    results = read(OUT/'RESULTS.json'); complete = read(OUT/'COMPLETE.json')
    assert complete['errors'] == []
    meshes = results['mesh_rows']; table = []
    for seed in [953101, 953102]:
        for group, ids in [('fixed_TRAIN_pair', [0, 1]), ('heldout_mesh_subset', [24, 26, 28, 30, 32, 34])]:
            for mode in ['correct', 'edit']:
                record = {'seed': seed, 'group': group, 'mode': mode}
                for arm in ['bounded_coverage', 'old_new_replay']:
                    rows = [r for r in meshes if r['seed'] == seed and r['arm'] == arm
                            and r['mode'] == mode and r['index'] in ids]
                    families = sorted({r['family'] for r in rows})
                    assert len(rows) == len(ids)
                    record[arm] = {'family_count': len(families), 'instances': len(rows),
                        'mesh_map_IoU': statistics.mean(statistics.mean(r['mesh_map_IoU'] for r in rows if r['family'] == f) for f in families),
                        'watertight_count': sum(r['watertight'] for r in rows)}
                table.append(record)
    notes = []
    for seed in [953101, 953102]:
        for index in [0, 1, 24, 26, 28, 30, 32, 34]:
            if seed == 953101 and index == 1:
                text = 'Original replay mesh has detached floating fragments above the building, visible in front/lateral/oblique views. Edited output has building volume. Saved Structure has two6connected components and a96.47% largest-component fraction; the extra support is already visible before fine decoding.'
            elif seed == 953101 and index in [24, 26, 32]:
                text = 'Recognizable connected building massing in original/edit. Replay replaces the low flat-roof form with a taller gabled form. Unknown roof/height changes are observations, not new failure gates. No obvious large detached fragment in these displayed views.'
            elif seed == 953101:
                text = 'Recognizable building volume; footprint edit changes roof/facade geometry as well as width. Edited ornamental edges remain rough, particularly object30. No empty or flat-sheet replacement in the displayed views.'
            elif index in [1, 24, 26, 32]:
                text = 'Replay produces a much lower, wide building than the control, still with nonzero3D volume. The top overlay shows incomplete footprint coverage. Unknown height alone is not graded as map failure; map and edit regression is measured separately.'
            else:
                text = 'Original/edit remain recognizable gabled building volumes under lateral views. Edit changes roof massing and leaves map-overlay discrepancies. No obvious large detached fragment in the displayed views.'
            p = OUT/f'comparison/seed{seed}/object{index:03d}.jpg'
            assert p.exists()
            notes.append({'seed': seed, 'index': index, 'source': str(p.relative_to(OUT)),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'observation': text})
    payload = {
        'verdict': 'REPLAY_NOT_A_ROBUST_FIX_STOP',
        'primary_checkpoint': 1000, 'training_updates_per_seed': 1000,
        'new_GPU_seconds_including_load_and_evaluation': complete['GPU_seconds'],
        'new_GPU_hours': complete['GPU_seconds']/3600,
        'final_native_meshes_independently_reloaded': 32, 'matched_sheets_actually_inspected': 16,
        'reviewer': 'AI assistant, not human acceptance',
        'mesh_comparisons_family_balanced': table,
        'per_scene_visual_observations': notes,
        'findings': [
            'Seed953101 improves coarse and final mesh map alignment; seed953102 regresses in both and retains less required footprint after editing.',
            'Increasing old-object exposure from83/84 to166/167 did not consistently preserve its geometry. Exposure dilution alone is therefore insufficient as an explanation or repair.',
            'A floating-fragment case is visible in Structure support before unchanged fine decoding. Do not treat this solely as an export/material problem.',
            'No empty or obvious flat-sheet replacement was seen in these fixed rendered cases. All32 candidate meshes remain nonwatertight; loading and plausible massing do not prove solid geometry.',
            'The results do not identify one defective parameter group or justify another unbounded training run.'
        ],
        'decision': 'Do not promote replay as the repair or advance this candidate to a new Map10 block demonstration.',
        'next_needed_evidence': 'Localize the remaining seed-dependent conditional Structure behavior using the saved checkpoints before selecting any further model intervention.',
        'no_more_training_queued': True, 'no_new_Map10': True, 'historical_v9_v11_v12_preserved': True,
        'limitations': results['limitations']
    }
    p = OUT/'DECISION.json'
    with p.open('x', encoding='utf-8') as h: json.dump(payload, h, indent=2, ensure_ascii=False)
    print(json.dumps({'decision': str(p), 'verdict': payload['verdict'], 'GPU_hours': payload['new_GPU_hours']}))
    for r in table: print(json.dumps(r))

if __name__ == '__main__': main()

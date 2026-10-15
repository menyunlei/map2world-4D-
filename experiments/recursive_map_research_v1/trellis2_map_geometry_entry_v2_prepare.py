"""Register one matched condition-entry correction, preserving v1 and all targets."""
from paths import JOB, read, save, sha
from pathlib import Path
import shutil

OUT = JOB / 'map_object_world_v1/trellis2_map_geometry_entry_v2'
PARENT = JOB / 'map_object_world_v1/trellis2_map_learning_v1/trial_fix1'
SEEDS = [953101, 953102]
ARMS = ['original', 'separated']


def register():
    assert not OUT.exists(), 'Do not overwrite a previous run'
    old = read(PARENT / 'PLAN.json')
    OUT.mkdir()
    for folder in ['logs', 'gpu_cost', 'audit']:
        (OUT / folder).mkdir()
    real = OUT / 'real_input'
    shutil.copytree(PARENT / 'real_input', real)
    parents = {}
    for seed in SEEDS:
        checkpoint = read(PARENT / f'seed{seed}/CHECKPOINT_000100.json')
        assert sha(checkpoint['path']) == checkpoint['sha256']
        parents[str(seed)] = checkpoint
    common = {**old, 'status': 'REGISTERED_CONDITION_ROUTING_TEST',
        'seeds': SEEDS, 'max_updates': 300, 'checkpoints': [20, 100, 300],
        'parent_checkpoints': parents, 'starting_updates': 100,
        'optimizer_reset': 'identical AdamW reset for both arms',
        'single_intervention': 'Whether the reference image enters SS/fine shape geometry, or texture only',
        'unchanged': ['315392-parameter map branch architecture', 'frozen TRELLIS.2', 'TRAIN records',
                      'native flow target', 'optimizer settings', 'data/noise/dropout order',
                      'sampler/CFG', 'native decoders', 'world coordinates'],
        'separated_fine_geometry': 'Frozen native ShapeFlow with zero image features; spatial support comes from map-conditioned SS. No separately trained fine-shape adapter.',
        'conditional_map_flag': 'Explicit payload in SS CFG, never infer mapped/unmapped from whether zero image features are nonzero',
        'appearance_reference': old['appearance_condition'],
        'appearance_is_not_real_recovered_facades': True,
        'real_map_input': str(real / 'MAP_INPUT.json'),
        'no_training_on_Map10': True, 'real_paired_TRAIN_count': 0,
        'no_automatic_extension': True,
        'max_training_GPU_seconds_per_job': 1200,
        'max_generation_GPU_seconds_per_job': 900,
        'evaluation': ['correct map', 'shuffled spatial routing', 'remove last roof proxy',
                       'independent asset reload', 'fixed whole-cube and input-road cameras'],
        'success_not_guaranteed': 'A single reference-like building is a FAIL, not a street-block deliverable'}
    for arm in ARMS:
        directory = OUT / arm
        directory.mkdir()
        plan = {**common, 'arm': arm,
                'geometry_image_features': arm == 'original',
                'texture_image_features': True}
        save(directory / 'PLAN.json', plan)
    save(OUT / 'PLAN.json', common)
    files = ['trellis2_map_geometry_entry_v2_prepare.py', 'trellis2_map_geometry_entry_v2_train.py',
             'trellis2_map_geometry_entry_v2_owner.py', 'trellis2_map_geometry_entry_v2_generate.py',
             'trellis2_map_object_module_v1.py', 'trellis2_map_learning_train_v1.py',
             'trellis2_map_learning_train_fix1.py', 'trellis2_map_learning_generate_v1.py']
    freeze = {str(JOB / name): sha(JOB / name) for name in files}
    freeze[str(OUT / 'PLAN.json')] = sha(OUT / 'PLAN.json')
    for arm in ARMS:
        freeze[str(OUT / arm / 'PLAN.json')] = sha(OUT / arm / 'PLAN.json')
    for checkpoint in parents.values():
        freeze[checkpoint['path']] = checkpoint['sha256']
    save(OUT / 'SOURCE_FREEZE.json', freeze)
    save(OUT / 'SESSION_STATE.json', {'stage': 'REGISTERED', 'trained': False,
         'hard_stop': 'four matched jobs, 300 additional updates each',
         'old_results_preserved': True})
    print('ENTRY_CORRECTION_REGISTERED', SEEDS, ARMS, flush=True)


if __name__ == '__main__':
    register()

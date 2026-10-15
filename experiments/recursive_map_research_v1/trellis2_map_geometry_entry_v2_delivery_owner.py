"""Four fixed trained trajectories -> native assets -> fresh-process views."""
from paths import JOB, read, save, sha
from trellis2_map_geometry_entry_v2_prepare import OUT, ARMS, SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures
import fcntl
import subprocess
import sys


def run():
    assert read(OUT / 'TRAINING_COMPLETE.json')['total_additional_updates'] == 1200
    assert read(OUT / 'CONDITION_CONTRACT_CPU_FIX1.json')['status'] == 'PASS'
    assert not any(OUT.glob('*/seed*/assets/*/RAW_NATIVE.pt')), 'Never retry or overwrite assets'
    save(OUT / 'EDIT_NOISE_PREREGISTRATION.json', {
         'before_any_new_asset_or_edit_output': True,
         'applies_identically_to_both_arms_and_both_map_versions': True,
         'reason': 'Sparse coordinate-count changes must not silently redefine noise at unrelated addresses',
         'dense_SS_noise': 'unchanged native fixed-seed tensor',
         'sparse_shape_and_texture_noise': 'fixed Gaussian grids [64,64,64,C], indexed at model-generated coordinates',
         'stage_seeds': 'generation seed + 200000 + {LR0,HR1,texture2}*1000',
         'sampler_steps_CFG_normalization_and_decoders_unchanged': True,
         'no_generated_geometry_placement_pruning_or_forced_support': True})
    files = ['trellis2_map_geometry_entry_v2_generate.py', 'trellis2_map_geometry_entry_v2_generate_fix1.py',
             'trellis2_map_geometry_entry_v2_delivery_owner.py', 'trellis2_map_entry_views_v2.py']
    freeze = {str(JOB / name): sha(JOB / name) for name in files}
    for arm in ARMS:
        for seed in SEEDS:
            record = read(OUT / arm / f'seed{seed}/CHECKPOINT_000300.json')
            assert sha(record['path']) == record['sha256']
            freeze[record['path']] = record['sha256']
    save(OUT / 'DELIVERY_SOURCE_FREEZE.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in freeze.items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    def pair(arm, seed, gpu):
        fn(gpu, 'entry_'+arm, seed, 'assets', 'trellis2_map_geometry_entry_v2_generate_fix1.py',
           ['--arm', arm, '--seed', str(seed)], seconds=900)
        modes = ['correct', 'edit'] + (['shuffled'] if arm == 'separated' else [])
        for mode in modes:
            fn(gpu, 'entry_'+arm, seed, 'views_'+mode, 'trellis2_map_entry_views_v2.py',
               ['--arm', arm, '--seed', str(seed), '--mode', mode, '--out', str(OUT)], seconds=180)
    with (OUT / 'delivery.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            jobs = [pool.submit(pair, arm, seed, GPUS[i*2+j])
                    for i, arm in enumerate(ARMS) for j, seed in enumerate(SEEDS)]
            for job in jobs:
                job.result()
    save(OUT / 'DELIVERY_COMPLETE.json', {'native_assets': 10, 'independent_reload_views': 120,
         'trained_arms': ARMS, 'seeds': SEEDS, 'no_more_training': True,
         'visual_acceptance_pending_not_claimed_from_export': True})


if __name__ == '__main__':
    if '--detach' in sys.argv:
        with (OUT / 'delivery_owner.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', '-u', __file__], stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(OUT / 'DELIVERY_OWNER.json', {'pid': process.pid, 'finite': True, 'no_automation': True,
             'no_new_training': True, 'native_asset_jobs': 4, 'GPU_indices': [0,1,2,3]})
        print('ENTRY_DELIVERY_OWNER', process.pid, flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(OUT / 'DELIVERY_FAILURE.json', {'type': type(error).__name__, 'error': str(error),
                                                'negative_outputs_preserved': True})
            raise

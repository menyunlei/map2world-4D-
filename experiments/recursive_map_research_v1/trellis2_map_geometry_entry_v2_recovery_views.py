"""Finish saved seed2 views after a legitimate empty shuffled-support failure."""
from paths import JOB, read, save, sha
from trellis2_map_geometry_entry_v2_prepare import OUT
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded

if __name__ == '__main__':
    assert read(OUT / 'DELIVERY_FAILURE.json')['type'] == 'AssertionError'
    freeze = {str(JOB / 'trellis2_map_entry_views_v2.py'): sha(JOB / 'trellis2_map_entry_views_v2.py')}
    for mode in ['correct', 'edit']:
        path = OUT / f'separated/seed953102/assets/{mode}/RAW_NATIVE.pt'
        assert path.exists()
        freeze[str(path)] = sha(path)
    save(OUT / 'RECOVERY_READOUT_FREEZE.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in freeze.items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    for mode in ['correct', 'edit']:
        fn(GPUS[3], 'entry_separated', 953102, 'recovery_views_'+mode, 'trellis2_map_entry_views_v2.py',
           ['--arm', 'separated', '--seed', '953102', '--mode', mode, '--out', str(OUT)], seconds=180)
    save(OUT / 'RELOADS_COMPLETE_WITH_EMPTY_NEGATIVE.json', {'native_assets': 9,
         'independent_views': 108, 'empty_native_support_case': 'separated/seed953102/shuffled',
         'no_retry_of_empty_generation': True, 'no_training_or_asset_repair': True})

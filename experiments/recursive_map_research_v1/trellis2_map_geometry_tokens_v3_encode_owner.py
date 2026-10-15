"""Four finite UUID-leased data-encoding jobs, not model training."""
from paths import JOB, read, save, sha
from trellis2_map_geometry_tokens_v3_prepare import OUT
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures
import subprocess
import sys


def run():
    freeze = {str(JOB / name):sha(JOB / name) for name in [
         'trellis2_map_geometry_tokens_v3_encode.py','trellis2_map_geometry_tokens_v3_encode_owner.py']}
    freeze[str(OUT / 'PLAN.json')] = sha(OUT / 'PLAN.json')
    save(OUT / 'ENCODE_SOURCE_FREEZE.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path,digest in freeze.items():
            assert sha(path) == digest,path
    fn.__globals__['frozen'] = verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn, gpu,'geometry_tokens',953101+worker,'encode'+str(worker),
             'trellis2_map_geometry_tokens_v3_encode.py',['--worker',str(worker)],seconds=600)
             for worker,gpu in enumerate(GPUS)]
        for job in jobs:
            job.result()
    plan=read(OUT / 'PLAN.json')
    records=[read(OUT / 'target_records' / (r['scene_id']+'.json')) for r in plan['records']]
    assert len(records)==120
    save(OUT / 'PAIRED_NATIVE_MANIFEST.json', {'records':records,
         'TRAIN':96,'HELDOUT':24,'independent_TRAIN_families':32,'independent_HELDOUT_families':8,
         'native_shape_encoder_frozen':True,'target_worlds_unchanged':True})
    print('ALL_NATIVE_SHAPE_TARGETS_READY',flush=True)


if __name__ == '__main__':
    if '--detach' in sys.argv:
        with (OUT / 'encode_owner.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT / 'ENCODE_OWNER.json',{'pid':process.pid,'finite':True,'no_optimizer':True})
        print('ENCODE_OWNER',process.pid,flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(OUT / 'ENCODE_FAILURE.json',{'type':type(error).__name__,'error':str(error),
                 'partial_targets_preserved':True,'no_training_launch':True})
            raise

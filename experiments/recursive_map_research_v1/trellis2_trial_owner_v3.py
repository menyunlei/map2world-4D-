"""Two existing UUID leases, finite zero-training inference jobs."""
from map_object_composition_owner_v1 import bounded, GPUS, JOB, read, save, sha
import concurrent.futures
import fcntl
import os
import subprocess
import sys

OUT=JOB/'map_object_world_v1/trellis2_resume_v3'

def run():
    assert (OUT/'SETUP_COMPLETE.json').exists()
    for name in ['logs','audit','gpu_cost']:(OUT/name).mkdir(exist_ok=True)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in read(OUT/'SOURCE_FREEZE.json').items():assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with (OUT/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(fn,gpu,'trellis2',seed,'assets','trellis2_asset_trial_v3.py',['--seed',str(seed)],seconds=2400)
                  for gpu,seed in zip(GPUS,[942201,942202])]
            for j in jobs:j.result()
    save(OUT/'INFERENCE_COMPLETE.json',{'seeds':[942201,942202],'training_updates':0})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'inference_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'INFERENCE_OWNER.json',{'pid':p.pid,'GPUs':GPUS,'finite':True,'cap_seconds_per_seed':2400})
        print('INFERENCE_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:
            save(OUT/'INFERENCE_FAILURE.json',{'type':type(e).__name__,'message':str(e),'preserve_outputs':True})
            raise

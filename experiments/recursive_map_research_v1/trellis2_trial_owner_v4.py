"""Two fixed inference seeds with a real native CUDA preflight before generation."""
from trellis2_trial_owner_v3 import OUT, GPUS, JOB, read, save, sha, bounded
import concurrent.futures
import fcntl
import os
import subprocess
import sys

def prepare():
    freeze=read(OUT/'SOURCE_FREEZE.json')
    for path,digest in freeze.items():assert sha(path)==digest,path
    for name in ['trellis2_trial_owner_v4.py','trellis2_gpu_preflight_v3.py','trellis2_mesh_stills_v3.py']:
        freeze[str(JOB/name)]=sha(JOB/name)
    save(OUT/'RUNTIME_SOURCE_FREEZE_V4.json',freeze)

def run():
    assert (OUT/'SETUP_COMPLETE.json').exists()
    for name in ['logs','audit','gpu_cost']:(OUT/name).mkdir(exist_ok=True)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in read(OUT/'RUNTIME_SOURCE_FREEZE_V4.json').items():assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with (OUT/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fn(GPUS[0],'trellis2',942201,'preflight','trellis2_gpu_preflight_v3.py',[],seconds=180)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(fn,gpu,'trellis2',seed,'assets','trellis2_asset_trial_v3.py',['--seed',str(seed)],seconds=2400)
                  for gpu,seed in zip(GPUS,[942201,942202])]
            for j in jobs:j.result()
    save(OUT/'INFERENCE_COMPLETE.json',{'seeds':[942201,942202],'training_updates':0})

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--detach' in sys.argv:
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

"""Retry only verified DINO API integration error; old records remain intact."""
from trellis2_map_learning_owner_v1 import OUT, JOB, read, save, sha, bounded, GPUS
import inspect, subprocess, sys, fcntl, concurrent.futures
import trellis2_map_learning_owner_v1 as old


def run():
    fn=bounded();fn.__globals__['NEW']=OUT
    freeze=read(OUT/'SOURCE_FREEZE.json')|read(OUT/'API_FIX1_SOURCE_FREEZE.json')
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with (OUT/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fn(GPUS[0],'trellis2_map',953101,'preflight_fix1','trellis2_map_learning_train_fix1.py',['--seed','953101','--preflight'],seconds=240)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(fn,gpu,'trellis2_map',seed,'train100','trellis2_map_learning_train_fix1.py',
                ['--seed',str(seed)],seconds=1800) for gpu,seed in zip(GPUS,[953101,953102])]
            for job in jobs:job.result()
    save(OUT/'TRAINING_COMPLETE.json',{'seeds':[953101,953102],'updates_per_seed':100,
        'automatic_extension':False,'asset_evaluation_pending':True})


if __name__=='__main__':
    if '--detach' in sys.argv:
        save(OUT/'API_FIX1_SOURCE_FREEZE.json',{str(JOB/n):sha(JOB/n) for n in [
            'trellis2_map_learning_train_fix1.py','trellis2_map_learning_owner_fix1.py']})
        with (OUT/'owner_fix1.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER_FIX1.json',{'pid':p.pid,'fix':'Dino helper .cuda returnsNone, use in-place call',
            'same_data_weights_seeds_objective_and_budget':True,'no_optimizer_updates_in_failed_preflight':True})
        print('OWNER_FIX1',p.pid,flush=True)
    else:
        try:run()
        except Exception as exc:
            save(OUT/'OWNER_FIX1_FAILURE.json',{'type':type(exc).__name__,'error':str(exc),'preserve_all_outputs':True})
            raise

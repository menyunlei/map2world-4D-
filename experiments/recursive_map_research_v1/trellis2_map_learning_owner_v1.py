"""Two live-checked A800 UUID leases, exactly one bounded map-module trial."""
from paths import JOB, read, save, sha
from trellis2_map_learning_prepare_fix1 import OUT
from map_object_composition_owner_v1 import bounded, GPUS
import sys, os, subprocess, concurrent.futures, fcntl


def register():
    files=['trellis2_map_learning_owner_v1.py','trellis2_map_learning_train_v1.py',
           'trellis2_map_object_module_v1.py','trellis2_map_learning_prepare_v1.py',
           'trellis2_map_learning_prepare_fix1.py']
    save(OUT/'SOURCE_FREEZE.json',{str(JOB/p):sha(JOB/p) for p in files}|{str(OUT/'PLAN.json'):sha(OUT/'PLAN.json')})
    for folder in ['audit','logs','gpu_cost']:(OUT/folder).mkdir(exist_ok=True)


def run():
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in read(OUT/'SOURCE_FREEZE.json').items():assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with (OUT/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fn(GPUS[0],'trellis2_map',953101,'preflight','trellis2_map_learning_train_v1.py',['--seed','953101','--preflight'],seconds=240)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpu,'trellis2_map',seed,'train100','trellis2_map_learning_train_v1.py',
                ['--seed',str(seed)],seconds=1800) for gpu,seed in zip(GPUS,[953101,953102])]
            for f in fs:f.result()
    save(OUT/'TRAINING_COMPLETE.json',{'seeds':[953101,953102],'updates_per_seed':100,
        'automatic_extension':False,'asset_evaluation_pending':True})


if __name__=='__main__':
    if '--register' in sys.argv:register()
    elif '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':p.pid,'finite_not_automation':True,'GPUs':GPUS,
            'maximum_optimizer_updates':200,'job_limit_per_seed_seconds':1800})
        print('MAP_LEARNING_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as exc:
            save(OUT/'OWNER_FAILURE.json',{'error':str(exc),'type':type(exc).__name__,
                'preserve_all_outputs':True,'no_unbounded_retry':True})
            raise

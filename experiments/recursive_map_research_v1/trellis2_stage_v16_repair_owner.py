"""Two fixed 100-update jobs, independent leases and shared 8 GPU-hour cap."""
from paths import JOB,read,save,sha
from trellis2_stage_v16 import OUT,SEEDS
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl

def verify():
    for name in ['SOURCE_FREEZE.json','REPAIR_SOURCE_FREEZE.json']:
        for f,h in read(OUT/name).items():assert sha(f)==h,('SOURCE_CHANGED',f)

def bounded():
    scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
    fn.__code__=fn.__code__.replace(co_consts=tuple(28700 if c==114900 else c for c in fn.__code__.co_consts))
    return fn

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fn=bounded();gpus=read(OUT/'PLAN.json')['GPUs']
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpus[i],'repair',seed,'train100','trellis2_stage_v16_train.py',
                ['--seed',str(seed)],seconds=1800) for i,seed in enumerate(SEEDS)]
            for f in fs:f.result()
        save(OUT/'REPAIR_TRAINING_COMPLETE.json',{'updates_each':100,'seeds':SEEDS,'no_training_extension':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'repair_owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'REPAIR_OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('REPAIR_OWNER',c.pid,flush=True)
    else:run()

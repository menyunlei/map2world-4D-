"""Finite four-GPU attribution owner; no automation and no training launched."""
from paths import JOB,read,save,sha
from trellis2_stage_v16 import OUT
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl

def verify():
    for f,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(28700 if c==114900 else c for c in fn.__code__.co_consts))
        p=read(OUT/'PLAN.json');errors=[]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures=[pool.submit(fn,p['GPUs'][gpu],stage,0,'diagnosis','trellis2_stage_v16.py',
                ['--stage',stage],seconds=900) for stage,gpu in p['stage_GPU'].items()]
            for future in futures:
                try:future.result()
                except Exception as e:errors.append(str(e))
        save(OUT/'DIAGNOSIS_COMPLETE.json',{'errors':errors,
             'gpu_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
             'no_training_started':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('STAGE_OWNER',c.pid,flush=True)
    else:run()

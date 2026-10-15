"""Finite existing-weight crossover on only verified idle GPU0/1. No training."""
from paths import JOB,read,save,sha
from trellis2_coverage_diagnosis_v13 import OUT,SEEDS
import subcell_owner_v1 as lease
import types,threading,fcntl,concurrent.futures,sys,subprocess

def verify():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,p

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        plan=read(OUT/'PLAN.json');gpus=plan['GPUs'];assert [g['index'] for g in gpus]==[0,1]
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
          'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        # Reserve teardown time inside the 900 physical GPU-second ceiling.
        fn.__code__=fn.__code__.replace(co_consts=tuple(870 if c==114900 else c for c in fn.__code__.co_consts))
        def task(i,seed):
            for mode in ['structure','mesh','views']:
                fn(gpus[i],'crossover',seed,mode,'trellis2_coverage_diagnosis_v13.py',
                  ['--seed',str(seed),'--mode',mode],seconds=360)
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for f in [pool.submit(task,i,s) for i,s in enumerate(SEEDS)]:
                try:f.result()
                except Exception as e:errors.append(str(e))
        save(OUT/'COMPLETE.json',{'errors':errors,'training_updates':0,'no_more_jobs':True,
          'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        verify()
        with (OUT/'owner.log').open('x') as h:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':p.pid,'no_automation':True,'no_optimizer':True,'only_GPUs':[0,1]})
        print('CROSSOVER_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'preserve_all':True});raise

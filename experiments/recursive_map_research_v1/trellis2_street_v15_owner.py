"""Phase-scoped, finite inference/rendering with shared physical occupancy cap."""
from paths import JOB,read,save,sha
from trellis2_street_v15_prepare import OUT,SEEDS
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl,argparse

def freeze(mode):
    scripts=[f'trellis2_street_v15_{mode}.py','trellis2_street_v15_owner.py',
             'trellis2_map10_block_v11_run.py','trellis2_map_edit_mesh_v9_run.py']
    sources={str(JOB/s):sha(JOB/s) for s in scripts}
    sources.update(read(OUT/'INPUT_FREEZE.json'));save(OUT/f'SOURCE_{mode}.json',sources)

def run(mode):
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def verify():
            for p,h in read(OUT/f'SOURCE_{mode}.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
          'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(3560 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_coverage_v12/PLAN.json')['GPUs']
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpus[i],'street',s,mode,f'trellis2_street_v15_{mode}.py',
                ['--seed',str(s)],seconds={'generate':600,'paint':900,'render':900}[mode]) for i,s in enumerate(SEEDS)]
            for f in fs:
                try:f.result()
                except Exception as e:errors.append(str(e))
        save(OUT/f'{mode.upper()}_COMPLETE.json',{'errors':errors,'phase':mode,'no_more_jobs':True,
            'GPU_seconds_to_date':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','paint','render'],required=True)
    p.add_argument('--register',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    if a.register:freeze(a.mode)
    elif a.detach:
        with (OUT/f'owner_{a.mode}.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--mode',a.mode],cwd=JOB,
                stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/f'OWNER_{a.mode}.json',{'pid':child.pid,'finite':True,'no_automation':True});print('STREET_OWNER',a.mode,child.pid,flush=True)
    else:run(a.mode)

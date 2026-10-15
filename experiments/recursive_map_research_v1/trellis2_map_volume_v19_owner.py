"""Reuse bounded GPU leases, all stages share one two-hour occupancy limit."""
from paths import JOB,read,save,sha
from trellis2_map_volume_v19 import OUT,SEED
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,fcntl,sys,subprocess

def verify():
    for name in ['DIAGNOSIS_SOURCE.json','CANDIDATE_SOURCE.json']:
        if (OUT/name).exists():
            for f,h in read(OUT/name).items():assert sha(f)==h,('SOURCE_CHANGED',f)

def bounded():
    scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
    fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
    return fn

def run(stage):
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fn=bounded()
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs'][:2]
        if stage=='diagnose':
            fn(gpus[0],'fixed',SEED,'diagnose','trellis2_map_volume_v19.py',['--diagnose'],seconds=180)
            return
        script={'coarse':'trellis2_map_volume_v19_coarse.py','generate':'trellis2_map_volume_v19_generate.py',
            'views':'trellis2_map_volume_v19_views.py'}[stage]
        limit={'coarse':600,'generate':1800,'views':600}[stage]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpus[i],mode,SEED,stage,script,['--mode',mode],seconds=limit) for i,mode in enumerate(['correct','edit'])]
            for f in fs:f.result()
        save(OUT/f'{stage.upper()}_COMPLETE.json',{'fixed_modes':['correct','edit'],'no_parameter_search':True})

if __name__=='__main__':
    stage=sys.argv[sys.argv.index('--stage')+1]
    if '--detach' in sys.argv:
        with (OUT/f'{stage}_owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__,'--stage',stage],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/f'{stage.upper()}_OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('OWNER',stage,c.pid,flush=True)
    else:run(stage)

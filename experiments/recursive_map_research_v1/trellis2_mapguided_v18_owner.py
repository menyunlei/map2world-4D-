"""One finite original/edit sampling attempt; no training or tuning continuation."""
from paths import JOB,read,save,sha
from trellis2_mapguided_v18 import OUT,SEED
import subcell_owner_v1 as lease
import types,threading,subprocess,sys,fcntl,concurrent.futures

def verify():
    for f,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs']
        def parallel(tasks):
            with concurrent.futures.ThreadPoolExecutor(len(tasks)) as pool:
                fs=[pool.submit(fn,gpu,mode,SEED,tag,script,['--mode',mode],seconds=limit)
                    for gpu,mode,tag,script,limit in tasks]
                for f in fs:f.result()
        parallel([(gpus[i],mode,'coarse','trellis2_mapguided_v18.py',600)
            for i,mode in enumerate(['correct','edit','baseline'])])
        save(OUT/'COARSE_COMPLETE.json',{'modes':['baseline','correct','edit'],'no_tuning':True})
        viable=[]
        for mode in ['correct','edit']:
            m=read(OUT/mode/'SS_METRICS.json')
            if m['coords32']>0 and min(m['native_extent_cells'])>=4:viable.append(mode)
            else:save(OUT/mode/'STOP.json',{'reason':'Empty or degenerate native Structure','no_substitution':True})
        parallel([(gpus[i],mode,'generate','trellis2_mapguided_v18_generate.py',1800) for i,mode in enumerate(viable)])
        parallel([(gpus[i+2],mode,'views','trellis2_mapguided_v18_views.py',600) for i,mode in enumerate(viable)])
        save(OUT/'COMPLETE.json',{'generated_modes':viable,'one_attempt_only':True,'no_more_jobs':True,
            'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'visual_and_map_review_pending':True,'new_model_training':False})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('MAPGUIDED_OWNER',c.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'preserve_partials':True,'no_parameter_tuning':True});raise

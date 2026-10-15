"""One native pretrained sample, then fresh saved-asset inspection; finite owner."""
from paths import JOB,read,save,sha
from trellis2_native_street_v17 import OUT,SEED
import subcell_owner_v1 as lease
import types,threading,subprocess,sys,fcntl

def verify():
    for f,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(3560 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs']
        fn(gpus[0],'native',SEED,'generate','trellis2_native_street_v17.py',[],seconds=2400)
        fn(gpus[1],'native',SEED,'views','trellis2_native_street_v17_views.py',[],seconds=900)
        save(OUT/'COMPLETE.json',{'one_seed_only':True,'no_more_jobs':True,
            'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),'quality_review_pending':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('NATIVE_OWNER',c.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'no_retry_or_seed_search':True});raise

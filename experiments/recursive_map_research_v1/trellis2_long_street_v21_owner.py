"""Single generation followed by fresh-process asset views, bounded occupancy."""
from paths import JOB,read,save
from trellis2_long_street_v21 import OUT,SEED,verify
import subcell_owner_v1 as lease
import types,threading,fcntl,sys,subprocess

def job_runner():
    scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
    fn.__code__=fn.__code__.replace(co_consts=tuple(3560 if c==114900 else c for c in fn.__code__.co_consts))
    return fn

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fn=job_runner();gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs']
        fn(gpus[0],'native',SEED,'generate','trellis2_long_street_v21.py',[],seconds=1800)
        fn(gpus[1],'native',SEED,'views','trellis2_long_street_v21_views.py',[],seconds=900)
        save(OUT/'FIXED_VIEWS_COMPLETE.json',{'new_sample_count':1,'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'road_tour_pending':True,'no_further_generation':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('V21_OWNER',c.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'no_automatic_new_generation':True});raise

"""Render-only continuation; preserve original logs and do not resample models."""
from trellis2_native_street_v17 import OUT,SEED
from trellis2_native_street_v17_owner import verify
from paths import JOB,read,save,sha
import subcell_owner_v1 as lease
import types,threading,subprocess,sys,fcntl

def check():
    verify()
    for f,h in read(OUT/'VIEW_FIX_FREEZE.json').items():assert sha(f)==h

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':check,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(3560 if c==114900 else c for c in fn.__code__.co_consts))
        gpu=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs'][1]
        fn(gpu,'native',SEED,'views_keys_fix1','trellis2_native_street_v17_views_fix1.py',[],seconds=900)
        save(OUT/'COMPLETE_FIXED_VIEWS.json',{'one_model_sample_only':True,'training':False,'no_more_jobs':True,
            'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        assert (OUT/'ASSET.json').exists()
        save(OUT/'VIEW_FIX_FREEZE.json',{str(JOB/n):sha(JOB/n) for n in ['trellis2_native_street_v17_views_fix1.py','trellis2_native_street_v17_view_owner_fix1.py']})
        with (OUT/'view_fix_owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'VIEW_FIX_OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('VIEW_ONLY_OWNER',c.pid,flush=True)
    else:run()

"""Finite original/edit jobs followed by independent geometry renders."""
from paths import JOB,read,save
from trellis2_map_anchor_v20 import OUT,SEED,verify
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,fcntl,sys,subprocess

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
          'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs'][:2]
        for stage,script,seconds in [('generate','trellis2_map_anchor_v20.py',1500),('views','trellis2_map_anchor_v20_views.py',600)]:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                fs=[pool.submit(fn,gpus[i],m,SEED,stage,script,['--mode',m],seconds=seconds) for i,m in enumerate(['correct','edit'])]
                for f in fs:f.result()
            save(OUT/f'{stage.upper()}_COMPLETE.json',{'modes':['correct','edit'],'no_retry':True})
        save(OUT/'GPU_COMPLETE.json',{'gpu_seconds':scope['costs'](),'no_more_jobs':True,'decision_pending_visual_review':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'no_automation':True});print('V20_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'preserve_partials':True,'no_automatic_retry':True});raise

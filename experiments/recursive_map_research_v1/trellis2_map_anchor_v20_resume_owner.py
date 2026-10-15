"""One implementation-only Shape continuation. No Structure resampling."""
from paths import JOB,read,save,sha
from trellis2_map_anchor_v20 import OUT,SEED,verify
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,fcntl,sys,subprocess

def frozen():
    verify()
    for p,h in read(OUT/'RESUME_SOURCE_FREEZE.json').items():assert sha(p)==h

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':frozen,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs'][:2]
        for stage,script,seconds in [('shape_fix1','trellis2_map_anchor_v20_shape_fix1.py',1500),('views','trellis2_map_anchor_v20_views.py',600)]:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                fs=[pool.submit(fn,gpus[i],m,SEED,stage,script,['--mode',m],seconds=seconds) for i,m in enumerate(['correct','edit'])]
                for f in fs:f.result()
            save(OUT/f'{stage.upper()}_COMPLETE.json',{'modes':['correct','edit'],'Structure_reused':True})
        save(OUT/'GPU_COMPLETE.json',{'gpu_seconds':scope['costs'](),'no_more_jobs':True})
        subprocess.run([sys.executable,'-B',str(JOB/'trellis2_map_anchor_v20_package.py')],check=True)

if __name__=='__main__':
    if '--detach' in sys.argv:
        save(OUT/'RESUME_SOURCE_FREEZE.json',{str(JOB/p):sha(JOB/p) for p in ['trellis2_map_anchor_v20_shape_fix1.py','trellis2_map_anchor_v20_resume_owner.py','trellis2_map_anchor_v20_package.py']})
        with (OUT/'resume_owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'RESUME_OWNER.json',{'pid':child.pid,'finite':True,'no_automation':True});print('V20_RESUME_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'RESUME_FAILURE.json',{'error':str(e),'preserve_partials':True});raise

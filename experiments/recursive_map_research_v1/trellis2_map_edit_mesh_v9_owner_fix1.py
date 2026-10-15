"""Same total cap, account failed jobs; resume only after own old owner exits."""
from trellis2_map_edit_mesh_v9_owner import *
from pathlib import Path

def run_fixed():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        for p,h in read(OUT/'FIX1_FREEZE.json').items():assert sha(p)==h
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
          'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(7200 if c==114900 else c for c in fn.__code__.co_consts))
        def task(gpu,seed):
            fn(gpu,'mesh',seed,'generate_fix1','trellis2_map_edit_mesh_v9_resume_fix1.py',['--seed',str(seed)],seconds=2700)
            fn(gpu,'mesh',seed,'reload','trellis2_map_edit_mesh_v9_views.py',['--seed',str(seed)],seconds=900)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(task,GPUS[i],seed) for i,seed in enumerate(SEEDS)]
            for f in fs:f.result()
        save(OUT/'COMPLETE.json',{'training':False,'no_queued_jobs':True,'physical_GPUs':[0,1],
          'gpu_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        assert not Path(f"/proc/{read(OUT/'OWNER.json')['pid']}").exists(),'OLD_OWNER_ACTIVE'
        save(OUT/'FIX1_FREEZE.json',{str(JOB/p):sha(JOB/p) for p in ['trellis2_map_edit_mesh_v9_resume_fix1.py','trellis2_map_edit_mesh_v9_owner_fix1.py']})
        with (OUT/'owner_fix1.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER_FIX1.json',{'pid':child.pid,'no_automation':True,'same_total_cap_including_failed':7200})
        print('MESH_EDIT_FIX1_OWNER',child.pid,flush=True)
    else:
        try:run_fixed()
        except Exception as e:save(OUT/'FAILURE_FIX1.json',{'error':str(e),'type':type(e).__name__,'preserve_all_outputs':True});raise

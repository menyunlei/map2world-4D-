"""Two currently idle A800 UUIDs; no training, one test, bounded occupancy."""
from paths import JOB,read,save,sha
from trellis2_edit_identity_v10_prepare import OUT,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl

def verify():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
          'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        assert 114900 in fn.__code__.co_consts
        fn.__code__=fn.__code__.replace(co_consts=tuple(1800 if c==114900 else c for c in fn.__code__.co_consts))
        def task(gpu,seed):
            fn(gpu,'anchor',seed,'generate','trellis2_edit_identity_v10_run.py',['--seed',str(seed)],seconds=600)
            fn(gpu,'anchor',seed,'reload','trellis2_edit_identity_v10_views.py',['--seed',str(seed)],seconds=240)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(task,GPUS[i],seed) for i,seed in enumerate(SEEDS)]
            for f in fs:f.result()
        subprocess.run([sys.executable,'-B',str(JOB/'trellis2_edit_identity_v10_score.py')],check=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
        save(OUT/'COMPLETE.json',{'training_updates':0,'no_queued_jobs':True,'physical_GPUs':[0,1],
          'gpu_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        assert not (OUT/'OWNER.json').exists()
        with (OUT/'owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'no_automation':True,'occupancy_cap_GPU_seconds':1800})
        print('IDENTITY_EDIT_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'type':type(e).__name__,'preserve_all':True});raise

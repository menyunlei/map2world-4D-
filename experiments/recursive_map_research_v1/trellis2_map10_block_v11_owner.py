"""Two fixed seeds, available authorized GPU0/1, finite inference only."""
from paths import JOB,read,save,sha
from trellis2_map10_block_v11_prepare import OUT,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl
def verify():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)
def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(3600 if c==114900 else c for c in fn.__code__.co_consts))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,GPUS[i],'map10',seed,'generate','trellis2_map10_block_v11_run.py',['--seed',str(seed)],seconds=1500) for i,seed in enumerate(SEEDS)]
            for f in fs:f.result()
        save(OUT/'GENERATION_COMPLETE.json',{'training_updates':0,'no_more_generation_queued':True,'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'no_automation':True});print('MAP10_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'error':str(e),'type':type(e).__name__,'preserve_all':True});raise

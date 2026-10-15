"""Finite matched jobs with existing exclusive GPU leases and occupancy cap."""
from paths import JOB,read,save,sha
from trellis2_coverage_v12_prepare import OUT,ARMS,SEEDS
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,subprocess,sys,fcntl,time

def verify():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)

def bounded():
    scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
      'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
      'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
    assert 114900 in fn.__code__.co_consts
    fn.__code__=fn.__code__.replace(co_consts=tuple(10800 if c==114900 else c for c in fn.__code__.co_consts))
    return fn

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();plan=read(OUT/'PLAN.json');fn=bounded();gpus=plan['GPUs']
        # Preflight weights discarded. Main jobs reset to registered v8 parents.
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpus[i],arm,SEEDS[0],'preflight','trellis2_coverage_v12_train.py',
                ['--arm',arm,'--seed',str(SEEDS[0]),'--preflight'],seconds=180) for i,arm in enumerate(ARMS)]
            for f in fs:f.result()
        for arm in ARMS:
            q=read(OUT/'small'/arm/f'seed{SEEDS[0]}'/'preflight/PREFLIGHT.json')
            assert q['lora_parameters']==245760 and all(x['lora_grad_norm']>0 for x in q['rows'])
        save(OUT/'PREFLIGHTS_COMPLETE.json',{'real_optimizer_updates_each':2,'discarded':True,'reset_before_main':True})
        for seed in SEEDS:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                fs=[pool.submit(fn,gpus[i],arm,seed,'train1000','trellis2_coverage_v12_train.py',
                    ['--arm',arm,'--seed',str(seed)],seconds=1500) for i,arm in enumerate(ARMS)]
                for f in fs:f.result()
        save(OUT/'TRAINING_COMPLETE.json',{'runs':4,'updates_each':1000,'no_more_training_queued':True,
          'actual_GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
          'evaluation_pending':True,'no_automation':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        verify()
        with (OUT/'owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'no_automation':True,'maximum_total_GPU_seconds':10800})
        print('COVERAGE_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'type':type(e).__name__,'error':str(e),'preserve_partials':True});raise

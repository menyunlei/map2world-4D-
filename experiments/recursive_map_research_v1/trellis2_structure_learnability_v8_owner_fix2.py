"""Same frozen experiment, two currently available UUIDs in matched seed batches.

0/1 acquired a different job after registration. Its safety rejection is kept;
this owner never touches it and does not change batch/update/noise semantics.
"""
from paths import JOB,read,save,sha
from trellis2_structure_learnability_v8_prepare import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS as ALL_GPUS
import trellis2_structure_learnability_v8_owner_fix1 as runner
import concurrent.futures,subprocess,sys,os,fcntl

GPUS=ALL_GPUS[2:4];SELF_HASH=sha(__file__)
def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner.verify();assert sha(__file__)==SELF_HASH
        fn=runner.bounded();cap=read(OUT/'PLAN.json')['gpu_budget']['total_occupancy_seconds']
        assert 114900 in fn.__code__.co_consts
        fn.__code__=fn.__code__.replace(co_consts=tuple(cap if v==114900 else v for v in fn.__code__.co_consts))
        assert read(OUT/'CPU_TESTS.json')['returncode']==0
        fn(GPUS[0],'native_prior',0,'check','trellis2_structure_learnability_v8_prior.py',[],seconds=600)
        assert len(read(OUT/'native_prior_check/RESULTS.json')['rows'])==6
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,GPUS[i],arm,SEEDS[0],'preflight','trellis2_structure_learnability_v8_train.py',
                ['--arm',arm,'--seed',str(SEEDS[0]),'--preflight'],seconds=600) for i,arm in enumerate(ARMS)]
            for f in fs:f.result()
        def task(gpu,arm,seed):
            args=['--arm',arm,'--seed',str(seed),'--phase','small']
            if arm=='frozen_structure':fn(gpu,arm,seed,'initial_pure','trellis2_structure_learnability_v8_eval.py',args+['--update','0'],seconds=900)
            fn(gpu,arm,seed,'small_train','trellis2_structure_learnability_v8_train.py',args,seconds=2700)
            for update in [100,1000]:fn(gpu,arm,seed,'small_eval'+str(update),'trellis2_structure_learnability_v8_eval.py',args+['--update',str(update)],seconds=900)
        for seed in SEEDS:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                fs=[pool.submit(task,GPUS[i],arm,seed) for i,arm in enumerate(ARMS)]
                for f in fs:f.result()
        subprocess.run([sys.executable,'-B',str(JOB/'trellis2_structure_learnability_v8_score.py')],cwd=JOB,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        save(OUT/'SMALL_COMPLETE.json',{'jobs':4,'physical_GPUs':[2,3],'two_matched_seed_batches':True,
            'training_batch_and_update_semantics_unchanged':True,'no_further_jobs_queued':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        assert not __import__('pathlib').Path(f"/proc/{read(OUT/'SMALL_FIX1_OWNER.json')['pid']}").exists()
        with (OUT/'small_fix2_owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'SMALL_FIX2_OWNER.json',{'pid':p.pid,'source_sha256':SELF_HASH,'physical_GPUs':[2,3],
            'reason':'0/1 occupied by unrelated PID2420863; no interference','finite_not_automation':True})
        print('LEARNABILITY_FIX2_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'SMALL_FIX2_FAILURE.json',{'type':type(e).__name__,'error':str(e),'preserve_partials':True});raise

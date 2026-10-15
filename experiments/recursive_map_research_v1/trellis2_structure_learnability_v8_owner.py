"""Finite existing UUID-lease orchestration, one shared occupancy cap, no automation."""
from paths import JOB,read,save,sha
from trellis2_structure_learnability_v8_prepare import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys,os,fcntl,time

def frozen():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,('SOURCE_CHANGED',p)

def cpu_tests():
    run=subprocess.run([sys.executable,'-B',str(JOB/'test_trellis2_structure_lora_v8.py')],
        cwd=JOB,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    save(OUT/'CPU_TESTS.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,
        'source_sha256':sha(JOB/'test_trellis2_structure_lora_v8.py')})
    print(run.stdout,run.stderr,flush=True);assert run.returncode==0

def run(phase):
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);frozen()
        fn=bounded();fn.__globals__['NEW']=OUT;fn.__globals__['frozen']=frozen
        fn.__globals__['costs']=lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))
        cap=read(OUT/'PLAN.json')['gpu_budget']['total_occupancy_seconds']
        # Preserve the proven lease/watchdog implementation and change ONLY its
        # previously hardcoded historical occupancy ceiling to this registered cap.
        assert 114900 in fn.__code__.co_consts
        fn.__code__=fn.__code__.replace(co_consts=tuple(cap if v==114900 else v for v in fn.__code__.co_consts))
        if phase=='small':
            assert read(OUT/'CPU_TESTS.json')['returncode']==0
            fn(GPUS[0],'native_prior',0,'check','trellis2_structure_learnability_v8_prior.py',[],seconds=600)
            assert len(read(OUT/'native_prior_check/RESULTS.json')['rows'])==6
            # Actual complete backwards/optimizer steps are discarded and reset.
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(fn,GPUS[i],a,SEEDS[0],'preflight','trellis2_structure_learnability_v8_train.py',
                    ['--arm',a,'--seed',str(SEEDS[0]),'--preflight'],seconds=600) for i,a in enumerate(ARMS)]
                for f in futures:f.result()
        else:assert read(OUT/'SMALL_DECISION.json')['launch_large_matched']
        def task(gpu,arm,seed):
            args=['--arm',arm,'--seed',str(seed),'--phase',phase]
            # Initial generation is evaluated only once per training seed;
            # zero-B LoRA is a verified strict identity, not a second sweep.
            if phase=='small' and arm=='frozen_structure':
                fn(gpu,arm,seed,'initial_pure','trellis2_structure_learnability_v8_eval.py',args+['--update','0'],seconds=900)
            fn(gpu,arm,seed,phase+'_train','trellis2_structure_learnability_v8_train.py',args,
               seconds=2700 if phase=='small' else 2000)
            for update in ([100,1000] if phase=='small' else [500]):
                fn(gpu,arm,seed,phase+'_eval'+str(update),'trellis2_structure_learnability_v8_eval.py',
                    args+['--update',str(update)],seconds=900)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            tasks=[(GPUS[i*2+j],a,s) for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
            fs=[pool.submit(task,*t) for t in tasks]
            for f in fs:f.result()
        if phase=='small':
            subprocess.run([sys.executable,'-B',str(JOB/'trellis2_structure_learnability_v8_score.py')],
                cwd=JOB,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        save(OUT/(phase.upper()+'_COMPLETE.json'),{'phase':phase,'finite':True,'jobs':4,'original_weights_frozen':True,
            'no_further_jobs_queued':True,'manual_output_inspection_before_conditional_phase':True})

if __name__=='__main__':
    phase='large' if '--large' in sys.argv else 'small'
    if '--cpu-tests' in sys.argv:cpu_tests()
    elif '--detach' in sys.argv:
        assert read(OUT/'CPU_TESTS.json')['returncode']==0
        with (OUT/f'{phase}_owner.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__]+(['--large'] if phase=='large' else []),
                cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/f'{phase.upper()}_OWNER.json',{'pid':child.pid,'phase':phase,'finite_not_automation':True,
            'GPUs':GPUS,'max_total_occupancy_seconds':21600})
        print('LEARNABILITY_OWNER',phase,child.pid,flush=True)
    else:
        try:run(phase)
        except Exception as e:
            save(OUT/(phase.upper()+'_FAILURE.json'),{'type':type(e).__name__,'error':str(e),'preserve_partials':True,'no_automatic_retry':True})
            raise

"""One finite evaluation dependency queue; no scheduler and no model training."""
from paths import JOB,read,save,sha
from trellis2_coverage_v12_prepare import OUT,SEEDS
from trellis2_coverage_v12_owner import bounded,verify as verify_training
import sys,subprocess,os,time,fcntl,concurrent.futures

def verify():
    verify_training()
    for p,h in read(OUT/'EVALUATION_FREEZE.json').items():assert sha(p)==h,('EVAL_SOURCE_CHANGED',p)

def register():
    assert not (OUT/'EVALUATION_FREEZE.json').exists()
    protocol={'fixed_structure_cases':40,'generation_seeds':[970201,970202],
      'correct_map_milestones':[0,100,300,1000],'edit_milestones':[0,1000],
      'mismatched_maps':'reuse fixed next-object output at current object pose, identical base noise; no new draws',
      'shared_parent':'evaluate once per seed; same starting weights for both arms',
      'execution_order':'shared parent, both final1000, then both100, then both300; no output-adaptive choices',
      'mesh':'original v9 pair plus six preregistered heldout objects, original/edit, baseline and both final arms, noise970201',
      'render':'independent GLB reload; reuse v9 fixed cameras; neutral geometry display only',
      'hard_total_GPU_seconds':10800,'training_and_preflight_cost_included':True,
      'no_extra_updates':True,'no_Map10':True,'no_materials':True,
      'full_neighbor_mesh_regeneration_not_measured':True,
      'partial_at_limit':'preserve and report incomplete, never call missing evidence success'}
    save(OUT/'EVALUATION_EXECUTION.json',protocol)
    names=['trellis2_coverage_v12_eval.py','trellis2_coverage_v12_score.py','trellis2_coverage_v12_eval_owner.py',
      'trellis2_structure_learnability_v8_eval.py','trellis2_map_edit_mesh_v9_run.py','trellis2_map_edit_mesh_v9_views.py']
    sources={str(JOB/n):sha(JOB/n) for n in names}
    for p in [OUT/'EVALUATION_EXECUTION.json',JOB/'map_object_world_v1/trellis2_map_edit_mesh_v9/PLAN.json']:
        sources[str(p)]=sha(p)
    save(OUT/'EVALUATION_FREEZE.json',sources)

def run():
    with (OUT/'eval_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        until=time.monotonic()+3600
        while not (OUT/'TRAINING_COMPLETE.json').exists():
            assert not (OUT/'FAILURE.json').exists(),'TRAINING_FAILED_NO_EVALUATION_LAUNCH'
            assert time.monotonic()<until,'FINITE_TRAIN_DEPENDENCY_TIMEOUT'
            time.sleep(15)
        # Ensures no live training residency is omitted from the shared cost ledger.
        with (OUT/'owner.lock').open('a') as training_lock:
            fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fn=bounded();fn.__globals__['frozen']=verify
            gpus=read(OUT/'PLAN.json')['GPUs'];assert [g['index'] for g in gpus]==[0,1]
            def task(i,seed):
                fn(gpus[i],'evaluation',seed,'structure','trellis2_coverage_v12_eval.py',
                  ['--seed',str(seed),'--mode','structure'],seconds=4000)
                fn(gpus[i],'evaluation',seed,'mesh','trellis2_coverage_v12_eval.py',
                  ['--seed',str(seed),'--mode','mesh'],seconds=2000)
                fn(gpus[i],'evaluation',seed,'views','trellis2_coverage_v12_eval.py',
                  ['--seed',str(seed),'--mode','views'],seconds=600)
            errors=[]
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(task,i,s) for i,s in enumerate(SEEDS)]
                for future in futures:
                    try:future.result()
                    except Exception as e:errors.append({'type':type(e).__name__,'error':str(e)})
            subprocess.run([sys.executable,'-B',str(JOB/'trellis2_coverage_v12_score.py')],cwd=JOB,
              env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
            save(OUT/'EVALUATION_QUEUE_COMPLETE.json',{'errors':errors,'no_more_jobs':True,
              'status':'COMPLETE' if not errors else 'PARTIAL_STOP_NO_EXTENSION',
              'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--register' in sys.argv:register();print('EVALUATION_FROZEN',flush=True)
    elif '--detach' in sys.argv:
        verify()
        with (OUT/'eval_owner.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
              stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'EVAL_OWNER.json',{'pid':child.pid,'only_GPUs':[0,1],'no_automation':True,
          'bounded_dependency_wait_seconds':3600,'no_training_commands':True})
        print('EVALUATION_OWNER',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'EVAL_FAILURE.json',{'type':type(e).__name__,'error':str(e),'preserve_all':True});raise

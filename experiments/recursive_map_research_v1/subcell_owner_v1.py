"""Finite four-worker matched experiment, bounded physical occupancy and native eval."""
from paths import *
import launch_training as infra
import os,sys,subprocess,fcntl,time,signal,threading,concurrent.futures
NEW=JOB/'subcell_structure_v1';OLD=JOB/'narrow_structure_v1'
GPUS=infra.GPUS+[
    {'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},
    {'index':3,'uuid':'GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d'}]
TASKS=list(zip(GPUS,['baseline','narrow','baseline','narrow'],[1701,1701,1702,1702]))
guard=threading.Lock();active={};halt=threading.Event()


def costs():
    paths=list((JOB/'gpu_cost').glob('*.json'))+list((OLD/'gpu_cost').glob('*.json'))+list((NEW/'gpu_cost').glob('*.json'))
    return sum(read(p)['gpu_seconds'] for p in paths)


def frozen():
    infra.verify()
    for path,digest in read(NEW/'SOURCE_FREEZE.json').items():assert sha(path)==digest,('SOURCE_CHANGED',path)


def run(gpu,arm,seed,tag,script,args,training=False,seconds=4000):
    name=f'subcell_{arm}_{seed}_{tag}';lockpath=JOB/'gpu_leases'/f'gpu{gpu["index"]}.lock'
    with lockpath.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        assert f'{gpu["index"]}, {gpu["uuid"]}' in inv and gpu['uuid'] not in owners,('GPU_NOT_FREE',gpu,owners)
        frozen();assert not halt.is_set(),'SHARED_GPU_CAP_REACHED'
        out=NEW/arm/'runs/g2_b'/f'seed{seed}' if training else NEW/'work'/name
        out.mkdir(parents=True,exist_ok=True)
        for folder in ['tmp','cache','extensions']:(out/folder).mkdir(exist_ok=True)
        command=[sys.executable,'-B','-u',str(JOB/script),*args]
        if training:
            policy=read(infra.policy(out,'g2_b',name))
            policy['read_paths'] += [str(JOB/p) for p in ['subcell_model_v1.py','subcell_runtime_v1.py','subcell_train_v1.py','subcell_preflight_v1.py']]
            policy['read_paths'] += [str(NEW/p) for p in ['PREREGISTRATION.json','SOURCE_FREEZE.json']]
            policy['read_paths'] += [str(OLD/p) for p in ['TARGETS_READY.json','TRAIN_FREEZE.json']]
            policy['read_paths'] += list(read(OLD/'TRAIN_FREEZE.json')['files'])
            policy['read_paths'] += [str(NEW/arm/'TRAINING_SLOTS.json'),str(JOB/'runs/g2_b'/f'seed{seed}/CHECKPOINT_000300.json')]
            policy['read_paths'] += [read(JOB/'runs/g2_b'/f'seed{seed}/CHECKPOINT_000300.json')['path']]
            policy['read_paths'] += [str(NEW/arm/'candidates/g2_b')]
            policy['list_paths'] += [str(NEW/arm/'candidates/g2_b'),str(NEW/arm/'candidates'),str(NEW/arm)]
            pol=NEW/'policies'/f'{name}.json';save(pol,policy)
            command=[sys.executable,'-B','-u',str(JOB/'sandbox.py'),str(pol),str(JOB/script),*args]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'CUDA_HOME':'/usr/local/cuda-11.8',
            'SPCONV_ALGO':'native','ATTN_BACKEND':'flash_attn','VIDEOX_ATTENTION_TYPE':'FLASH_ATTENTION',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(out/'tmp'),
            'XDG_CACHE_HOME':str(out/'cache'),'CUDA_CACHE_PATH':str(out/'cache'),'TORCH_EXTENSIONS_DIR':str(out/'extensions')}
        log=NEW/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x') as handle:
            with guard:
                assert costs()+sum(time.time()-v[0] for v in active.values())<114900,'SHARED_GPU_CAP'
                start=time.time();child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                active[name]=(start,child.pid)
            save(NEW/'audit'/f'{name}_LAUNCH.json',{'pid':child.pid,'gpu':gpu,'command':command,'epoch':start})
            while child.poll() is None:
                with guard:
                    if costs()+sum(time.time()-v[0] for v in active.values())>=114900:halt.set()
                if halt.is_set() or time.time()-start>=seconds:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(5)
            stop=time.time();code=child.wait()
            with guard:
                save(NEW/'gpu_cost'/f'{name}.json',{'name':name,'gpu':gpu,'start_epoch':start,'end_epoch':stop,
                    'gpu_seconds':stop-start,'includes_load_idle':True,'returncode':code})
                active.pop(name)
        assert code==0,(name,code,str(log));print('COMPLETE',name,flush=True)


def cpu(script,args=()):
    subprocess.run([sys.executable,'-B',str(JOB/script),*args],cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})


def parallel(fn):
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(fn,*task) for task in TASKS]
        for future in futures:future.result()


def train(gpu,arm,seed,until,resume=None):
    args=['--arm',arm,'--candidate','g2_b','--seed',str(seed),'--until',str(until),'--attempt','subcell_v1']
    if resume:args+=['--resume',str(resume)]
    run(gpu,arm,seed,f'train{resume or 0}_{until}','subcell_train_v1.py',args,True,1800 if until==20 else 3600 if until==100 else 7200)


def first(gpu,arm,seed):
    run(gpu,arm,seed,'preflight','subcell_preflight_v1.py',['--arm',arm,'--seed',str(seed)],True,1800)
    train(gpu,arm,seed,20)


def evaluate(update):
    for mode in ['structure','native','transmittance']:
        def task(gpu,arm,seed):
            run(gpu,arm,seed,f'{mode}{update}','subcell_eval_v1.py',
                ['--arm',arm,'--mode',mode,'--candidate','g2_b','--seed',str(seed),'--update',str(update)])
        parallel(task)
    cpu('subcell_score_v1.py',['--update',str(update)])


def main():
    with (NEW/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parallel(first);evaluate(20)
        parallel(lambda gpu,arm,seed:train(gpu,arm,seed,100,20));evaluate(100)
        result=read(NEW/'RESULTS_100.json')
        negative=[]
        for seed in [1701,1702]:
            b=next(r for r in result['records'] if r['seed']==seed and r['arm']=='baseline')['summary']
            n=next(r for r in result['records'] if r['seed']==seed and r['arm']=='narrow')['summary']
            worse=all(n['narrow'][k]>b['narrow'][k] for k in ['free_air_wrong_fraction','overhead_column_fraction'])
            negative.append(worse and not next(r for r in result['two_seed_comparison'] if r['seed']==seed)['structural_preservation'])
        if all(negative):
            save(NEW/'DELIVERY.json',{'status':'STOP_AT100_PREDECLARED_FUTILITY','results':'RESULTS_100.json','no_more_training':True})
            return
        parallel(lambda gpu,arm,seed:train(gpu,arm,seed,300,100));evaluate(300)
        # Same candidate weights and noise, only subcell address association altered.
        for seed,gpu in zip([1701,1702],[GPUS[0],GPUS[1]]):
            for mode in ['structure','native','transmittance']:
                run(gpu,'narrow',seed,f'shuffled_{mode}300','subcell_eval_v1.py',
                    ['--arm','narrow','--shuffled','--mode',mode,'--candidate','g2_b','--seed',str(seed),'--update','300'])
        save(NEW/'DELIVERY.json',{'status':'MATCHED_300_COMPLETE','results':'RESULTS_300.json',
            'shuffled_native_outputs':'narrow_shuffled','no_more_training':True,'HUMAN_REVIEW_PENDING':True,'human_not_blocking':True})
        save(JOB/'audit/SUBCELL_FINAL_RESULTS_V1.json',read(NEW/'RESULTS_300.json'))


if __name__=='__main__':
    if '--detach' in sys.argv:
        with (NEW/'owner.log').open('x') as handle:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid}),flush=True)
    else:
        try:main()
        except Exception as exc:
            save(NEW/'FAILURE.json',{'error':str(exc),'type':type(exc).__name__,'epoch':time.time(),'preserve_partials':True})
            raise

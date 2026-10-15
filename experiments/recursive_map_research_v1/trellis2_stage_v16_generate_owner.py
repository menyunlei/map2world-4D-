"""Four concurrent object jobs, then the second fixed training seed. No retries."""
from paths import JOB,read,save,sha
from trellis2_stage_v16 import OUT,SEEDS
from trellis2_stage_v16_repair_owner import bounded
import concurrent.futures,subprocess,sys,fcntl

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUT/'REPAIR_TRAINING_COMPLETE.json')['updates_each']==100
        for f,h in read(OUT/'GENERATION_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)
        fn=bounded();p=read(OUT/'PLAN.json');errors=[]
        for seed in SEEDS:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                fs=[pool.submit(fn,p['GPUs'][i],'generate',seed,f'object{i}','trellis2_stage_v16_generate.py',
                    ['--seed',str(seed),'--index',str(i)],seconds=900) for i in range(4)]
                for f in fs:
                    try:f.result()
                    except Exception as e:errors.append(str(e))
        save(OUT/'GENERATION_COMPLETE.json',{'errors':errors,'seeds':SEEDS,'all_object_slots_retained':True,
            'no_retry_or_more_training':True,'gpu_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_stage_v16_generate.py','trellis2_stage_v16_generate_owner.py']}
        for seed in SEEDS:
            ck=read(OUT/'repair'/f'seed{seed}/CHECKPOINT_000100.json');freeze[ck['path']]=ck['sha256']
        save(OUT/'GENERATION_FREEZE.json',freeze)
        with (OUT/'generate_owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'GENERATION_OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('GENERATE_OWNER',c.pid,flush=True)
    else:run()

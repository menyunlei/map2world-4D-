"""Concurrent two saved-asset street renderers and two tiny native SS checks."""
from paths import JOB,read,save,sha
from trellis2_stage_v16 import OUT,SEEDS
from trellis2_stage_v16_repair_owner import bounded
import concurrent.futures,subprocess,sys,fcntl

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for f,h in read(OUT/'POST_FREEZE.json').items():assert sha(f)==h
        fn=bounded();gpus=read(OUT/'PLAN.json')['GPUs'];errors=[]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            fs=[]
            for i,seed in enumerate(SEEDS):
                fs.append(pool.submit(fn,gpus[i],'street',seed,'matched','trellis2_stage_v16_scene.py',['--seed',str(seed)],seconds=1200))
                fs.append(pool.submit(fn,gpus[i+2],'check',seed,'pure_noise','trellis2_stage_v16_check.py',['--seed',str(seed)],seconds=900))
            for f in fs:
                try:f.result()
                except Exception as e:errors.append(str(e))
        save(OUT/'POST_COMPLETE.json',{'errors':errors,'no_more_GPU_jobs':True,
            'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__=='__main__':
    if '--detach' in sys.argv:
        assert (OUT/'GENERATION_COMPLETE.json').exists()
        save(OUT/'POST_FREEZE.json',{str(JOB/n):sha(JOB/n) for n in ['trellis2_stage_v16_scene.py','trellis2_stage_v16_check.py','trellis2_stage_v16_post_owner.py']})
        with (OUT/'post_owner.log').open('x') as h:
            c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'POST_OWNER.json',{'pid':c.pid,'finite':True,'no_automation':True});print('POST_OWNER',c.pid,flush=True)
    else:run()

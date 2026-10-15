from trellis2_long_street_v21 import OUT,SEED
from trellis2_long_street_v21_owner import job_runner
from paths import JOB,read,save,sha
import fcntl,sys,subprocess

if '--detach' in sys.argv:
    save(OUT/'TOUR_FREEZE.json',{str(p):sha(p) for p in [OUT/'STREET_ROUTE.json',JOB/'trellis2_long_street_v21_tour.py',JOB/'trellis2_long_street_v21_tour_owner.py']})
    with (OUT/'tour_owner.log').open('x') as h:
        c=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'TOUR_OWNER.json',{'pid':c.pid,'no_new_generation':True});print('V21_TOUR_OWNER',c.pid,flush=True)
else:
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p,h in read(OUT/'TOUR_FREEZE.json').items():assert sha(p)==h
        gpu=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs'][1]
        job_runner()(gpu,'native',SEED,'tour','trellis2_long_street_v21_tour.py',[],seconds=900)
        save(OUT/'COMPLETE.json',{'no_more_GPU_jobs':True,'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

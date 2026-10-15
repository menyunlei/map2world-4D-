"""Finite scene-fitting owner; no scheduler, no map-model training."""
from paths import *
import concurrent.futures,sys,time,subprocess,os,fcntl
import dice_generated_state_owner_v1 as infra
PHASE=JOB/'map_object_world_v1/world_composition_v1'
GPUS=[{'index':0,'uuid':'GPU-e0a56bf9-85ef-c658-67fa-457b76585254'},
      {'index':1,'uuid':'GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa'}]
def bounded():
    fn=infra.runner();fn.__globals__['NEW']=PHASE;fn.__globals__['START']=time.time()
    def check():
        for path,digest in read(PHASE/'SOURCE_FREEZE.json').items():assert sha(path)==digest,('SOURCE_CHANGED',path)
    fn.__globals__['frozen']=check
    return fn
def run():
    with (PHASE/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fn=bounded()
        def prep(gpu,seed):
            fn(gpu,'composition',seed,'data','map_object_composition_entry_v1.py',['prepare','--seed',str(seed)],seconds=1200)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for f in [pool.submit(prep,g,s) for g,s in zip(GPUS,[832001,832002])]:f.result()
        fn(GPUS[0],'composition',832001,'preflight10','map_object_composition_entry_v1.py',['fit','--seed','832001','--steps','10','--preflight'],seconds=1800)
        save(PHASE/'PREFLIGHT_COMPLETE.json',{'optimizer_updates':10,'official_trainer':True,'map_model_training':False})
        def fit(gpu,seed):
            fn(gpu,'composition',seed,'fit8000','map_object_composition_entry_v1.py',['fit','--seed',str(seed),'--steps','8000'],seconds=7200)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for f in [pool.submit(fit,g,s) for g,s in zip(GPUS,[832001,832002])]:f.result()
        save(PHASE/'FITS_COMPLETE.json',{'seeds':[832001,832002],'updates_per_scene':8000,'map_model_training':False,'next':'fresh fixed-camera asset reload'})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (PHASE/'owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(PHASE/'OWNER.json',{'pid':p.pid,'finite_not_automation':True,'task_kind':'per_scene_3DGS_fitting','map_model_training':False})
        print('COMPOSITION_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as exc:
            save(PHASE/'OWNER_FAILURE.json',{'type':type(exc).__name__,'error':str(exc),'preserve_partials':True})
            raise

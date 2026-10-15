"""Finite two-card diagnosis owner; no optimizer and no model search."""
from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_ss_failure_v4'

def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_ss_failure_v4_probe.py','trellis2_map_ss_failure_v4_owner.py']}
    save(OUT/'SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(fn,GPUS[i],'ss_failure',seed,'fixed_weights',
            'trellis2_map_ss_failure_v4_probe.py',['--seed',str(seed)],seconds=600) for i,seed in enumerate([953101,953102])]
        for j in jobs:j.result()
    save(OUT/'DIAGNOSIS_COMPLETE.json',{'optimizer_updates':0,'seeds':[953101,953102]})

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'optimizer_updates':0})
        print('SS_DIAGNOSIS_OWNER',child.pid,flush=True)
    else:run()

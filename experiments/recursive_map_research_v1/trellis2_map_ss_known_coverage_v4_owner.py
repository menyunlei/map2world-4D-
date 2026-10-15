from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_ss_failure_v4'
def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_ss_known_coverage_v4.py','trellis2_map_ss_known_coverage_v4_owner.py']}
    save(OUT/'COVERAGE_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(fn,GPUS[i],'known_coverage',seed,'fixed_weights','trellis2_map_ss_known_coverage_v4.py',['--seed',str(seed)],seconds=300) for i,seed in enumerate([953101,953102])]
        for f in futures:f.result()
    save(OUT/'COVERAGE_COMPLETE.json',{'optimizer_updates':0,'no_new_training':True})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'coverage_owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'COVERAGE_OWNER.json',{'pid':child.pid,'finite':True,'no_optimizer':True})
        print('COVERAGE_DIAGNOSIS_OWNER',child.pid,flush=True)
    else:run()

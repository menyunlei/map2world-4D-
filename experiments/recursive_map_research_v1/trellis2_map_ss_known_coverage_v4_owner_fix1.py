"""Only relocate never-launched diagnosis to available GPU3; never touch other owners."""
from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_ss_failure_v4'
def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_ss_known_coverage_v4.py','trellis2_map_ss_known_coverage_v4_owner_fix1.py']}
    save(OUT/'COVERAGE_SOURCE_FREEZE_FIX1.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    for seed in [953101,953102]:
        assert not (OUT/f'KNOWN_COVERAGE_SEED{seed}.json').exists()
        fn(GPUS[3],'known_coverage_available',seed,'fixed_weights','trellis2_map_ss_known_coverage_v4.py',['--seed',str(seed)],seconds=300)
    save(OUT/'COVERAGE_COMPLETE.json',{'optimizer_updates':0,'no_new_training':True,'GPU3_only':True,
        'original_owner_refused_unrelated_live_GPUs0_1':True,'original_failed_owner_and_logs_preserved':True})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'coverage_owner_fix1.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'COVERAGE_OWNER_FIX1.json',{'pid':child.pid,'finite':True,'no_optimizer':True,'gpu':GPUS[3]})
        print('AVAILABLE_GPU_COVERAGE_OWNER',child.pid,flush=True)
    else:run()

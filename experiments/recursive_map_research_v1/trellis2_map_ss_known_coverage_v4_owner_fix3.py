"""Corrected official-noise attribution on the available, verified GPU3."""
from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_ss_failure_v4/cpu_noise_fix1'
def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_ss_known_coverage_v4.py','trellis2_map_ss_known_coverage_v4_fix2.py','trellis2_official_ss_noise_fix1.py','trellis2_map_ss_known_coverage_v4_owner_fix3.py']};save(OUT/'SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    for seed in [953101,953102]:
        fn(GPUS[3],'known_coverage_official',seed,'fixed_weights','trellis2_map_ss_known_coverage_v4_fix2.py',['--seed',str(seed)],seconds=300)
    save(OUT/'COVERAGE_COMPLETE.json',{'optimizer_updates':0,'matched_original_CPU_noise':True,'previous_CUDA_noise_numbers_not_used_for_causal_claim':True})
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'optimizer_updates':0});print('EXACT_NOISE_COVERAGE_OWNER',child.pid,flush=True)
    else:run()

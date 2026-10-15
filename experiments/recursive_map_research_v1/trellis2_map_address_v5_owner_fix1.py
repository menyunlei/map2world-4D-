from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5/cpu_noise_fix1'
def run():
    save(OUT/'PLAN.json',{'sole_model_change':'content confidence no longer blanks valid world query XYZ',
        'harness_correction':'official CPU torch.randn then move to CUDA; earlier CUDA-noise results retained but not a historical matched replay',
        'normalization_hypothesis_rejected':True,'arms':['baseline','address_valid'],'seeds':[953101,953102],
        'optimizer_updates':0,'native_weights_data_inputs_CFG_samplers_unchanged':True,
        'fixed_cases':['graph_0111_a0','graph_0112_a0','correct','edit'],'no_more_model_interventions':True})
    files=['trellis2_map_address_validity_v5.py','trellis2_map_address_v5_probe.py','trellis2_map_address_v5_probe_fix1.py','trellis2_official_ss_noise_fix1.py','trellis2_map_address_v5_owner_fix1.py']
    freeze={str(JOB/f):sha(JOB/f) for f in files};save(OUT/'SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(fn,GPUS[i*2+j],'address_'+arm,seed,'official_noise','trellis2_map_address_v5_probe_fix1.py',['--arm',arm,'--seed',str(seed)],seconds=300) for i,arm in enumerate(['baseline','address_valid']) for j,seed in enumerate([953101,953102])]
        for f in futures:f.result()
    save(OUT/'PROBES_COMPLETE.json',{'optimizer_updates':0,'four_matched_jobs':True,'same_official_SS_noise_contract':True})
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'optimizer_updates':0});print('OFFICIAL_NOISE_ADDRESS_OWNER',child.pid,flush=True)
    else:run()

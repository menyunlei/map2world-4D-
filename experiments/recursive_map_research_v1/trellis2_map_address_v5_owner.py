"""One registered confidence/address fix; four finite matched jobs, no training."""
from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5'
def run():
    save(OUT/'PLAN.json',{'single_adoptable_correction':'Retain valid query coordinates through final content-confidence gate; unknown map values remain masked',
        'normalization_hypothesis_v4_rejected_not_adopted':True,'arms':['baseline','address_valid'],'seeds':[953101,953102],
        'optimizer_updates':0,'checkpoint':'same existing separated v3 branch_000300 each matched seed',
        'native_backbone_codec_samplers_noise_maps_unchanged':True,'no_training_or_loss_change':True,
        'fixed_cases':['graph_0111_a0','graph_0112_a0','correct','edit'],'no_more_candidate_search':True})
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_address_validity_v5.py','trellis2_map_address_v5_probe.py','trellis2_map_address_v5_owner.py']}
    save(OUT/'SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'address_'+arm,seed,'fixed_test','trellis2_map_address_v5_probe.py',['--arm',arm,'--seed',str(seed)],seconds=300)
            for i,arm in enumerate(['baseline','address_valid']) for j,seed in enumerate([953101,953102])]
        for job in jobs:job.result()
    save(OUT/'PROBES_COMPLETE.json',{'optimizer_updates':0,'four_matched_jobs':True})
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'no_optimizer':True});print('ADDRESS_MATCHED_OWNER',child.pid,flush=True)
    else:run()

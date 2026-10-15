"""Four matched fixed-weight jobs for one suspected condition interface issue."""
from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_condition_contract_v4'

def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_condition_contract_v4_probe.py','trellis2_map_condition_contract_v4_owner.py']}
    save(OUT/'SOURCE_FREEZE.json',freeze)
    save(OUT/'PLAN.json',{'arms':['raw','native_norm'],'seeds':[953101,953102],'optimizer_updates':0,
        'single_change':'normalize map geometry tokens with the same non-affine per-token LayerNorm used by native DINO extractor',
        'sampler_checkpoint_inputs_noise_unchanged':True,'cases_before_new_outputs':['graph_0064_a0','graph_0065_a0','graph_0111_a0','graph_0112_a0','map10_correct','map10_edit'],
        'success_not_claimed_from_intermediate_metrics':True,'no_automatic_training_or_more_interventions':True})
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'condition_'+arm,seed,'fixed_test',
            'trellis2_map_condition_contract_v4_probe.py',['--arm',arm,'--seed',str(seed)],seconds=600)
            for i,arm in enumerate(['raw','native_norm']) for j,seed in enumerate([953101,953102])]
        for job in jobs:job.result()
    save(OUT/'PROBES_COMPLETE.json',{'optimizer_updates':0,'fixed_four_jobs_complete':True})

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':child.pid,'finite':True,'optimizer_updates':0,'GPUs':GPUS})
        print('CONDITION_CONTRACT_OWNER',child.pid,flush=True)
    else:run()

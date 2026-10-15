"""Four bounded native asset/reload pairs; original weights and historical outputs intact."""
from paths import JOB,read,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys,shutil,numpy as np
OUT=JOB/'map_object_world_v1/trellis2_map_condition_contract_v4'
OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'

def run():
    assert (OUT/'PROBES_COMPLETE.json').exists()
    shutil.copytree(OLD/'real_input',OUT/'real_input')
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_condition_contract_v4_assets.py','trellis2_map_condition_contract_v4_views.py','trellis2_map_condition_contract_v4_delivery.py']}
    comparisons=[]
    for seed in [953101,953102]:
        for mode in ['correct','edit']:
            raw=OUT/'raw'/f'seed{seed}'/('map10_'+mode)/'SS_GENERATED.npz'
            historical=OLD/'separated'/f'seed{seed}'/'assets'/mode/'SS_GENERATED.npz'
            with np.load(raw) as r,np.load(historical) as h:
                delta=float(np.abs(r['latent']-h['latent']).max())
                equal=bool(np.array_equal(r['support'],h['native_support64']))
            comparisons.append({'seed':seed,'mode':mode,'historical_support_identical':equal,'latent_max_abs':delta,
                'baseline_native_asset':str(historical.parent/'world_basecolor.glb')})
            assert equal and delta==0,('HISTORICAL_BASELINE_NOT_EXACTLY_MATCHED',seed,mode,delta)
    save(OUT/'BASELINE_REUSE_VERIFIED.json',{'comparisons':comparisons,'no_duplicate_baseline_generation':True})
    save(OUT/'DELIVERY_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    def pair(index,seed,mode):
        scope='norm_asset_'+mode
        fn(GPUS[index],scope,seed,'native_decode','trellis2_map_condition_contract_v4_assets.py',['--seed',str(seed),'--mode',mode],seconds=600)
        folder=OUT/'separated'/f'seed{seed}'/'assets'/mode
        if (folder/'EMPTY_NATIVE_SUPPORT.json').exists():return
        fn(GPUS[index],scope,seed,'independent_views','trellis2_map_condition_contract_v4_views.py',
            ['--arm','separated','--seed',str(seed),'--mode',mode,'--out',str(OUT)],seconds=180)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(pair,i*2+j,seed,mode) for i,seed in enumerate([953101,953102]) for j,mode in enumerate(['correct','edit'])]
        for job in jobs:job.result()
    save(OUT/'DELIVERY_COMPLETE.json',{'new_model_training':0,'four_native_cases_processed':True,
        'quality_pending_actual_image_inspection':True,'no_more_intervention_or_automatic_training':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'delivery_owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'DELIVERY_OWNER.json',{'pid':child.pid,'finite':True,'no_optimizer':True})
        print('CONDITION_DELIVERY_OWNER',child.pid,flush=True)
    else:run()

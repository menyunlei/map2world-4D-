from paths import JOB,save,sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys,shutil,numpy as np
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5';OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'
def run():
    assert (OUT/'PROBES_COMPLETE.json').exists();shutil.copytree(OLD/'real_input',OUT/'real_input')
    comparisons=[]
    for seed in [953101,953102]:
        for mode in ['correct','edit']:
            with np.load(OUT/'baseline'/f'seed{seed}'/'support'/mode/'SS_GENERATED.npz') as b,np.load(OLD/'separated'/f'seed{seed}'/'assets'/mode/'SS_GENERATED.npz') as h:
                equal=bool(np.array_equal(b['native_support64'],h['native_support64']));delta=float(np.abs(b['latent']-h['latent']).max())
            assert equal and delta==0,('BASELINE_CHANGED',seed,mode,delta)
            comparisons.append({'seed':seed,'mode':mode,'same_support':equal,'latent_max_abs':delta,'existing_asset':str(OLD/'separated'/f'seed{seed}'/'assets'/mode/'world_basecolor.glb')})
    save(OUT/'BASELINE_REUSE_VERIFIED.json',{'comparisons':comparisons,'baseline_not_regenerated':True})
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_address_v5_assets.py','trellis2_map_address_v5_views.py','trellis2_map_address_v5_delivery.py','trellis2_map_address_validity_v5.py']};save(OUT/'DELIVERY_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    results=[]
    def pair(index,seed,mode):
        scope='address_asset_'+mode
        try:
            fn(GPUS[index],scope,seed,'native_decode','trellis2_map_address_v5_assets.py',['--seed',str(seed),'--mode',mode],seconds=900)
            if (OUT/'separated'/f'seed{seed}'/'assets'/mode/'EMPTY_NATIVE_SUPPORT.json').exists():return {'seed':seed,'mode':mode,'status':'EMPTY'}
            fn(GPUS[index],scope,seed,'reload','trellis2_map_address_v5_views.py',['--arm','separated','--seed',str(seed),'--mode',mode,'--out',str(OUT)],seconds=240)
            return {'seed':seed,'mode':mode,'status':'ASSET_AND_INDEPENDENT_VIEWS'}
        except Exception as error:
            return {'seed':seed,'mode':mode,'status':'FAILED_PRESERVED','error':str(error),'no_retry':True}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(pair,i*2+j,seed,mode) for i,seed in enumerate([953101,953102]) for j,mode in enumerate(['correct','edit'])]
        results=[j.result() for j in jobs]
    save(OUT/'DELIVERY_COMPLETE.json',{'cases':results,'no_more_corrections_or_training':True,'quality_requires_actual_inspection':True})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'delivery_owner.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'DELIVERY_OWNER.json',{'pid':child.pid,'finite':True,'optimizer_updates':0});print('ADDRESS_NATIVE_DELIVERY_OWNER',child.pid,flush=True)
    else:run()

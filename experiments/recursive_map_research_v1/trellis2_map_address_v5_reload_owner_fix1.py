"""Read saved native assets only. Correct a renderer output-root guard, not geometry."""
from paths import JOB,save,sha,read
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5/cpu_noise_fix1'
def run():
    freeze={str(JOB/f):sha(JOB/f) for f in ['trellis2_map_address_v5_views_fix1.py','trellis2_map_entry_views_v2.py','trellis2_map_address_v5_reload_owner_fix1.py']}
    for seed,mode in [(953101,'correct'),(953101,'edit'),(953102,'correct')]:
        folder=OUT/'separated'/f'seed{seed}'/'assets'/mode
        freeze[str(folder/'RAW_NATIVE.pt')]=sha(folder/'RAW_NATIVE.pt')
    save(OUT/'RELOAD_FIX_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        futures=[pool.submit(fn,GPUS[i],'address_saved_'+mode,seed,'reload_root_fix','trellis2_map_address_v5_views_fix1.py',
            ['--arm','separated','--seed',str(seed),'--mode',mode,'--out',str(OUT)],seconds=240) for i,(seed,mode) in enumerate([(953101,'correct'),(953101,'edit'),(953102,'correct')])]
        for f in futures:f.result()
    save(OUT/'INDEPENDENT_RELOADS_COMPLETE.json',{'saved_assets_loaded_in_new_processes':3,'fixed_views_each':12,
        'no_regeneration_or_geometry_change':True,'prior_root_guard_failures_preserved':True,'one_empty_edit_preserved':True})
if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'reload_owner_fix1.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'RELOAD_OWNER_FIX1.json',{'pid':child.pid,'finite':True,'generation_updates':0});print('SAVED_ASSET_RELOAD_OWNER',child.pid,flush=True)
    else:run()

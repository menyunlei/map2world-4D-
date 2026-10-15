"""At most four bounded native generation jobs and four independent reloads."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys

def run():
    assert read(OUT/'MATERIAL_GATE.json')['run_native_fine_geometry_and_texture']
    assert (OUT/'CAMERA_PROTOCOL.json').exists()
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_object_generate_v6.py','trellis2_map_object_views_v6.py','trellis2_map_object_assets_owner_v6.py','trellis2_map_object_contract_v6.py']}
    freeze[str(OUT/'CAMERA_PROTOCOL.json')]=sha(OUT/'CAMERA_PROTOCOL.json')
    save(OUT/'ASSET_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    for phase,script,seconds in [('assets','trellis2_map_object_generate_v6.py',1200),('reload','trellis2_map_object_views_v6.py',300)]:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            jobs=[pool.submit(fn,GPUS[i*2+j],'object_'+a,s,phase,script,
                 ['--arm',a,'--seed',str(s)],seconds=seconds)
                 for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
            for j in jobs:j.result()
    save(OUT/'ASSET_DELIVERY_COMPLETE.json',{'arms':ARMS,'seeds':SEEDS,'same_pose_footprint_edit':True,
         'independent_complete_GLBS_reloaded':True,'no_more_training_or_seeds':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'assets_owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'ASSET_OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS,'time_limit_each_generation':1200});print('ASSET_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'ASSET_FAILURE.json',{'error':str(e),'all_outputs_preserved':True});raise

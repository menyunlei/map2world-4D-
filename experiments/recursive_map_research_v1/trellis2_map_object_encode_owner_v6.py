"""Four bounded native codec jobs, followed by immutable paired manifest."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys

def run():
    names=['trellis2_map_object_data_v6.py','trellis2_map_object_data_v6_fix1.py',
        'trellis2_map_object_encode_v6.py','trellis2_map_object_encode_owner_v6.py']
    freeze={str(JOB/n):sha(JOB/n) for n in names}
    freeze[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    save(OUT/'ENCODE_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,g,'object_contract',953101+i,'encode'+str(i),
              'trellis2_map_object_encode_v6.py',['--worker',str(i)],seconds=900) for i,g in enumerate(GPUS)]
        for j in jobs:j.result()
    targets={r['key']:read(OUT/'target_records'/f"{r['key']}.json") for r in read(OUT/'SOURCE_GEOMETRIES.json')['records']}
    records=[{**r,**{k:targets[r['geometry_key']][k] for k in ['native_target','native_target_sha256','shape_cells','SS_codec_reconstruction_IoU']}}
             for r in read(OUT/'PLAN.json')['records']]
    save(OUT/'PAIRED_NATIVE_MANIFEST.json',{'records':records,'unique_targets':len(targets),
         'TRAIN_objects':sum(r['split']=='adapter_TRAIN' for r in records),
         'HELDOUT_objects':sum(r['split']=='adapter_HELDOUT' for r in records),
         'valid_per_object_native_SS_and_32D_shape':True,'model_training_updates':0})
    print('PAIRED_OBJECT_CODEC_COMPLETE',len(records),flush=True)

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'encode_owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'ENCODE_OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS,'no_training':True})
        print('ENCODE_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:
            save(OUT/'ENCODE_FAILURE.json',{'type':type(e).__name__,'error':str(e),'partials_preserved':True});raise

from paths import JOB,read,save,sha
from trellis2_object_confidence_prepare_v7 import OUT
from trellis2_map_object_data_v6_fix1 import ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys

def run():
    assert read(OUT/'CONFIDENCE_CPU_TESTS.json')['passed']
    save(OUT/'PAIRED_NATIVE_MANIFEST.json',{'records':[],
        'no_training_or_target_inference':'this one-fix replay uses real inputs only, no repeated full heldout sweep'})
    files=['trellis2_object_confidence_fix_v7.py','trellis2_object_confidence_prepare_v7.py',
        'trellis2_object_confidence_structure_v7.py','trellis2_map_object_structure_v6.py','trellis2_object_confidence_owner_v7.py']
    freeze={str(JOB/n):sha(JOB/n) for n in files}
    for n in ['PLAN.json','REAL_OBJECT_INPUTS.json']:freeze[str(OUT/n)]=sha(OUT/n)
    save(OUT/'SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'confidence_'+a,s,'structure','trellis2_object_confidence_structure_v7.py',
              ['--arm',a,'--seed',str(s)],seconds=300) for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
        for j in jobs:j.result()
    save(OUT/'STRUCTURE_COMPLETE_ALL.json',{'arms':ARMS,'seeds':SEEDS,'training_updates':0,
        'single_input_fix_only':True,'historical_baseline_reused':True,'no_material_started':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS,'no_training':True});print('CONFIDENCE_FIX_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'type':type(e).__name__,'error':str(e),'partials_preserved':True});raise

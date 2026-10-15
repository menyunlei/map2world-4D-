"""Finite four-card diagnosis, zero optimizer/model updates."""
from paths import JOB,read,save,sha
from trellis2_map_geometry_entry_v2_owner import GPUS
from trellis2_map_object_data_v6_fix1 import ARMS,SEEDS,OUT as OLD
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys
OUT=JOB/'map_object_world_v1/trellis2_structure_path_diagnosis_v7'

def run():
    assert (OLD/'SESSION_STATE_FINAL.json').exists()
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_structure_path_diagnosis_v7.py','trellis2_structure_path_owner_v7.py',
        'trellis2_map_object_contract_v6.py','trellis2_map_geometry_tokens_v3_generate.py','trellis2_map_learning_train_v1.py']}
    for a in ARMS:
        for s in SEEDS:
            ck=read(OLD/a/f'seed{s}'/'CHECKPOINT_000300.json');freeze[ck['path']]=ck['sha256']
    save(OUT/'SOURCE_FREEZE.json',freeze)
    save(OUT/'PLAN.json',{'purpose':'fixed-weight map input -> injection -> velocity -> clean Structure localization',
        'arms':ARMS,'seeds':SEEDS,'cases':'2 existing TRAIN families, 1 adapter-heldout family; fixed Map10 object0/object3 plus footprint edit',
        'training_updates':0,'no_material_or_fine_generation':True,'seconds_each':300,
        'no_tuning_or_model_switch':True,'no_change_until_evidence':True})
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'path_'+a,s,'trace','trellis2_structure_path_diagnosis_v7.py',
            ['--arm',a,'--seed',str(s)],seconds=300) for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
        for j in jobs:j.result()
    save(OUT/'TRACE_COMPLETE.json',{'arms':ARMS,'seeds':SEEDS,'optimizer_updates':0,'sources_unchanged':True})

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS,'training':False});print('PATH_DIAGNOSIS_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'FAILURE.json',{'type':type(e).__name__,'error':str(e),'partials_preserved':True});raise

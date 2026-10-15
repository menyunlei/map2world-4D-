from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys

def run():
    assert (OUT/'TRAINING_COMPLETE.json').exists()
    frozen={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_object_structure_v6.py','trellis2_map_object_structure_owner_v6.py','trellis2_map_object_contract_v6.py']}
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in frozen.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'object_'+a,s,'structure',
              'trellis2_map_object_structure_v6.py',['--arm',a,'--seed',str(s)],seconds=600)
              for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
        for j in jobs:j.result()
    save(OUT/'STRUCTURE_SCREEN_COMPLETE.json',{'arms':ARMS,'seeds':SEEDS,'material_generation_not_started':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'structure_owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'STRUCTURE_OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS});print('STRUCTURE_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:save(OUT/'STRUCTURE_FAILURE.json',{'error':str(e),'partials_preserved':True});raise

"""One final four-job training screen; genuine two-stage microbatch before launch."""
from paths import JOB,read,save,sha
from trellis2_map_geometry_tokens_v3_prepare import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures
import subprocess
import sys


def run():
    assert len(read(OUT/'PAIRED_NATIVE_MANIFEST.json')['records'])==120
    files=['trellis2_map_geometry_tokens_v3_train.py','trellis2_map_geometry_tokens_v3_train_owner.py',
           'trellis2_map_geometry_token_module_v3.py','trellis2_map_object_module_v1.py']
    freeze={str(JOB/name):sha(JOB/name) for name in files}
    for name in ['PLAN.json','PAIRED_NATIVE_MANIFEST.json']:
        freeze[str(OUT/name)]=sha(OUT/name)
    save(OUT/'TRAIN_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():
            assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(fn,GPUS[i*2],'geometry_tokens_'+arm,SEEDS[0],'preflight',
              'trellis2_map_geometry_tokens_v3_train.py',['--arm',arm,'--seed',str(SEEDS[0]),'--preflight'],seconds=300)
              for i,arm in enumerate(ARMS)]
        for job in jobs:
            job.result()
    save(OUT/'PREFLIGHTS_COMPLETE.json',{'native_SS_and_native_ShapeFlow_actual_optimizers_pass':True,
         'discarded_updates_each':2,'same_registered_start_used_for_primary_runs':True})
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'geometry_tokens_'+arm,seed,'train300',
              'trellis2_map_geometry_tokens_v3_train.py',['--arm',arm,'--seed',str(seed)],seconds=1800)
              for i,arm in enumerate(ARMS) for j,seed in enumerate(SEEDS)]
        for job in jobs:
            job.result()
    save(OUT/'TRAINING_COMPLETE.json',{'arms':ARMS,'seeds':SEEDS,'additional_updates_each':300,
         'total_primary_updates':1200,'no_third_mechanism_or_extension':True})


if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'train_owner.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'TRAIN_OWNER.json',{'pid':process.pid,'finite':True,'GPUs':GPUS,'max_updates':300,
             'no_automation':True,'no_base_model_training':True})
        print('JOINT_GEOMETRY_TRAIN_OWNER',process.pid,flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(OUT/'TRAIN_FAILURE.json',{'type':type(error).__name__,'error':str(error),
                 'all_outputs_preserved':True,'no_automatic_budget_extension':True})
            raise

"""One finite same-seed native asset comparison; no extra optimizer/search."""
from paths import JOB,read,save,sha
from trellis2_map_geometry_tokens_v3_prepare import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures
import subprocess
import sys


def run():
    assert read(OUT/'TRAINING_COMPLETE.json')['total_primary_updates']==1200
    files=['trellis2_map_geometry_tokens_v3_generate.py','trellis2_map_geometry_tokens_v3_views.py',
           'trellis2_map_entry_views_v2.py','trellis2_map_geometry_token_module_v3.py',
           'trellis2_map_geometry_tokens_v3_delivery_owner.py']
    freeze={str(JOB/name):sha(JOB/name) for name in files}
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint=read(OUT/arm/f'seed{seed}/CHECKPOINT_000300.json')
            assert sha(checkpoint['path'])==checkpoint['sha256']
            freeze[checkpoint['path']]=checkpoint['sha256']
    save(OUT/'DELIVERY_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():
            assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    def pair(arm,seed,gpu):
        fn(gpu,'geometry_tokens_'+arm,seed,'assets','trellis2_map_geometry_tokens_v3_generate.py',
           ['--arm',arm,'--seed',str(seed)],seconds=900)
        modes=['correct','edit']+(['shuffled'] if arm=='separated' else [])
        for mode in modes:
            directory=OUT/arm/f'seed{seed}/assets'/mode
            if (directory/'EMPTY_NATIVE_SUPPORT.json').exists():continue
            fn(gpu,'geometry_tokens_'+arm,seed,'views_'+mode,'trellis2_map_geometry_tokens_v3_views.py',
               ['--arm',arm,'--seed',str(seed),'--mode',mode,'--out',str(OUT)],seconds=180)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(pair,arm,seed,GPUS[i*2+j]) for i,arm in enumerate(ARMS) for j,seed in enumerate(SEEDS)]
        for job in jobs:job.result()
    save(OUT/'DELIVERY_COMPLETE.json',{'all_fixed_cases_processed':True,
         'no_more_training_or_mechanism_expansion':True,'visual_acceptance_not_claimed_from_export':True})


if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'delivery_owner.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'DELIVERY_OWNER.json',{'pid':process.pid,'finite':True,'no_automation':True,
             'no_more_training':True,'fixed_two_seeds':SEEDS})
        print('SHARED_GEOMETRY_DELIVERY_OWNER',process.pid,flush=True)
    else:
        try:run()
        except Exception as error:
            save(OUT/'DELIVERY_FAILURE.json',{'type':type(error).__name__,'error':str(error),
                 'all_negative_outputs_preserved':True,'no_seed_search':True})
            raise

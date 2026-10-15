"""Two genuine preflights, then four matched bounded adapter trajectories."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
import concurrent.futures,subprocess,sys,math

def run():
    assert read(OUT/'PAIRED_NATIVE_MANIFEST.json')['valid_per_object_native_SS_and_32D_shape']
    assert read(OUT/'CPU_CONTRACT_TESTS.json')['passed']
    files=['trellis2_map_object_train_v6.py','trellis2_map_object_train_owner_v6.py',
        'trellis2_map_object_contract_v6.py','trellis2_map_object_data_v6.py','trellis2_map_object_data_v6_fix1.py']
    freeze={str(JOB/n):sha(JOB/n) for n in files}
    for n in ['PLAN.json','PAIRED_NATIVE_MANIFEST.json']:freeze[str(OUT/n)]=sha(OUT/n)
    save(OUT/'TRAIN_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(fn,GPUS[i*2],'object_'+arm,SEEDS[0],'preflight',
             'trellis2_map_object_train_v6.py',['--arm',arm,'--seed',str(SEEDS[0]),'--preflight'],seconds=360)
             for i,arm in enumerate(ARMS)]
        for j in jobs:j.result()
    probes=[read(OUT/a/f'seed{SEEDS[0]}'/'PREFLIGHT.json') for a in ARMS]
    assert probes[0]['initial_branch_sha256']==probes[1]['initial_branch_sha256']
    slow=max(r['sec_update'] for p in probes for r in p['rows'])
    # Same capped time for every arm; no extensions after outputs.
    seconds=min(1800,max(600,math.ceil(300*slow*1.5+180)))
    save(OUT/'TRAIN_BUDGET_FREEZE.json',{'preflight_sec_update_max':slow,
        'updates':300,'effective_batch':2,'object_exposures_each':600,
        'time_limit_seconds_each':seconds,'seeds':SEEDS,'arms':ARMS,
        'discarded_preflight_updates':2,'reset_same_registered_parent':True,
        'no_automatic_budget_extension':True,'both_SS_and_shape_gradients_verified':True})
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(fn,GPUS[i*2+j],'object_'+a,s,'train300',
            'trellis2_map_object_train_v6.py',['--arm',a,'--seed',str(s)],seconds=seconds)
            for i,a in enumerate(ARMS) for j,s in enumerate(SEEDS)]
        for j in jobs:j.result()
    for s in SEEDS:
        starts=[read(OUT/a/f'seed{s}'/'TRAIN_START.json') for a in ARMS]
        assert starts[0]['initial_branch_sha256']==starts[1]['initial_branch_sha256']
        assert starts[0]['order_sha256']==starts[1]['order_sha256']
        e=[read(OUT/a/f'seed{s}'/'TRAIN_COMPLETE.json')['exposures'] for a in ARMS]
        assert e[0]==e[1]
    save(OUT/'TRAINING_COMPLETE.json',{'arms':ARMS,'seeds':SEEDS,'additional_updates_each':300,
         'total_updates':1200,'same_per_object_inputs_targets_data_noise_budget':True})

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'train_owner.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'TRAIN_OWNER.json',{'pid':p.pid,'finite':True,'GPUs':GPUS,'max_updates':300})
        print('OBJECT_TRAIN_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as e:
            save(OUT/'TRAIN_FAILURE.json',{'type':type(e).__name__,'error':str(e),'all_outputs_preserved':True});raise

"""Bounded two-seed matched real-map assets after optimizer completion."""
from trellis2_map_learning_owner_v1 import OUT, JOB, read, save, sha, GPUS, bounded
import os, sys, subprocess, fcntl, concurrent.futures


def run():
    assert read(OUT/'TRAINING_COMPLETE.json')['updates_per_seed']==100
    inputs=OUT/'real_input'
    assert (inputs/'MAP_INPUT.json').exists()
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_learning_generate_v1.py',
        'trellis2_map_learning_generate_owner_v1.py','trellis2_map_object_module_v1.py',
        'trellis2_map_learning_train_v1.py','trellis2_map_learning_train_fix1.py']}
    freeze.update({str(inputs/n):sha(inputs/n) for n in ['MAP_INPUT.json','map.npz','shuffled_map.npz','removed_last_building_map.npz']})
    for seed in [953101,953102]:
        record=read(OUT/f'seed{seed}/CHECKPOINT_000100.json')
        freeze[record['path']]=record['sha256']
    save(OUT/'GENERATION_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for path,digest in freeze.items():assert sha(path)==digest,path
    fn.__globals__['frozen']=verify
    with (OUT/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(fn,gpu,'trellis2_map',seed,'matched_assets','trellis2_map_learning_generate_v1.py',
                ['--seed',str(seed)],seconds=900) for gpu,seed in zip(GPUS,[953101,953102])]
            for job in jobs:job.result()
    save(OUT/'ASSETS_COMPLETE.json',{'seeds':[953101,953102],'native_assets':8,'model_training_updates':200,
        'real_map_images_not_training':True,'maps':['baseline','correct','shuffled','edit'],
        'visual_acceptance_pending':True})


if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'generation_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'GENERATION_OWNER.json',{'pid':p.pid,'finite':True,'hard_seconds_per_seed':900,
            'new_training':False,'two_registered_training_seeds_only':True})
        print('GENERATION_OWNER',p.pid,flush=True)
    else:
        try:run()
        except Exception as exc:
            save(OUT/'GENERATION_FAILURE.json',{'type':type(exc).__name__,'error':str(exc),'preserve_negatives':True})
            raise

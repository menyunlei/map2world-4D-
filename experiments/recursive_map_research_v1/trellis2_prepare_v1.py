"""Isolated zero-training preparation. Never modifies existing model environments."""
from paths import *
import concurrent.futures
import os
import subprocess
import sys
import time

DEST = JOB / 'trellis2_native_quality_v1'
REPO = DEST / 'vendor/TRELLIS.2'


def command(args, tag, timeout=1800):
    log = DEST / 'setup_logs' / f'{tag}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'PIP_CACHE_DIR': str(DEST/'cache/pip'), 'CONDA_PKGS_DIRS': str(DEST/'cache/conda')}
    with log.open('x') as handle:
        result = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout)
    save(DEST/'setup_audit'/f'{tag}.json', {'command':args, 'returncode':result.returncode,
         'wall_seconds':time.time()-start, 'GPU_seconds':0, 'log':str(log)})
    assert result.returncode == 0, (tag, result.returncode, str(log))


def register():
    assert not DEST.exists(), 'Do not overwrite a previous benchmark'
    reference = DATA/'data/courtyard_new_031_closed'
    plan = {
        'task':'TRELLIS.2 native 3D quality, zero training',
        'official_source':'https://github.com/microsoft/TRELLIS.2',
        'weights':'microsoft/TRELLIS.2-4B',
        'reference_scene':str(reference),
        'reference_hashes':{str(reference/name):sha(reference/name) for name in
                           ('MANIFEST.json','target_mesh.npz','map_tensor.npz')},
        'input_units':['reference_building_1','reference_complete_four_building_block'],
        'seeds':[942201,942202],
        'resolution':1024,
        'pipeline_type':'1024_cascade',
        'sampler':'official pretrained defaults, no parameter/seed selection',
        'A':['mesh_to_O_Voxel_to_mesh','frozen_shape_and_PBR_VAE_reconstruction'],
        'B':'one complete asset per input; no independent-building assembly',
        'input_policy':'fixed reference camera and foreground alpha; preserve entire block; no rembg removal',
        'export_policy':'save raw native outputs; no hole filling, remeshing, pruning or scene repair',
        'interpretation':'image-conditioned quality potential, NOT map-conditioned superiority to G2-B',
        'reference_quality':'authored synthetic four-building scene; not real urban reconstruction',
        'models_unchanged':['DreamX','G2-B','old TRELLIS Q/SLAT/Gaussian decoder'],
        'training_updates':0,'human_review':'PENDING',
        'authorized_physical_GPUs':[0,1,2],
        'per_GPU_job_limit_seconds':1800,
        'total_benchmark_GPU_seconds_limit':14400,
        'automatic_map_adaptation_training':False,
    }
    save(DEST/'PLAN.json',plan)
    save(DEST/'SOURCE_FREEZE.json',{str(JOB/'trellis2_prepare_v1.py'):sha(JOB/'trellis2_prepare_v1.py')})
    print('TRELLIS2_QUALITY_REGISTERED', flush=True)


def prepare():
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        tasks = [pool.submit(command,
            ['git','clone','--depth','1','--recursive','https://github.com/microsoft/TRELLIS.2.git',str(REPO)],
            'official_source',900),
            pool.submit(command,
            ['/hpc2hdd/home/xhe989/miniconda3/bin/conda','create','-y','-p',str(DEST/'environment'),
             '--override-channels','-c','nvidia/label/cuda-12.4.0','-c','conda-forge',
             'python=3.10','cuda-nvcc','cuda-cudart-dev','cuda-cccl','pip','cmake','ninja'],
            'isolated_cuda_environment',1800)]
        for task in tasks:
            task.result()
    revision = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    save(DEST/'OFFICIAL_SOURCE.json',{'commit':revision,
         'submodules':subprocess.check_output(['git','-C',str(REPO),'submodule','status'],text=True),
         'license_sha256':sha(REPO/'LICENSE'), 'no_source_modified':True})
    print('SOURCE_AND_CUDA_ENV_READY', flush=True)


if __name__ == '__main__':
    if '--register' in sys.argv:
        register()
    elif '--detach' in sys.argv:
        with (DEST/'prepare_owner.log').open('x') as out:
            child = subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=out,
               stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(DEST/'PREPARE_OWNER.json',{'pid':child.pid,'finite_not_automation':True,'uses_GPU':False})
        print('TRELLIS2_PREPARE_OWNER',child.pid,flush=True)
    else:
        try:
            prepare()
        except Exception as exc:
            save(DEST/'SETUP_FAILURE.json',{'type':type(exc).__name__,'message':str(exc),
                 'preserve_logs':True,'no_training_launched':True})
            raise

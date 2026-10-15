"""Finite CPU rendering of each finished saved GLB; no recurring task or GPU use."""
from trellis2_asset_trial_v3 import JOB, OUT, save, sha, SEEDS
from paths import ROOT
import concurrent.futures
import os
import subprocess
import sys
import time

BLENDER=ROOT/'experiments/mapworld_dit/real_target_gate_v1/tools/blender-4.5.9-linux-x64/blender'
LIBS=ROOT/'experiments/mapworld_dit/real_target_gate_v1/tools/aptdeps/root/usr/lib/x86_64-linux-gnu'
CASES=[(seed,name) for seed in SEEDS for name in ('map10_block','official_building')]

def render(case):
    seed,name=case;folder=OUT/f'seed{seed}'/name
    start=time.time()
    while not (folder/'ASSET.json').exists():
        assert time.time()-start<2500,'ASSET_NOT_READY_WITHIN_BOUND'
        if (OUT/'INFERENCE_FAILURE.json').exists():raise RuntimeError('GENERATION_FAILED')
        time.sleep(5)
    output=folder/'views'
    assert not output.exists(),'PRESERVE_EXISTING_RENDER'
    command=[str(BLENDER),'-b','--factory-startup','-t','8','--disable-autoexec','--python-exit-code','1',
             '--python',str(JOB/'trellis2_mesh_stills_v3.py'),'--',
             '--asset',str(folder/'world.glb'),'--output',str(output),
             '--bounds','{"min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]}','--_blender-worker']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':str(LIBS),
         'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8'}
    with (OUT/'logs'/f'render_{seed}_{name}.log').open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    save(folder/'RELOAD_RENDER.json',{'command':command,'exit_code':result.returncode,
         'asset_sha256':sha(folder/'world.glb'),'GPU_seconds':0,'fresh_process':True})
    assert result.returncode==0,(case,result.returncode)
    print('SAVED_ASSET_RELOADED_RENDERED',seed,name,flush=True)

def main():
    save(OUT/'RENDER_SOURCE_FREEZE.json',{str(JOB/n):sha(JOB/n) for n in
          ['trellis2_render_owner_v3.py','trellis2_mesh_stills_v3.py','trellis2_saved_mesh_render_v2.py']})
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(render,c) for c in CASES]
        for j in jobs:j.result()
    save(OUT/'RENDER_COMPLETE.json',{'cases':CASES,'fresh_asset_reload':True,'GPU_seconds':0})

if __name__=='__main__':
    if '--detach' in sys.argv:
        (OUT/'logs').mkdir(exist_ok=True)
        with (OUT/'render_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'RENDER_OWNER.json',{'pid':p.pid,'finite':True,'GPU_seconds':0})
        print('RENDER_OWNER',p.pid,flush=True)
    else:
        try:main()
        except Exception as e:
            save(OUT/'RENDER_FAILURE.json',{'type':type(e).__name__,'message':str(e)})
            raise

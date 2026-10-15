"""Bounded CPU Blender verification; no generation or GPU work."""
from trellis2_render_owner_v3 import OUT, JOB, BLENDER, LIBS, SEEDS, save, sha
import concurrent.futures
import subprocess
import os
import sys

def render(seed):
    root=OUT/f'seed{seed}/official_building';asset=root/'building_basecolor_preview.glb'
    command=[str(BLENDER),'-b','--factory-startup','-t','8','--disable-autoexec','--python-exit-code','1',
       '--python',str(JOB/'trellis2_compat_reload_v1.py'),'--','--asset',str(asset),
       '--output',str(root/'compatibility_views'),'--bounds','{"min":[-.5,-.5,-.5],"max":[.5,.5,.5]}','--_blender-worker']
    # Strict JSON does not accept leading decimal shorthand.
    command[command.index('--bounds')+1]='{"min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]}'
    with (OUT/'logs'/f'compat_cpu_{seed}.log').open('x') as log:
        result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':str(LIBS)},
                              stdout=log,stderr=subprocess.STDOUT,timeout=600)
    save(root/'BLENDER_COMPATIBILITY.json',{'command':command,'exit_code':result.returncode,
        'asset_sha256':sha(asset),'GPU_seconds':0,'full_geometry':True,'native_full_PBR_fields_not_in_preview':True})
    assert result.returncode==0,(seed,result.returncode)
    print('BLENDER_COMPATIBLE_ASSET_RELOADED',seed,flush=True)

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'compat_cpu_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'COMPAT_CPU_OWNER.json',{'pid':p.pid,'finite':True,'GPU_seconds':0})
        print('COMPAT_CPU_OWNER',p.pid,flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for r in pool.map(render,SEEDS):pass
        save(OUT/'COMPAT_CPU_COMPLETE.json',{'seeds':SEEDS,'GPU_seconds':0})

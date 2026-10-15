"""Same saved assets/cameras, corrected Blender module path; preserve initial logs."""
from trellis2_render_owner_v3 import *
import inspect

scope=dict(main.__globals__)
source=inspect.getsource(render).replace("'views'","'views_v4'")
source=source.replace('trellis2_mesh_stills_v3.py','trellis2_mesh_stills_v4.py')
source=source.replace("f'render_{seed}_{name}.log'","f'render_fix1_{seed}_{name}.log'")
source=source.replace("'RELOAD_RENDER.json'","'RELOAD_RENDER_FIX1.json'")
exec(compile(source,__file__,'exec'),scope)
source=inspect.getsource(main).replace('RENDER_SOURCE_FREEZE.json','RENDER_SOURCE_FIX1.json').replace('RENDER_COMPLETE.json','RENDER_COMPLETE_FIX1.json')
source=source.replace('trellis2_mesh_stills_v3.py','trellis2_mesh_stills_v4.py').replace('trellis2_render_owner_v3.py','trellis2_render_owner_fix1.py')
exec(compile(source,__file__,'exec'),scope)

if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'render_fix1_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'RENDER_FIX1_OWNER.json',{'pid':p.pid,'finite':True,'GPU_seconds':0,'sole_fix':'Blender import path'})
        print('RENDER_FIX1_OWNER',p.pid,flush=True)
    else:
        try:scope['main']()
        except Exception as e:
            save(OUT/'RENDER_FIX1_FAILURE.json',{'type':type(e).__name__,'message':str(e)})
            raise

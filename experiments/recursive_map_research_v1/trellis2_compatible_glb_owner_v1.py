"""Finite saved-native-asset format conversion, on free authorized GPU2."""
from trellis2_trial_owner_v3 import OUT, JOB, bounded, save, sha
import subprocess
import sys
if '--detach' in sys.argv:
    with (OUT/'compatible_glb_owner.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'COMPATIBLE_GLB_OWNER.json',{'pid':p.pid,'finite':True,'gpu':2,'seconds_cap':300})
    print('COMPATIBLE_GLB_OWNER',p.pid,flush=True)
else:
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_compatible_glb_owner_v1.py','trellis2_building_compatible_glb_v1.py']}
    save(OUT/'COMPATIBLE_GLB_SOURCE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h
    fn.__globals__['frozen']=verify
    fn({'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},'trellis2',942201,
       'compatible_glb','trellis2_building_compatible_glb_v1.py',[],seconds=300)

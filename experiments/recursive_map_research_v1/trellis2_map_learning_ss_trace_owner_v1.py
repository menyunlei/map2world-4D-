"""One bounded zero-training support attribution; no seed/model search."""
from trellis2_map_learning_owner_v1 import OUT, JOB, read, save, sha, bounded
import subprocess, sys

if '--detach' in sys.argv:
    with (OUT/'ss_trace_owner.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'SS_TRACE_OWNER.json',{'pid':p.pid,'finite':True,'GPU':2,'model_updates':0,'hard_seconds':180})
    print('SS_TRACE_OWNER',p.pid,flush=True)
else:
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_learning_ss_trace_v1.py','trellis2_map_learning_ss_trace_owner_v1.py']}
    save(OUT/'SS_TRACE_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    fn({'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},'trellis2_map',953101,
       'SS_attribution','trellis2_map_learning_ss_trace_v1.py',[],seconds=180)

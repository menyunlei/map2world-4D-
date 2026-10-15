"""Bounded saved-asset chunked inspection; same two completed generation seeds."""
from trellis2_trial_owner_v3 import OUT, JOB, bounded, read, save, sha
import subprocess
import sys

if '--detach' in sys.argv:
    with (OUT/'raw_chunk_owner.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'RAW_CHUNK_OWNER.json',{'pid':p.pid,'finite':True,'physical_gpu':2,'hard_seconds':600})
    print('RAW_CHUNK_OWNER',p.pid,flush=True)
else:
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_raw_chunk_owner_fix2.py','trellis2_raw_views_chunk_fix2.py','trellis2_raw_asset_views_v3.py']}
    save(OUT/'RAW_CHUNK_SOURCE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h
    fn.__globals__['frozen']=verify
    gpu={'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'}
    fn(gpu,'trellis2',942201,'raw_chunked','trellis2_raw_views_chunk_fix2.py',[],seconds=600)

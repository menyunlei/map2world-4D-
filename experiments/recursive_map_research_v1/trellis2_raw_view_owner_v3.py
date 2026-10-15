"""One bounded lease on authorized free GPU2 for saved-asset inspection."""
from trellis2_trial_owner_v3 import OUT, JOB, bounded, read, save, sha
import subprocess
import sys

if '--detach' in sys.argv:
    with (OUT/'raw_view_owner.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'RAW_VIEW_OWNER.json',{'pid':p.pid,'finite':True,'physical_gpu':2,'hard_seconds':240})
    print('RAW_VIEW_OWNER',p.pid,flush=True)
else:
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_raw_view_owner_v3.py','trellis2_raw_asset_views_v3.py']}
    save(OUT/'RAW_VIEW_SOURCE_FREEZE.json',freeze)
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h
    fn.__globals__['frozen']=verify
    gpu={'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'}
    fn(gpu,'trellis2',942201,'raw_views','trellis2_raw_asset_views_v3.py',[],seconds=240)

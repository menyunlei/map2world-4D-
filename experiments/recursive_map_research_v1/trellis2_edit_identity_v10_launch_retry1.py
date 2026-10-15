"""Retry only a prelaunch GPU ownership conflict; no model/output changes."""
from trellis2_edit_identity_v10_owner import run,OUT
from paths import save,sha,read
import os,subprocess,sys
if __name__=='__main__':
    if '--detach' in sys.argv:
        assert not list(OUT.glob('seed*')), 'A generation already started'
        prior=read(OUT/'FAILURE.json')
        assert 'GPU_NOT_FREE' in prior['error']
        with (OUT/'owner_retry1.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'OWNER_RETRY1.json',{'pid':child.pid,'reason':'prelaunch GPU occupancy conflict cleared; same frozen inference','script_sha256':sha(__file__)})
        print('OWNER_RETRY1',child.pid,flush=True)
    else:
        try:run()
        except Exception as e:
            save(OUT/'FAILURE_RETRY1.json',{'type':type(e).__name__,'error':str(e),'preserve_all':True})
            raise

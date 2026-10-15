"""Independent read-only native render owner; uses one free authorized A800."""
from trellis2_map_learning_owner_v1 import OUT, JOB, save, sha, bounded
import subprocess, sys


if __name__=='__main__':
    if '--detach' in sys.argv:
        with (OUT/'view_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'VIEW_OWNER.json',{'pid':p.pid,'finite':True,'physical_GPU':2,'training':False})
        print('VIEW_OWNER',p.pid,flush=True)
    else:
        freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_learning_views_v1.py','trellis2_map_learning_views_owner_v1.py']}
        save(OUT/'VIEW_SOURCE_FREEZE.json',freeze)
        fn=bounded();fn.__globals__['NEW']=OUT
        def verify():
            for p,h in freeze.items():assert sha(p)==h,p
        fn.__globals__['frozen']=verify
        gpu={'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'}
        for seed in [953101,953102]:
            fn(gpu,'trellis2_map',seed,'reloaded_views','trellis2_map_learning_views_v1.py',['--seed',str(seed)],seconds=600)
        save(OUT/'NATIVE_RELOADS_COMPLETE.json',{'seeds':[953101,953102],'native_asset_count':8,
            'view_count':72,'no_model_resampling':True,'geometry_unmodified':True})

"""Finish only the unstarted feedback sample; never replay candidate0.

All earlier costs and errors remain, and reduce the original720s ceiling.
"""
from pathlib import Path
original=Path(__file__).with_name('v27_worker.py')
source=original.read_text().split("\nif __name__=='__main__':")[0]
source=source.replace('for i in range(2):','for i in range(1,2):')
source=source.replace("str(JOB/'v27_verify.py')","str(JOB/'v27_verify_fix1.py')")
for name in ['ONE_OWNER','OWNER_LAUNCH','GPU_COST','MODEL_READY','REQUEST_SEEN']:
    source=source.replace("'"+name+".json'","'"+name+"_FINAL.json'")
source=source.replace("'worker.log'","'worker_FINAL.log'")
exec(compile(source,str(original),'exec'),globals())
original_read=read
def read(p):
    value=original_read(p)
    if Path(p)==OUT/'CONTRACT.json':
        value=dict(value)
        for name in ['GPU_COST.json','GPU_COST_FIX1.json','GPU_COST_FIX2.json']:
            value['lease_seconds']-=original_read(OUT/name)['gpu_seconds']
    return value
if __name__=='__main__':
    if '--work' in sys.argv:work()
    elif '--detach' in sys.argv:
        # Entire first inference and its independent verdict are already saved.
        assert (OUT/'attempts/attempt-0/REPORT.json').exists()
        assert not (OUT/'attempts/attempt-1/ONE_ATTEMPT.json').exists()
        for name in ['ONE_OWNER_FINAL.json','MODEL_READY_FINAL.json','REQUEST_SEEN_FINAL.json','GPU_COST_FINAL.json']:
            assert not (OUT/name).exists(),name
        with (OUT/'owner_FINAL.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid,remaining_seconds=read(OUT/'CONTRACT.json')['lease_seconds'])),flush=True)
    else:owner()

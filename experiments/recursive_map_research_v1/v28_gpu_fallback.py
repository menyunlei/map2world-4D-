"""Same frozen candidate, use an actually idle authorized GPU, never share others."""
from pathlib import Path
p=Path(__file__).with_name('v28_native64.py')
s=p.read_text().split("\nif __name__=='__main__':")[0]
s=s.replace("idx=2 if which=='generate' else 3;uuid=GPUS[idx][1]", """owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    free=[(i,u) for i,u in GPUS if u not in owners]
    assert free,'No idle authorized GPU; do not use another live owner'
    idx,uuid=free[0]""")
exec(compile(s,str(p),'exec'),globals())
if __name__=='__main__':
    if '--work-generate' in sys.argv:generate()
    elif '--work-verify' in sys.argv:verify()
    elif '--verify' in sys.argv:owner('verify')
    elif '--detach' in sys.argv:
        assert not (OUT/'ONE_GENERATE.json').exists(),'No repeated inference'
        with (OUT/'generation_owner_fallback.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner('generate')

"""Resume after verifier hash-API failure, without regenerating attempt0.

Preserve all original worker/cost/error evidence. Sum both physical lease windows.
"""
from pathlib import Path
original=Path(__file__).with_name('v27_worker.py')
source=original.read_text().split("\nif __name__=='__main__':")[0]
source=source.replace('for i in range(2):','for i in range(1,2):')
source=source.replace("str(JOB/'v27_verify.py')","str(JOB/'v27_verify_fix1.py')")
for name in ['ONE_OWNER','OWNER_LAUNCH','GPU_COST']:
    source=source.replace("'"+name+".json'","'"+name+"_FIX1.json'")
source=source.replace("'worker.log'","'worker_FIX1.log'")
exec(compile(source,str(original),'exec'),globals())
original_work=work
def work():
    verify_sources()
    status('verifying_saved_attempt0',new_inference=False)
    d=OUT/'attempts/attempt-0'
    assert sha(d/'world.glb')==read(d/'COMPLETE.json')['sha256']
    assert not (d/'REPORT.json').exists(),'No repeat evaluation after a completed verdict'
    if (d/'TOP_NATIVE.png').exists():(d/'TOP_NATIVE.png').rename(d/'TOP_NATIVE_BEFORE_HASH_FIX.png')
    with (d/'verify_FIX1.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B','-u',str(JOB/'v27_verify_fix1.py'),str(d)],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,timeout=90)
    assert rc.returncode==0,'Verifier compatibility replay failed; stop, do not regenerate'
    r=read(d/'REPORT.json')
    save(d/'QUARANTINED.json',dict(epoch=time.time(),reasons=r['decision']['reasons'],world_unchanged=True))
    # Patch metadata changes no measurement/threshold, prompt, seed or weights.
    save(OUT/'COMPAT_REPAIR.json',dict(original_verifier_sha256=sha(JOB/'v27_verify.py'),
        compatibility_wrapper_sha256=sha(JOB/'v27_verify_fix1.py'),
        resume_sha256=sha(__file__),change='hashlib.file_digest -> identical streaming SHA256 for Python3.10',
        reused_attempt0_sha256=r['asset_sha256'],extra_native_proposals=0,
        thresholds_unchanged=True,original_failure_evidence_retained=True))
    if not any(x in r['decision']['reasons'] for x in ['KNOWN_ROAD_OBSTRUCTED','KNOWN_ROAD_SURFACE_NOT_ESTABLISHED']):
        status('stopped',reason='no_justified_road_retry',unresolved=r['decision']['reasons']);return
    original_work()
# Override contract read ONLY to enforce the unused portion of its original GPU budget.
original_read=read
def read(p):
    value=original_read(p)
    if Path(p)==OUT/'CONTRACT.json':
        value=dict(value);value['lease_seconds']-=original_read(OUT/'GPU_COST.json')['gpu_seconds']
    return value
if __name__=='__main__':
    if '--work' in sys.argv:work()
    elif '--detach' in sys.argv:
        with (OUT/'owner_FIX1.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner()

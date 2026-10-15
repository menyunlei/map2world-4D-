"""Use the audited subcell UUID runner directly; preserve failed owner receipt.

The older composition wrapper resolves to a different historic runner without
the expected cap literal. That launch stopped before any GPU worker started.
This changes orchestration only, not the frozen experiment/model contract.
"""
from paths import JOB,read,save,sha
import trellis2_structure_learnability_v8_owner as owner
import subcell_owner_v1 as lease
import types,threading,subprocess,sys,fcntl

original_frozen=owner.frozen
SELF_HASH=sha(__file__)
def verify():
    original_frozen();assert sha(__file__)==SELF_HASH

def bounded():
    scope={**lease.run.__globals__,'NEW':owner.OUT,'frozen':verify,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (owner.OUT/'gpu_cost').glob('*.json')),
        'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    return types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)

owner.bounded=bounded;owner.frozen=verify

if __name__=='__main__':
    phase='large' if '--large' in sys.argv else 'small'
    if '--detach' in sys.argv:
        assert read(owner.OUT/'CPU_TESTS.json')['returncode']==0
        # Detect a still-running prior owner without ever stopping it.
        previous=read(owner.OUT/'SMALL_OWNER.json')['pid']
        assert not __import__('pathlib').Path(f'/proc/{previous}').exists(),'PRIOR_OWNER_STILL_ACTIVE'
        with (owner.OUT/f'{phase}_fix1_owner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__]+(['--large'] if phase=='large' else []),
                cwd=JOB,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(owner.OUT/f'{phase.upper()}_FIX1_OWNER.json',{'pid':p.pid,'repair_sha256':SELF_HASH,
            'repair':'existing subcell leased runner, unchanged job/model budget','finite_not_automation':True})
        print('LEARNABILITY_FIX1_OWNER',p.pid,flush=True)
    else:
        try:owner.run(phase)
        except Exception as e:
            save(owner.OUT/f'{phase.upper()}_FIX1_FAILURE.json',{'type':type(e).__name__,'error':str(e),'preserve_partials':True});raise

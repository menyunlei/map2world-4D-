"""Record only an actually executed, successful CPU test process."""
from paths import JOB,save,sha
from trellis2_object_confidence_prepare_v7 import OUT
import subprocess,sys,os
if __name__=='__main__':
    script=JOB/'trellis2_object_confidence_fix_v7.py'
    result=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=60)
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    successful=result.returncode==0 and 'Ran 4 tests' in result.stderr and '\nOK\n' in result.stderr
    assert successful,('CPU_TEST_PROCESS_DID_NOT_PASS',result.returncode,result.stderr)
    save(OUT/'CONFIDENCE_CPU_TESTS.json',{'passed':successful,'tests':4,
        'actual_command':[sys.executable,'-B',str(script)],'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'test_source_sha256':sha(script),
        'no_GPU_launch_in_this_verifier':True})
    print('VERIFIED_FROM_ACTUAL_CPU_TEST_PROCESS',result.returncode,flush=True)

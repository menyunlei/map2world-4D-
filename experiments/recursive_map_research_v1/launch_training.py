"""Finite two-seed owner, verifier hash check and physical occupancy accounting."""
from paths import *
import argparse,concurrent.futures,fcntl,os,signal,subprocess,sys,time

GPUS=[{'index':0,'uuid':'GPU-e0a56bf9-85ef-c658-67fa-457b76585254'},
      {'index':1,'uuid':'GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa'}]

def verify():
    sealed=read(JOB/'VERIFIER_FREEZE.json')
    for path,expected in sealed['files'].items():
        if sha(path)!=expected:raise RuntimeError('VERIFIER_OR_DATA_TAMPER: '+path)
    return sha(JOB/'VERIFIER_FREEZE.json')

def policy(out,candidate,tag):
    read_paths=['/usr','/lib','/lib64','/etc','/proc','/sys',sys.prefix,sys.base_prefix,
        str(NATIVE/'environment'),str(NATIVE/'TRELLIS'),str(NATIVE/'sources'),str(NATIVE/'weights'),
        str(D2/'code'),str(HIER/'code'),str(C4/'code'),str(ROOT/'models'),str(ROOT/'wan'),
        str(ROOT/'utils'),str(ROOT/'pipeline'),str(ROOT/'camera'),str(ROOT/'dist'),
        str(ROOT/'experiments/dreamx_injection_readout_screen_v1/code'),
        str(ROOT/'experiments/dreamx_injection_readout_topofix_v1/code'),
        str(ROOT/'experiments/dreamx_courtyard_preservation_v1/code'),
        str(JOB/'candidates'/candidate)]
    read_paths += [str(p) for p in ROOT.glob('*.py')]
    read_paths += read(JOB/'TRAIN_READ_FILES.json')
    read_paths += [str(JOB/f) for f in ['paths.py','runtime.py','detail_model.py','train_candidate.py','native_parent_identity.py','TRAIN_DATA.json','TRAINING_SLOTS.json','TRAIN_CONTRACT.json']]
    read_paths += [str(C4/f) for f in ['SUPERVISION_LABEL_AUDIT.json','AUXILIARY_GROUP_COUNTS.json','AUXILIARY_SCHEDULE.json','AUXILIARY_CONTRACT.json']]
    read_paths += [v['path'] for v in read(C4/'SUPERVISION_LABEL_AUDIT.json')['records'].values()]
    for stage in ['structure','detail']:
        meta=D2/f'D2_local_map/{stage}/CHECKPOINT_005000.json'
        read_paths += [str(meta),read(meta)['path']]
    meta=NATIVE/'prior_training_v1/C3/structure/CHECKPOINT_005000.json'
    read_paths += [str(meta),read(meta)['path']]
    read_paths += [str(p) for p in (NATIVE/'audit').glob('*.json') if 'TEST' not in p.name.upper()]
    cf=read(NATIVE/'audit/CODEC_FREEZE_V1.json');read_paths.append(cf['checkpoint_path'])
    # Frozen Structure codec is loaded by structure_decoder from TRELLIS weights.
    read_paths += [str(NATIVE/'checkpoints')]
    out.mkdir(parents=True,exist_ok=True)
    for folder in ['tmp','cache','extensions']:(out/folder).mkdir(exist_ok=True)
    value={'read_paths':read_paths,'list_paths':[str(JOB),str(ROOT)],'write_root':str(out),'probe_output':f'ISOLATION_{tag}.json',
       'deny_probes':{'verifier_code':str(JOB/'verifier_core.py'),
          'private_reference':str(JOB/'verifier_private/courtyard_new_001_closed.npz'),
          'unfiltered_dataset':str(NATIVE/'data/paired_records_v3/DATASET_MANIFEST.json')}}
    path=JOB/'policies'/f'{candidate}_{out.name}_{tag}.json';save(path,value);return path

def run(candidate,seed,gpu,until,resume,version,probe_only=False):
    tag=f'{resume or 0:03d}_{until:03d}_{version}';name=f'{candidate}_s{seed}_{tag}'
    lockpath=JOB/'gpu_leases'/f'gpu{gpu["index"]}.lock';lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        assert f'{gpu["index"]}, {gpu["uuid"]}' in inv and gpu['uuid'] not in owners,('OCCUPIED',gpu,owners)
        verify();contract=read(JOB/'CONTRACT.json')
        previous=[read(p) for p in (JOB/'gpu_cost').glob('*.json')]
        used=sum(x['gpu_seconds'] for x in previous)
        assert used+2*contract['operational_max_job_seconds']<=contract['operational_safety_ceiling_gpu_seconds'],'INTERNAL_GPU_CAP'
        out=JOB/'runs'/candidate/f'seed{seed}';pol=policy(out,candidate,tag)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'CUDA_HOME':'/usr/local/cuda-11.8',
            'SPCONV_ALGO':'native','ATTN_BACKEND':'flash_attn','VIDEOX_ATTENTION_TYPE':'FLASH_ATTENTION',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(out/'tmp'),
            'XDG_CACHE_HOME':str(out/'cache'),'CUDA_CACHE_PATH':str(out/'cache'),
            'TORCH_EXTENSIONS_DIR':str(out/'extensions')}
        args=[sys.executable,'-B','-u',str(JOB/'sandbox.py'),str(pol),str(JOB/'train_candidate.py'),
              '--candidate',candidate,'--seed',str(seed),'--until',str(until),'--attempt',version]
        if resume:args+=['--resume',str(resume)]
        if probe_only:args+=['--probe-only']
        log=JOB/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True);start=time.time()
        with log.open('x') as handle:
            child=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(JOB/'audit'/f'{name}_LAUNCH.json',{'epoch':start,'pid':child.pid,'GPU_UUID':gpu['uuid'],'command':args,'verifier_sha256':verify()})
            try:code=child.wait(timeout=contract['operational_max_job_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        stop=time.time()
        save(JOB/'gpu_cost'/f'{name}.json',{'name':name,'gpu':gpu,'start_epoch':start,'end_epoch':stop,'gpu_seconds':stop-start,
            'includes_loading_idle_initialization':True,'exclusive_physical_GPU':True,'returncode':code})
        save(JOB/'audit'/f'{name}_EXIT.json',{'epoch':stop,'returncode':code,'next_stage_automatic':False})
        verify();return {'seed':seed,'returncode':code}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['g1_a','g1_b','g2_a'],required=True)
    ap.add_argument('--until',type=int,choices=[20,100,300],required=True);ap.add_argument('--resume',type=int,choices=[20,100])
    ap.add_argument('--version',default='v1');ap.add_argument('--probe-only',action='store_true');a=ap.parse_args()
    verify()
    if a.candidate=='g2_a':
        spec=read(JOB/'archive/g2_a_PREREGISTRATION.json')
        registration=read(JOB/'archive/g2_a_REGISTERED.json')
        assert spec['generation']==2 and spec['parent_candidate']=='g1_a'
        assert read(JOB/'archive/g1_a_100_DECISION.json')['decision']=='STOP_AT100'
        assert registration['branch']=='g2_a' and spec['same_evaluator_data_loss_LR_seeds']
        for path,expected in registration['files'].items():
            if '/candidates/g2_a/' in path:assert sha(path)==expected,'unregistered G2 mechanism change'
    trainseal=read(JOB/'TRAIN_ASSET_FREEZE.json')
    # Check complete TRAIN/source assets once before either GPU is occupied.
    if not a.probe_only:
        for path,expected in trainseal['files'].items():
            assert sha(path)==expected,('TRAIN_ASSET_CHANGED',path)
    if a.until>20:
        gate=read(JOB/'admission'/f'{a.candidate}_{a.until}.json')
        assert gate['admitted'] and gate['two_training_seeds'] and gate['verifier_sha256']==sha(JOB/'VERIFIER_FREEZE.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,a.candidate,seed,gpu,a.until,a.resume,a.version,a.probe_only) for seed,gpu in zip([1701,1702],GPUS)]
        results=[]
        for f in concurrent.futures.as_completed(fs):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
        assert all(r['returncode']==0 for r in results),'training worker failed; no automatic retry'

if __name__=='__main__':main()

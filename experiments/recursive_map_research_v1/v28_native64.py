"""One native1024 Shape entry test, keeping v27's exact decoded Structure64.

No training, new foundation model, mesh editing, frame fitting, or retry loop.
"""
from paths import JOB,read,save,sha
from pathlib import Path
import os,sys,time,json,shutil,subprocess,fcntl,signal
import numpy as np
OUT=JOB/'map_object_world_v1/trellis2_road_stage_v28'
OLD=JOB/'map_object_world_v1/trellis2_autonomous_v27'
D=OUT/'native64'
def prepare():
    assert (OUT/'STAGE_DIAGNOSIS.json').exists()
    assert not D.exists(),'Preserve the single fixed test'
    D.mkdir()
    p=read(OLD/'attempts/attempt-1/PLAN.json')
    shutil.copyfile(OLD/'attempts/attempt-1/input.png',D/'input.png')
    p.update(input=str(D/'input.png'),input_sha256=sha(D/'input.png'),attempt=2,
        pipeline_type='1024',intervention='Retain actual Structure64 support via official1024 Shape path; no32 pooling or LR512 cascade',
        matched_control='Saved v27 attempt1: exact same Structure latent, image, seed942271, frame, native weights, CFG and12steps',
        frozen_Structure_sha256=sha(OLD/'attempts/attempt-1/STRUCTURE_LATENT.npz'),
        no_extra_Structure_generation=True,geometry_postprocessing='None; retain native triangles; skip historical optional fill_holes only',
        random_noise_caveat='Same fixed global seed and native Structure noise consumption. Shape support dimensions/path differ; per-token noise is not claimed identical.',
        map_control_claim=False)
    save(D/'PLAN.json',p)
    save(OUT/'PREGISTRATION.json',dict(candidate_count=1,seed=942271,training_updates=0,
        baseline='Existing v27 attempt1, no repeated equivalent full generation',
        selected_change='Use official native1024 Structure64-to-Shape entry, not1024_cascade Structure32 entry',
        hypothesis='Max pooling64->32 discards vertical subcell support; native64 entry may reduce elevated road surfaces without replacing generated geometry.',
        immutable_inputs={'old_latent':sha(OLD/'attempts/attempt-1/STRUCTURE_LATENT.npz'),
            'old_mesh':sha(OLD/'attempts/attempt-1/world.glb'),'queries':sha(OLD/'EVALUATION_QUERIES.npz'),
            'contract':sha(OLD/'CONTRACT.json'),'image':sha(D/'input.png')},
        success_not_guaranteed=True,forbidden=['coordinate/frame shift','threshold change','mesh repair','training','further retries'],
        single_generation_gpu_seconds_max=480,verification_gpu_seconds_max=90,
        interpretation='Native entry-path intervention bundles support resolution and omission of LR cascade. Not a pure isolated pooling-only or noise-identical causal estimate. No generalization claim.'))
    save(OUT/'CANDIDATE_SOURCE_FREEZE.json',{str(x):sha(x) for x in [Path(__file__),D/'PLAN.json',OUT/'PREGISTRATION.json',JOB/'trellis2_native_street_v17.py',JOB/'v27_verify.py',JOB/'v27_verify_fix1.py']})
    print('V28_NATIVE64_REGISTERED',flush=True)
def generate():
    for p,h in read(OUT/'CANDIDATE_SOURCE_FREEZE.json').items():assert sha(p)==h
    import trellis2_native_street_v17 as native
    os.environ['CUDA_HOME']=str(native.DEST/'environment');os.environ['PATH']=str(native.DEST/'environment/bin')+':'+os.environ['PATH']
    sys.path.insert(0,str(native.REPO))
    import torch,types
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from v26_resident import fixed_export
    p=read(D/'PLAN.json');original=Trellis2ImageTo3DPipeline.from_pretrained
    dense=np.load(OUT/'STRUCTURE64_REPLAY.npz')['occupied']
    expected=np.c_[np.zeros(int(dense.sum()),dtype=np.int32),np.argwhere(dense).astype(np.int32)]
    @classmethod
    def load(cls,*a,**kw):
        pipe=original(*a,**kw)
        def frozen_structure(self,cond,resolution,num_samples=1,sampler_params={}):
            assert resolution==64 and num_samples==1,'Must use actual native64 interface'
            model=self.models['sparse_structure_flow_model']
            # Consume the exact native initial Structure noise without resampling
            # Structure; fixed saved support is the matched variable.
            noise=torch.randn(num_samples,model.in_channels,model.resolution,model.resolution,model.resolution).to(self.device)
            coords=torch.tensor(expected,device=self.device,dtype=torch.int32)
            np.savez_compressed(D/'GENERATED_STRUCTURE.npz',coords=expected)
            save(D/'STRUCTURE_REUSE.json',dict(resolution=resolution,coords=len(expected),
                 same_saved_decoded64=True,noise_sha256=__import__('hashlib').sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                 source_latent_sha256=p['frozen_Structure_sha256'],decoder_or_output_masking=False))
            return coords
        pipe.sample_sparse_structure=types.MethodType(frozen_structure,pipe)
        original_run=pipe.run
        def run(*a,**kw):kw['pipeline_type']='1024';return original_run(*a,**kw)
        pipe.run=run
        return pipe
    Trellis2ImageTo3DPipeline.from_pretrained=load
    native.OUT=D;native.SEED=p['seed'];began=time.time();native.generate()
    fixed_export(D/'native_street_basecolor.glb',D/'world.glb',p['boundary']['fixed_world_frame']['glTF_root_matrix'])
    save(D/'COMPLETE.json',dict(id='native64',seed=p['seed'],seconds=time.time()-began,bytes=(D/'world.glb').stat().st_size,
        sha256=sha(D/'world.glb'),frame_changed=False,geometry_repaired=False))
    assert sha(OLD/'attempts/attempt-1/world.glb')==read(OUT/'PREGISTRATION.json')['immutable_inputs']['old_mesh']
    print('V28_NATIVE64_ASSET_READY',flush=True)
def verify():
    import v27_verify_fix1 as verifier
    verifier.OUT=OLD;verifier.evaluate(D)
def owner(which):
    from v26_resident import GPUS
    idx=2 if which=='generate' else 3;uuid=GPUS[idx][1]
    assert f'{idx}, {uuid}' in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lease=(JOB/'gpu_leases'/f'gpu{idx}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    save(OUT/f'ONE_{which.upper()}.json',dict(epoch=time.time(),uuid=uuid,source_sha256=sha(__file__)))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'ATTN_BACKEND':'flash_attn','SPCONV_ALGO':'native','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
        'CUDA_HOME':'/usr/local/cuda-11.8'}
    began=time.time();limit=480 if which=='generate' else 90
    try:
        with (OUT/f'{which}.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--work-'+which],cwd=JOB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            save(OUT/f'LAUNCH_{which.upper()}.json',dict(pid=child.pid,epoch=began,uuid=uuid,limit_seconds=limit))
            try:code=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        save(OUT/f'GPU_COST_{which.upper()}.json',dict(index=idx,uuid=uuid,gpu_seconds=time.time()-began,returncode=code))
        print(json.dumps(dict(task=which,returncode=code)),flush=True)
    finally:lease.close()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--work-generate' in sys.argv:generate()
    elif '--work-verify' in sys.argv:verify()
    elif '--verify' in sys.argv:owner('verify')
    elif '--detach' in sys.argv:
        with (OUT/'generation_owner.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner('generate')

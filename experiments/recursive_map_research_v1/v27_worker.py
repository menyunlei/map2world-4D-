"""One request, one proposal, at most one feedback retry. Bounded resident prior."""
from pathlib import Path
from paths import JOB,read,save,sha
import os,sys,time,json,subprocess,signal,fcntl,types,copy
OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
from v26_resident import GPUS,fixed_export
def atomic(p,v):
    p=Path(p);tmp=p.with_suffix('.next');tmp.write_text(json.dumps(v));os.replace(tmp,p)
def status(stage,**kw):atomic(OUT/'WORKER_STATUS.json',dict(stage=stage,epoch=time.time(),**kw))
def verify_sources():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(JOB/p)==h,('Frozen source changed',p)
def work():
    verify_sources();contract=read(OUT/'CONTRACT.json');status('loading_python',generation_started=False)
    import trellis2_native_street_v17 as native
    os.environ['CUDA_HOME']=str(native.DEST/'environment');os.environ['PATH']=str(native.DEST/'environment/bin')+':'+os.environ['PATH']
    sys.path.insert(0,str(native.REPO))
    import torch,numpy as np
    from trellis2.pipelines import Trellis2ImageTo3DPipeline,rembg
    rembg.UnusedBackgroundRemoval=lambda:None
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    start=time.time();p=read(OUT/'attempts/attempt-0/PLAN.json');status('loading_model',generation_started=False)
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(p['runtime']);pipe.cuda()
    methods=['get_cond','sample_sparse_structure','sample_shape_slat_cascade','sample_tex_slat','decode_shape_slat','decode_latent']
    canonical={name:getattr(type(pipe),name) for name in methods};sampler=copy.deepcopy(pipe.sparse_structure_sampler)
    @classmethod
    def resident_get(cls,path,*a,**kw):
        assert str(path)==p['runtime'];return pipe
    Trellis2ImageTo3DPipeline.from_pretrained=resident_get
    save(OUT/'MODEL_READY.json',dict(epoch=time.time(),load_seconds=time.time()-start,pipe_id=id(pipe),
        low_vram=pipe.low_vram,resident_CPU_weights=True,stagewise_GPU_offload=True,
        parameter_dtypes={k:sorted({str(x.dtype) for x in m.parameters()}) for k,m in pipe.models.items()},
        pipeline_sha256=sha(Path(p['runtime'])/'pipeline.json'),model_weight_updates=0))
    status('resident_ready',generation_started=False)
    wait=time.time()
    while not (OUT/'REQUEST.json').exists():
        if time.time()-wait>contract['idle_seconds']:status('stopped',reason='idle_timeout_without_request');return
        time.sleep(.1)
    request=read(OUT/'REQUEST.json');save(OUT/'REQUEST_SEEN.json',request)
    for i in range(2):
        d=OUT/'attempts'/f'attempt-{i}';p=read(d/'PLAN.json');verify_sources()
        if i:
            prev=read(OUT/'attempts/attempt-0/REPORT.json')
            reasons=prev['decision']['reasons']
            if not any(x in reasons for x in ('KNOWN_ROAD_OBSTRUCTED','KNOWN_ROAD_SURFACE_NOT_ESTABLISHED')):
                status('stopped',reason='no_justified_road_retry',unresolved=reasons);return
            save(d/'FEEDBACK.json',prev['targeted_feedback'])
        save(d/'ONE_ATTEMPT.json',dict(epoch=time.time(),seed=p['seed'],request=request,automatic_feedback_retry=bool(i)))
        began=time.time();status('generating',attempt=i,started_epoch=began,pipe_id=id(pipe))
        def stage(label,**kw):
            torch.cuda.synchronize();v=dict(stage=label,epoch=time.time(),attempt=i,**kw)
            atomic(d/'STATUS.json',v)
            with (d/'STAGES.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
        for name,fn in canonical.items():setattr(pipe,name,types.MethodType(fn,pipe))
        pipe.sparse_structure_sampler=copy.deepcopy(sampler)
        if i:
            from v24_boundary_sampler import install
            try:install(pipe,d)
            except AssertionError as e:
                save(d/'RETRY_BLOCKED.json',dict(error=str(e),meaning='Map observation codec preflight failed; no second inference launched'))
                status('stopped',reason='retry_codec_preflight_blocked',attempt=1);return
        else:
            original=pipe.sample_sparse_structure
            def record(*a,**kw):
                value=original(*a,**kw);np.savez_compressed(d/'GENERATED_STRUCTURE.npz',coords=value.cpu().numpy());return value
            pipe.sample_sparse_structure=record
        for method,label in [('get_cond','image_condition'),('sample_sparse_structure','structure'),('sample_shape_slat_cascade','shape'),('sample_tex_slat','appearance'),('decode_shape_slat','decode_geometry')]:
            fn=getattr(pipe,method)
            def wrapped(*a,_fn=fn,_label=label,**kw):
                stage(_label);v=_fn(*a,**kw);stage(_label+'_done');return v
            setattr(pipe,method,wrapped)
        native.OUT=d;native.SEED=p['seed'];native.generate()
        fixed_export(d/'native_street_basecolor.glb',d/'world.glb',p['boundary']['fixed_world_frame']['glTF_root_matrix'])
        save(d/'COMPLETE.json',dict(id=f'attempt-{i}',seed=p['seed'],sha256=sha(d/'world.glb'),bytes=(d/'world.glb').stat().st_size,
             seconds=time.time()-began,completed_epoch=time.time(),resident_pipe_id=id(pipe),same_model_instance=True,
             generated_geometry=True,posthoc_repair=False,placement_fitted=False))
        status('verifying',attempt=i,started_epoch=began)
        # Fresh process and fixed independent verifier, not model self-scoring.
        with (d/'verify.log').open('x') as f:
            run=subprocess.run([sys.executable,'-B','-u',str(JOB/'v27_verify.py'),str(d)],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,timeout=90)
        assert run.returncode==0,'Independent verifier failed; never promote asset'
        report=read(d/'REPORT.json')
        if report['decision']['accepted']:
            # Atomic no-clobber commit; accepted chunks cannot be replaced by retries.
            save(OUT/'ACCEPTED.json',dict(id='map10-0',attempt=i,sha256=report['asset_sha256'],report_sha256=sha(d/'REPORT.json')))
            status('stopped',reason='accepted_committed',attempt=i);return
        save(d/'QUARANTINED.json',dict(epoch=time.time(),reasons=report['decision']['reasons'],world_unchanged=True))
        status('feedback_ready',attempt=i,reasons=report['decision']['reasons'])
    status('stopped',reason='bounded_retry_exhausted_keep_scaffold',attempt=1)
def owner():
    contract=read(OUT/'CONTRACT.json');verify_sources()
    save(OUT/'ONE_OWNER.json',dict(epoch=time.time(),pid=os.getpid()))
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    lease=None
    for index,uuid in GPUS:
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if uuid in owners or f'{index}, {uuid}' not in inventory:continue
        handle=(JOB/'gpu_leases'/f'gpu{index}.lock').open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();continue
        now=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if uuid in now:handle.close();continue
        lease=handle;gpu=dict(index=index,uuid=uuid);break
    assert lease is not None,'No authorized idle GPU; no work launched'
    for name in ['tmp','cache','extensions']:(OUT/name).mkdir(exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'CUDA_HOME':'/usr/local/cuda-11.8','SPCONV_ALGO':'native','ATTN_BACKEND':'flash_attn','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'TMPDIR':str(OUT/'tmp'),'XDG_CACHE_HOME':str(OUT/'cache'),'TORCH_EXTENSIONS_DIR':str(OUT/'extensions')}
    began=time.time()
    try:
        with (OUT/'worker.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--work'],cwd=JOB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            save(OUT/'OWNER_LAUNCH.json',dict(epoch=began,child_pid=child.pid,gpu=gpu,limit_seconds=contract['lease_seconds']))
            while child.poll() is None:
                current=read(OUT/'WORKER_STATUS.json') if (OUT/'WORKER_STATUS.json').exists() else {}
                overdue=time.time()-began>contract['lease_seconds'] or current.get('stage')=='generating' and time.time()-current['started_epoch']>contract['per_attempt_seconds']
                if overdue:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    status('stopped',reason='bounded_watchdog');break
                time.sleep(.5)
            code=child.wait()
        save(OUT/'GPU_COST.json',dict(gpu=gpu,start_epoch=began,end_epoch=time.time(),gpu_seconds=time.time()-began,
             includes_warmup_and_idle=True,physical_lease_count=1,returncode=code))
        if code:status('failed',reason='worker_error',returncode=code)
    finally:lease.close()
if __name__=='__main__':
    if '--work' in sys.argv:work()
    elif '--detach' in sys.argv:
        with (OUT/'owner.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner()

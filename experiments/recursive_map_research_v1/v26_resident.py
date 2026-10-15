"""Camera-requested resident TRELLIS.2 service, fixed3 chunks, zero training.

One pipeline load per worker, not per chunk. Native samples/decoders unchanged.
Lease900s maximum; idle300s; per-generation300s watchdog; no unrelated jobs touched.
"""
from paths import JOB,read,save,sha
from pathlib import Path
import os,sys,time,json,shutil,subprocess,fcntl,signal
OUT=JOB/'map_object_world_v1/trellis2_walk_generate_v26'
GPUS=[(0,'GPU-e0a56bf9-85ef-c658-67fa-457b76585254'),(1,'GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa'),(2,'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'),(3,'GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d')]
def atomic(path,data):
    p=Path(path);tmp=p.with_suffix('.next');tmp.write_text(json.dumps(data));os.replace(tmp,p)
def status(stage,**kw):atomic(OUT/'WORKER_STATUS.json',dict(stage=stage,epoch=time.time(),**kw))
def prepare():
    assert not OUT.exists(),'Preserve this run'
    OUT.mkdir(parents=True)
    prior=JOB/'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1'
    for i in (4,5,6):
        d=OUT/'chunks'/f'walk-0--{i}';d.mkdir(parents=True)
        for n in ('input.png','ROAD_OBSERVATION.npz'):shutil.copyfile(prior/n,d/n)
        p=read(prior/'PLAN.json');scale=p['boundary']['fixed_world_frame']['global_native_to_meter_scale']
        p.update(seed=942260+i,input=str(d/'input.png'),input_sha256=sha(d/'input.png'),chunk_id=f'walk-0--{i}',
            fixed_chunk_translation_GLTF=[0,0,-i*scale],streaming_only_change='One resident pipeline, actual navigation gates every request',
            no_automatic_extension=True,GPU_limit_seconds=300)
        save(d/'PLAN.json',p)
        atomic(d/'STATUS.json',dict(status='waiting_camera',epoch=time.time()))
    (OUT/'tmp').mkdir();(OUT/'cache').mkdir();(OUT/'extensions').mkdir()
    save(OUT/'RUN_CONTRACT.json',dict(max_new_chunks=3,seeds=[942264,942265,942266],
        resident_limit_seconds=900,idle_limit_seconds=300,per_generation_limit_seconds=300,
        no_generation_before_camera=True,stop_new_requests_after_budget=True,
        native_pipeline='1024_cascade',native_sampling_steps=12,training=False,
        map10_control=False,seamless_claim=False,mesh_repair=False,
        independent_world_reset=False,cached_v21_v24_v25_assets_immutable=True))
    files=[Path(__file__),JOB/'trellis2_native_street_v17.py',JOB/'v24_boundary_sampler.py',JOB/'v24_native_job.py']
    save(OUT/'SOURCE_FREEZE.json',{str(f):sha(f) for f in files})
    status('prepared',no_inference_yet=True)
    print('V26_PREPARED',flush=True)

def fixed_export(source,dest,matrix):
    import struct
    with source.open('rb') as src,dest.open('xb') as dst:
        magic,version,total=struct.unpack('<III',src.read(12));n,t=struct.unpack('<II',src.read(8));doc=json.loads(src.read(n))
        scene=doc['scenes'][doc.get('scene',0)];children=scene['nodes'];idx=len(doc['nodes'])
        doc['nodes'].append(dict(matrix=matrix,children=children));scene['nodes']=[idx]
        text=json.dumps(doc,separators=(',',':')).encode();text+=b' '*((-len(text))%4)
        dst.write(struct.pack('<III',magic,version,total+len(text)-n));dst.write(struct.pack('<II',len(text),t));dst.write(text);shutil.copyfileobj(src,dst,8*1024*1024)

def work():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h
    status('loading_python',generation_started=False)
    import trellis2_native_street_v17 as native
    os.environ['CUDA_HOME']=str(native.DEST/'environment');os.environ['PATH']=str(native.DEST/'environment/bin')+':'+os.environ['PATH']
    sys.path.insert(0,str(native.REPO))
    import torch,numpy as np,types,trimesh,copy
    from trellis2.pipelines import Trellis2ImageTo3DPipeline,rembg
    from v24_boundary_sampler import install
    rembg.UnusedBackgroundRemoval=lambda:None
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    start=time.time();status('loading_model',generation_started=False)
    runtime=read(OUT/'chunks/walk-0--4/PLAN.json')['runtime']
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(runtime);pipe.cuda()
    methods=['get_cond','sample_sparse_structure','sample_shape_slat_cascade','sample_tex_slat','decode_shape_slat','decode_latent']
    canonical={name:getattr(type(pipe),name) for name in methods}
    original_sampler=copy.deepcopy(pipe.sparse_structure_sampler)
    hits=[]
    @classmethod
    def resident_get(cls,path,*a,**kw):
        assert str(path)==runtime,'Do not silently swap model'
        hits.append(dict(epoch=time.time(),pipe_id=id(pipe),same_instance=True))
        return pipe
    Trellis2ImageTo3DPipeline.from_pretrained=resident_get
    save(OUT/'MODEL_READY.json',dict(epoch=time.time(),load_seconds=time.time()-start,pipe_id=id(pipe),
        pipeline_sha256=sha(Path(runtime)/'pipeline.json'),low_vram=pipe.low_vram,
        parameter_dtypes={k:sorted({str(x.dtype) for x in m.parameters()}) for k,m in pipe.models.items()},
        resident_CPU_weights=True,CUDA_allocated_GiB=torch.cuda.memory_allocated()/2**30,
        generation_count=0))
    completed=[];idle=time.time()
    status('resident_ready',generation_started=False,completed=completed,pipe_id=id(pipe))
    while len(completed)<3 and time.time()-idle<300:
        if (OUT/'STOP').exists():break
        waiting=[d for d in sorted((OUT/'chunks').iterdir()) if (d/'REQUEST.json').exists() and not (d/'ONE_ATTEMPT.json').exists()]
        if not waiting:time.sleep(.1);continue
        d=waiting[0];p=read(d/'PLAN.json');request=read(d/'REQUEST.json')
        assert request['chunk_id']==p['chunk_id']
        with (d/'ONE_ATTEMPT.json').open('x') as f:json.dump(dict(epoch=time.time(),request=request),f)
        began=time.time();save(d/'LAUNCH.json',dict(epoch=began,worker_pid=os.getpid(),resident_pipe_id=id(pipe),model_reloaded=False))
        status('generating',chunk_id=p['chunk_id'],completed=completed,pipe_id=id(pipe))
        def stage(name,**kw):
            torch.cuda.synchronize();value=dict(stage=name,status='generating',epoch=time.time(),**kw)
            atomic(d/'STATUS.json',value)
            with (d/'STAGES.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        try:
            # Reset wrappers, not weights or learned state, before rebinding the current chunk.
            for name,method in canonical.items():setattr(pipe,name,types.MethodType(method,pipe))
            pipe.sparse_structure_sampler=copy.deepcopy(original_sampler)
            install(pipe,d)
            for method,label in [('get_cond','image_condition'),('sample_sparse_structure','structure'),('sample_shape_slat_cascade','shape'),('sample_tex_slat','appearance')]:
                original=getattr(pipe,method)
                def wrapped(*a,_fn=original,_label=label,**kw):
                    stage(_label);value=_fn(*a,**kw);stage(_label+'_done');return value
                setattr(pipe,method,wrapped)
            original_shape=pipe.decode_shape_slat
            def shape(*a,**kw):
                stage('decode_geometry');value=original_shape(*a,**kw);m=value[0][0]
                v=m.vertices.detach().float().cpu().numpy();f=m.faces.detach().cpu().numpy()
                axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
                scene=trimesh.Scene(trimesh.Trimesh(vertices=v@axis.T,faces=f,process=False))
                scene.apply_transform(np.array(p['boundary']['fixed_world_frame']['glTF_root_matrix']).reshape(4,4).T)
                path=d/'geometry_preview.glb';scene.export(path)
                save(d/'PREVIEW.json',dict(epoch=time.time(),sha256=sha(path),faces=len(f),vertices=len(v),actual_native_geometry=True))
                stage('geometry_preview_ready');return value
            pipe.decode_shape_slat=shape
            native.OUT=d;native.SEED=p['seed']
            # Existing official pipeline and exact historical export code; no model rebuild.
            native.generate()
            stage('exporting')
            asset=d/'chunk_100m.glb';fixed_export(d/'native_street_basecolor.glb',asset,p['boundary']['fixed_world_frame']['glTF_root_matrix'])
            scene=trimesh.load(asset,force='scene',process=False)
            assert all(len(g.faces)>0 and np.isfinite(g.vertices).all() for g in scene.geometry.values())
            save(d/'RELOAD.json',dict(epoch=time.time(),faces=sum(len(g.faces) for g in scene.geometry.values()),bounds=scene.bounds.tolist(),repair=False))
            result=dict(id=p['chunk_id'],seed=p['seed'],sha256=sha(asset),bytes=asset.stat().st_size,
                position=p['fixed_chunk_translation_GLTF'],generation_seconds=time.time()-began,
                resident_pipe_id=id(pipe),model_reloaded=False,completed_epoch=time.time())
            save(d/'COMPLETE.json',result);atomic(d/'STATUS.json',dict(status='ready',stage='cached',**result))
            completed.append(p['chunk_id']);idle=time.time()
            atomic(OUT/'CACHE_HITS.json',hits)
            status('resident_ready',completed=completed,pipe_id=id(pipe))
        except Exception as e:
            atomic(d/'STATUS.json',dict(status='failed',stage='failed',epoch=time.time(),error=str(e)))
            status('failed',chunk_id=p['chunk_id'],error=str(e),completed=completed);raise
    status('stopped',completed=completed,reason='budget_complete' if len(completed)==3 else 'idle_timeout_or_stop',pipe_id=id(pipe))

def owner():
    with (OUT/'ONE_OWNER.json').open('x') as f:json.dump(dict(epoch=time.time(),pid=os.getpid()),f)
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    lease=None
    for index,uuid in GPUS:
        if f'{index}, {uuid}' not in inventory or uuid in owners:continue
        lock=(JOB/'gpu_leases'/f'gpu{index}.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        now=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if uuid in now:lock.close();continue
        lease=lock;gpu=dict(index=index,uuid=uuid);break
    assert lease is not None,'No authorized idle A800; no process launched'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'CUDA_HOME':'/usr/local/cuda-11.8','SPCONV_ALGO':'native','ATTN_BACKEND':'flash_attn','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'TMPDIR':str(OUT/'tmp'),'XDG_CACHE_HOME':str(OUT/'cache'),'CUDA_CACHE_PATH':str(OUT/'cache'),'TORCH_EXTENSIONS_DIR':str(OUT/'extensions')}
    start=time.time()
    try:
        with (OUT/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--work'],cwd=JOB,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(OUT/'OWNER_LAUNCH.json',dict(epoch=start,child_pid=child.pid,gpu=gpu,limit_seconds=900))
            while child.poll() is None:
                current=read(OUT/'WORKER_STATUS.json')
                stage=current.get('stage')
                # Separate finite total residency and generation watchdogs.
                too_long=time.time()-start>=900 or stage=='generating' and time.time()-current['epoch']>=300
                if too_long:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    status('stopped',reason='bounded_watchdog');break
                time.sleep(.5)
            code=child.wait()
        save(OUT/'GPU_COST.json',dict(gpu=gpu,start_epoch=start,end_epoch=time.time(),gpu_seconds=time.time()-start,
            includes_warmup_and_idle=True,physical_lease_window=True,returncode=code))
    finally:lease.close()

if __name__=='__main__':
    if '--prepare'in sys.argv:prepare()
    elif '--work'in sys.argv:work()
    elif '--detach'in sys.argv:
        with (OUT/'owner.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--owner'],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner()

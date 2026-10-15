"""One camera-requested native TRELLIS.2 inference. Prepare is CPU-only."""
from paths import JOB, read, save, sha
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,shutil,signal,hashlib
OUT=JOB/'map_object_world_v1/trellis2_online_v23/chunks/online-0--1'
SEED=942219
GPUS=[(0,'GPU-e0a56bf9-85ef-c658-67fa-457b76585254'),(1,'GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa'),(2,'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'),(3,'GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d')]
def status(state,**kw):
    value={'status':state,'epoch':time.time(),'seed':SEED,**kw};p=OUT/'STATUS.next';p.write_text(json.dumps(value));os.replace(p,OUT/'STATUS.json')
def prepare():
    assert not OUT.exists(),'Preserve existing attempt; no restart or second sample'
    OUT.mkdir(parents=True)
    for name in ['logs','work','cache','tmp','extensions']:(OUT/name).mkdir()
    old=JOB/'map_object_world_v1/trellis2_streaming_v22/chunk/native-adjacent-001'
    shutil.copyfile(old/'input.png',OUT/'input.png')
    p=read(old/'PLAN.json')|{'seed':SEED,'input':str(OUT/'input.png'),'input_sha256':sha(OUT/'input.png'),
      'chunk_id':'online-0--1','one_generation_only':True,'GPU_limit_seconds':600,
      'generation_trigger':'Actual browser camera position; no asset or inference before request',
      'parent_chunk_id':'native-v21-000','fixed_chunk_translation_GLTF':[0,0,-100],
      'reference_kind':'Reuse legitimate fixed v22 conditioning image; new geometry sampled only after camera event'}
    save(OUT/'PLAN.json',p)
    frozen=[Path(__file__),JOB/'trellis2_native_street_v17.py',OUT/'input.png',OUT/'PLAN.json',Path(p['runtime'])/'pipeline.json']
    save(OUT/'SOURCE_FREEZE.json',{str(x):sha(x) for x in frozen})
    status('waiting_camera',asset_exists=False,training=False)
    print(json.dumps({'prepared':str(OUT),'asset_exists':False,'seed':SEED}),flush=True)
def verify():
    for p,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(p)==h,('CHANGED',p)
def generate():
    verify();import trellis2_native_street_v17 as native
    native.OUT=OUT;native.SEED=SEED;native.generate()
def run():
    verify()
    assert (OUT/'REQUEST.json').is_file(),'Camera request must precede inference'
    with (OUT/'ONE_ATTEMPT.json').open('x') as f:json.dump({'epoch':time.time(),'camera_request':read(OUT/'REQUEST.json'),'no_asset_before_launch':not (OUT/'ASSET.json').exists()},f)
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    lease=None
    for index,uuid in GPUS:
        if f'{index}, {uuid}' not in inventory or uuid in owners:continue
        candidate=(JOB/'gpu_leases'/f'gpu{index}.lock').open('a')
        try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:candidate.close();continue
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if uuid in owners:candidate.close();continue
        lease=candidate;gpu={'index':index,'uuid':uuid};break
    assert lease is not None,'No authorized idle GPU; attempt stops without launching'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'CUDA_HOME':'/usr/local/cuda-11.8','SPCONV_ALGO':'native','ATTN_BACKEND':'flash_attn',
      'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
      'TMPDIR':str(OUT/'tmp'),'XDG_CACHE_HOME':str(OUT/'cache'),'CUDA_CACHE_PATH':str(OUT/'cache'),'TORCH_EXTENSIONS_DIR':str(OUT/'extensions')}
    start=time.time();code=None
    try:
        with (OUT/'logs/generation.log').open('x') as log:
            command=[sys.executable,'-B','-u',str(Path(__file__)),'--generate']
            child=subprocess.Popen(command,cwd=JOB,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(OUT/'LAUNCH.json',{'epoch':start,'pid':child.pid,'GPU':gpu,'command':command,'request':read(OUT/'REQUEST.json')})
            status('generating',start_epoch=start,gpu=gpu,pid=child.pid,hard_limit_seconds=600)
            while child.poll() is None:
                if time.time()-start>=600:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(1)
            code=child.wait()
        elapsed=time.time()-start
        save(OUT/'GPU_COST.json',{'gpu':gpu,'start_epoch':start,'end_epoch':time.time(),'gpu_seconds':elapsed,'includes_load_idle':True,'returncode':code})
        assert code==0,('Native generation failed',code)
    finally:lease.close()
    status('exporting',start_epoch=start,generation_seconds=elapsed)
    import trellis2_long_street_v21_package as export
    export.OUT=OUT;export.metric_export()
    os.rename(OUT/'large_street_100m.glb',OUT/'chunk_100m.glb')
    # Independent CPU reload of the saved complete asset, never a source image.
    import trimesh,numpy as np
    asset=OUT/'chunk_100m.glb';scene=trimesh.load(asset,force='scene',process=False)
    v=sum(len(g.vertices) for g in scene.geometry.values());f=sum(len(g.faces) for g in scene.geometry.values())
    assert v>0 and f>0 and all(np.isfinite(g.vertices).all() for g in scene.geometry.values())
    save(OUT/'RELOAD.json',{'new_process_export_stage':True,'vertices':v,'faces':f,'world_bounds':scene.bounds.tolist(),'geometry_repair':False})
    result={'id':'online-0--1','status':'ready','seed':SEED,'sha256':sha(asset),'bytes':asset.stat().st_size,'vertices':v,'faces':f,
      'position':[0,0,-100],'native_generation_seconds':read(OUT/'ASSET.json')['seconds'],'gpu_seconds':elapsed,'completed_epoch':time.time(),
      'camera_request':read(OUT/'REQUEST.json'),'map_aligned':False,'seamless_guarantee':False,'no_more_jobs':True}
    save(OUT/'COMPLETE.json',result);status('ready',**{k:v for k,v in result.items() if k!='status'})
    print('ONLINE_CHUNK_READY',json.dumps(result),flush=True)
if __name__=='__main__':
    try:
        if '--prepare' in sys.argv:prepare()
        elif '--generate' in sys.argv:generate()
        elif '--detach' in sys.argv:
            with (OUT/'owner.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--run'],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(OUT/'OWNER.json',{'pid':child.pid,'epoch':time.time(),'bounded':True});print(json.dumps({'pid':child.pid}),flush=True)
        else:run()
    except Exception as e:
        if OUT.exists():status('failed',error=str(e),no_retry=True)
        raise

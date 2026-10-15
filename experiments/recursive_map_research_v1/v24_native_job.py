"""One online road-boundary experiment. Reuse v23 ownership and frozen prior."""
from pathlib import Path
_old=Path(__file__).with_name('v23_native_job.py').read_text()
_old=_old.split("\nif __name__=='__main__':")[0]
_old=_old.replace('trellis2_online_v23','trellis2_boundary_online_v24').replace('online-0--1','road-boundary-0--1')
_old=_old.replace("'position':[0,0,-100]","'position':read(OUT/'PLAN.json')['fixed_chunk_translation_GLTF']")
_old=_old.replace("    import trellis2_long_street_v21_package as export\n    export.OUT=OUT;export.metric_export()",'    fixed_world_export()')
exec(compile(_old,str(Path(__file__).with_name('v23_native_job.py')),'exec'),globals())
import numpy as np,struct
def prepare():
    assert not OUT.exists(),'Preserve this one fixed attempt'
    OUT.mkdir(parents=True)
    for n in ['logs','work','cache','tmp','extensions']:(OUT/n).mkdir()
    prior=JOB/'map_object_world_v1/trellis2_online_v23/chunks/online-0--1'
    parent=JOB/'map_object_world_v1/trellis2_long_street_v21'
    p=read(prior/'PLAN.json');shutil.copyfile(JOB/'map_object_world_v1/trellis2_streaming_v22/chunk/native-adjacent-001/input.png',OUT/'input.png')
    frame=read(parent/'ASSUMED_SCALE.json');route=read(parent/'STREET_ROUTE.json');scale=frame['global_native_to_meter_scale']
    # Reuse the already observed road interval and measured parent floor.
    center=-.007;half_width=.035;floor=route['ground_native_z'];c=(np.arange(64)+.5)/64-.5
    road_x=np.abs(c-center)<=half_width;zi=int(np.argmin(np.abs(c-floor)))
    ground=np.zeros((64,64,64),bool);ground[road_x,:,zi]=True
    known=road_x[:,None,None]&np.ones((1,64,1),bool)&((c>=floor-3/scale)&(c<=floor+8/scale))[None,None,:]
    np.savez_compressed(OUT/'ROAD_OBSERVATION.npz',target=ground.astype(np.float32),known=known,ground=ground)
    p.update(input=str(OUT/'input.png'),input_sha256=sha(OUT/'input.png'),chunk_id='road-boundary-0--1',
      fixed_chunk_translation_GLTF=[0,0,-scale],GPU_limit_seconds=600,
      boundary={'method':'local flow-time latent observation replacement; unknown region remains native',
       'native_inpainting_supported':False,'not_v20_global_projection':True,'encoder':str(JOB.parent/'map_native_scene_dit_v1/weights/codec/ckpts/ss_enc_conv3d_16l8_fp16'),
       'source_parent_asset':str(parent/'large_street_100m.glb'),'source_parent_sha256':sha(parent/'large_street_100m.glb'),
       'source_route':str(parent/'STREET_ROUTE.json'),'road_center_native_x':center,'road_half_width_native':half_width,
       'ground_native_z':floor,'clearance_display_units':8,'scope':'straight road extension hypothesis from existing endpoint, not Map10 truth',
       'fixed_world_frame':frame,'fixed_before_output':True,'no_output_fit_or_rotation':True,
       'only_intervention':'world-addressed road-boundary conditioning contract, including the same fixed frame at generation and export'},
      matched_control='v23 seed942219; same image/native models/samplers; old independently scaled world frame is preserved as failed control',
      GO_gates={'road_center_hit_fraction':.95,'max_road_surface_step_display_units':.5,'road_low_camera_obstruction_fraction':.05,
       'recognized_buildings_both_sides':True,'old_asset_SHA_unchanged':True},
      no_automatic_extension=True,no_posthoc_geometry_repair=True)
    save(OUT/'PLAN.json',p)
    enc=p['boundary']['encoder']
    files=[Path(__file__),JOB/'v24_boundary_sampler.py',JOB/'v23_native_job.py',JOB/'trellis2_native_street_v17.py',OUT/'PLAN.json',OUT/'ROAD_OBSERVATION.npz',OUT/'input.png',Path(p['runtime'])/'pipeline.json',parent/'large_street_100m.glb',parent/'STREET_ROUTE.json',Path(enc+'.json'),Path(enc+'.safetensors')]
    save(OUT/'SOURCE_FREEZE.json',{str(f):sha(f) for f in files});status('waiting_camera',asset_exists=False)
    print('V24_PREPARED',json.dumps(p['boundary']),flush=True)
def generate():
    verify()
    import trellis2_native_street_v17 as native
    os.environ['CUDA_HOME']=str(native.DEST/'environment');sys.path.insert(0,str(native.REPO))
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from v24_boundary_sampler import install
    original=Trellis2ImageTo3DPipeline.from_pretrained
    @classmethod
    def load(cls,*args,**kwargs):
        pipe=original(*args,**kwargs);install(pipe,OUT);return pipe
    Trellis2ImageTo3DPipeline.from_pretrained=load
    native.OUT=OUT;native.SEED=SEED;native.generate()
    # Saved native geometry, diagnostic rays only. No mesh changes.
    import torch,nvdiffrast.torch as dr
    with np.load(OUT/'RAW_MESH.npz') as a:v=torch.tensor(a['vertices'],device='cuda',dtype=torch.float32);faces=torch.tensor(a['faces'],device='cuda',dtype=torch.int32)
    ctx=dr.RasterizeCudaContext();height=torch.full((512,512),-2.,device='cuda');hit=torch.zeros((512,512),device='cuda',dtype=torch.bool)
    clip=torch.cat([v[:,:2]*2,-v[:,2:3],torch.ones_like(v[:,:1])],-1)[None].contiguous()
    for start in range(0,len(faces),8000000):
        fc=faces[start:start+8000000].contiguous();rast,_=dr.rasterize(ctx,clip,fc,resolution=[512,512]);z,_=dr.interpolate(v[:,2:3][None].contiguous(),rast,fc)
        mask=rast[0,:,:,3]>0;height=torch.maximum(height,torch.where(mask,z[0,:,:,0],-2.));hit|=mask
    np.savez_compressed(OUT/'FINAL_HEIGHTFIELD.npz',height=height.cpu().numpy(),occupied=hit.cpu().numpy())
    p=read(OUT/'PLAN.json');frame=p['boundary']['fixed_world_frame'];s=frame['global_native_to_meter_scale'];f=frame['native_floor_for_display']
    clip=torch.cat([v[:,:1]*2,((v[:,2:3]-f)*s/5)*2-1,v[:,1:2]*2,torch.ones_like(v[:,:1])],-1)[None].contiguous();blocked=torch.zeros((512,512),device='cuda',dtype=torch.bool)
    for start in range(0,len(faces),8000000):
        rast,_=dr.rasterize(ctx,clip,faces[start:start+8000000].contiguous(),resolution=[512,512]);blocked|=rast[0,:,:,3]>0
    np.savez_compressed(OUT/'FORWARD_VISIBILITY.npz',occupied=blocked.cpu().numpy(),vertical_world_range=[0,5],native_x_range=[-.5,.5],native_y_range=[-.5,.5],no_near_plane_exclusion=True)
def fixed_world_export():
    # A frame fixed from the parent before inference, never based on new bounds.
    p=read(OUT/'PLAN.json');matrix=p['boundary']['fixed_world_frame']['glTF_root_matrix']
    source=OUT/'native_street_basecolor.glb';dest=OUT/'large_street_100m.glb'
    with source.open('rb') as src,dest.open('xb') as dst:
        magic,version,total=struct.unpack('<III',src.read(12));n,t=struct.unpack('<II',src.read(8));doc=json.loads(src.read(n))
        scene=doc['scenes'][doc.get('scene',0)];children=scene['nodes'];index=len(doc['nodes'])
        doc['nodes'].append({'name':'parent_world_frame_fixed_before_generation','matrix':matrix,'children':children});scene['nodes']=[index]
        text=json.dumps(doc,separators=(',',':')).encode();text+=b' '*((-len(text))%4)
        dst.write(struct.pack('<III',magic,version,total+len(text)-n));dst.write(struct.pack('<II',len(text),t));dst.write(text);shutil.copyfileobj(src,dst,8*1024*1024)
    save(OUT/'FIXED_WORLD_EXPORT.json',{'matrix':matrix,'translation':p['fixed_chunk_translation_GLTF'],'output_bounds_used_for_alignment':False,'buffers_unchanged':True,'sha256':sha(dest)})
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

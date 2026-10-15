"""One native pretrained image-to-3D street-block sample; no map adapters."""
from paths import JOB,read,save,sha
from trellis2_prepare_v1 import DEST,REPO
from pathlib import Path
import argparse,os,sys,time,random,ast,inspect,textwrap,types

OUT=JOB/'map_object_world_v1/trellis2_native_street_v17'
SEED=942217

def prepare():
    from PIL import Image
    assert not OUT.exists()
    source=REPO/'assets/example_image/4bc7abe209c8673dd3766ee4fad14d40acbed02d118e7629f645c60fd77313f1.webp'
    image=Image.open(source).convert('RGBA')
    side=max(image.size);canvas=Image.new('RGBA',(side,side),(255,255,255,255))
    canvas.alpha_composite(image,((side-image.width)//2,(side-image.height)//2))
    path=OUT/'input.png';path.parent.mkdir(parents=True);canvas.convert('RGB').save(path)
    runtime=JOB/'map_object_world_v1/trellis2_resume_v3/runtime'
    save(OUT/'PLAN.json',{'model':'Original pretrained TRELLIS.2-4B','source_commit':'75fbf0183001ed9876c8dbb35de6b668552ee08bd',
        'reference':str(source),'reference_sha256':sha(source),'reference_kind':'Official multi-building urban block example, stylized not a real survey',
        'input':str(path),'input_sha256':sha(path),'preprocessing':'All pixels preserved; alpha composite on white; square padding; no rembg',
        'runtime':str(runtime),'runtime_sha256':sha(runtime/'pipeline.json'),'seed':SEED,
        'pipeline_type':'1024_cascade','samplers':'Original pretrained defaults unchanged','max_num_tokens':49152,
        'training_updates':0,'map_adapter_loaded':False,'LoRA_loaded':False,'map_control_claim':False,
        'one_complete_scene_sample':True,'not_separate_buildings_assembled':True,'no_seed_search':True,
        'geometry_postprocessing':'Skip optional decode_latent fill_holes only; no geometry repair/simplification',
        'export':'All native triangles and model base colors; complete PBR also saved in RAW_NATIVE.pt',
        'GPU_limit_seconds':3600,'fixed_before_generation':True,
        'street_camera_rule':'After reload: ground=min Z, eye=ground+0.08*height, +/-1.1 XY; same fixed rule without view selection'})
    save(OUT/'SOURCE_FREEZE.json',{str(JOB/n):sha(JOB/n) for n in ['trellis2_native_street_v17.py','trellis2_native_street_v17_views.py','trellis2_native_street_v17_owner.py']}
        |{str(path):sha(path),str(source):sha(source),str(runtime/'pipeline.json'):sha(runtime/'pipeline.json')})
    print('NATIVE_BLOCK_REGISTERED',SEED,image.size,flush=True)

def generate():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUDA_HOME']=str(DEST/'environment');os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
    sys.path.insert(0,str(REPO))
    import torch,numpy as np,trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline,rembg
    rembg.UnusedBackgroundRemoval=lambda:None
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    p=read(OUT/'PLAN.json');assert sha(p['input'])==p['input_sha256']
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    start=time.time();pipe=Trellis2ImageTo3DPipeline.from_pretrained(p['runtime']);pipe.cuda()
    save(OUT/'LOAD.json',{'parameter_dtypes':{k:sorted({str(x.dtype) for x in m.parameters()}) for k,m in pipe.models.items()},
        'samplers':{'Structure':pipe.sparse_structure_sampler_params,'Shape':pipe.shape_slat_sampler_params,'texture':pipe.tex_slat_sampler_params},
        'low_vram':pipe.low_vram,'native_models':read(Path(p['runtime'])/'pipeline.json')['args']['models'],
        'adapter_weights_loaded':False,'original_weights_frozen':True})
    # Preserve the raw native decoder geometry; remove only the optional repair call.
    original_decode=inspect.unwrap(pipe.decode_latent.__func__)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original_decode)))
    class NoRepair(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='fill_holes':
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    tree=ast.fix_missing_locations(NoRepair().visit(tree));scope=dict(original_decode.__globals__)
    exec(compile(tree,'native_decoder_without_optional_hole_fill','exec'),scope)
    pipe.decode_latent=types.MethodType(scope['decode_latent'],pipe)
    with torch.no_grad():
        meshes,latents=pipe.run(Image.open(p['input']).convert('RGB'),seed=SEED,pipeline_type='1024_cascade',
            preprocess_image=False,return_latent=True,max_num_tokens=49152)
    mesh=meshes[0];v=mesh.vertices.detach().float().cpu().numpy().copy();f=mesh.faces.detach().cpu().numpy().copy()
    assert len(f)>0 and np.isfinite(v).all()
    with (OUT/'RAW_MESH.npz').open('xb') as h:np.savez_compressed(h,vertices=v,faces=f)
    torch.save({'vertices':mesh.vertices.cpu(),'faces':mesh.faces.cpu(),'attrs':mesh.attrs.cpu(),
        'coords':mesh.coords.cpu(),'layout':mesh.layout,'voxel_size':mesh.voxel_size,
        'shape_slat':latents[0].feats.cpu(),'shape_coords':latents[0].coords.cpu(),
        'tex_slat':latents[1].feats.cpu(),'resolution':latents[2]},OUT/'RAW_NATIVE.pt')
    colors=[]
    for offset in range(0,len(v),250000):
        with torch.no_grad():a=mesh.query_attrs(mesh.vertices[offset:offset+250000])
        rgb=a[:,mesh.layout['base_color']].clamp(0,1)
        colors.append((rgb*255).byte().cpu().numpy())
    axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
    rgba=np.c_[np.concatenate(colors),np.full(len(v),255,np.uint8)]
    model=trimesh.Trimesh(vertices=v@axis.T,faces=f,vertex_colors=rgba,process=False)
    asset=OUT/'native_street_basecolor.glb';trimesh.Scene(model).export(asset)
    fresh=trimesh.load(asset,force='scene',process=False);g=list(fresh.geometry.values())[0]
    assert np.array_equal(g.faces,f) and np.allclose(g.vertices,v@axis.T,atol=1e-7)
    save(OUT/'ASSET.json',{'asset':str(asset),'sha256':sha(asset),'seed':SEED,'vertices':len(v),'faces':len(f),
        'native_bounds':[v.min(0).tolist(),v.max(0).tolist()],'all_native_geometry_preserved':True,
        'generated_geometry':True,'generated_appearance':True,'generated_as_single_scene':True,
        'appearance_display':'native base-color sampled at unchanged vertices; opaque preview; full PBR in RAW_NATIVE.pt',
        'PBR_sha256':sha(OUT/'RAW_NATIVE.pt'),'raw_mesh_sha256':sha(OUT/'RAW_MESH.npz'),
        'no_geometry_repair':True,'zero_training':True,'no_map_control':True,'seconds':time.time()-start,
        'max_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
    print('NATIVE_STREET_ASSET_READY',len(v),len(f),asset,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else generate()

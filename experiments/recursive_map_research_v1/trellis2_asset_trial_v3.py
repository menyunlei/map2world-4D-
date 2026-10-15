"""Fixed zero-training TRELLIS.2 trials, local official weights, raw meshes retained."""
from paths import JOB, save, read, sha
from trellis2_prepare_v1 import DEST, REPO
from pathlib import Path
import argparse
import ast
import copy
import inspect
import json
import os
import random
import sys
import time

OUT = JOB / 'map_object_world_v1/trellis2_resume_v3'
DINO = Path('/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/facebook/dinov3-vitl16-pretrain-lvd1689m')
SEEDS = [942201, 942202]

def register():
    from PIL import Image
    source = JOB/'map10_source.jpg'
    assert sha(source) == '066bb11bfc99ebccae10505f235167cb0f9c9dcca06d24343d1328a29842cc7b'
    source2 = REPO/'assets/example_image/b205f4483c47bd1fec8e229163361e4fdff9f77923c5e968343b8f1dd76b61dc.webp'
    inputs = []
    for name, path, crop in [('map10_block', source, [30,470,1045,1680]),
                             ('official_building', source2, None)]:
        image = Image.open(path).convert('RGBA')
        if crop:
            image = image.crop(crop)
        canvas = Image.new('RGBA', (max(image.size), max(image.size)), (0,0,0,255))
        canvas.alpha_composite(image, ((canvas.width-image.width)//2,(canvas.height-image.height)//2))
        target = OUT/'inputs'/name/'input.png'
        target.parent.mkdir(parents=True, exist_ok=False)
        canvas.convert('RGB').save(target)
        inputs.append({'id':name,'source':str(path),'source_sha256':sha(path),
                       'crop_xyxy':crop,'input':str(target),'input_sha256':sha(target),
                       'preprocessing':'fixed crop if declared, preserve all source pixels; alpha composite on black; square padding; no background segmentation'})
    config = copy.deepcopy(read(DEST/'weights/TRELLIS.2-4B/pipeline.json'))
    for key, val in config['args']['models'].items():
        if val.startswith('microsoft/'):
            config['args']['models'][key] = str(DEST/'weights/TRELLIS-image-large/ckpts'/Path(val).name)
        else:
            config['args']['models'][key] = str(DEST/'weights/TRELLIS.2-4B'/val)
    config['args']['image_cond_model']['args']['model_name'] = str(DINO)
    config['args']['rembg_model'] = {'name':'UnusedBackgroundRemoval','args':{}}
    config['args']['low_vram'] = True
    save(OUT/'runtime/pipeline.json',config)
    save(OUT/'PLAN.json',{'model':'official TRELLIS.2-4B','source_commit': '75fbf0183001ed9876c8dbb35de6b668552ee08bd',
         'inputs':inputs,'seeds':SEEDS,'pipeline_type':'1024_cascade','samplers':'official checkpoint defaults',
         'texture_size':4096,'low_vram':True,'preprocess_image':False,
         'max_num_tokens':49152,'per_seed_GPU_limit_seconds':2400,'training_updates':0,
         'fixed_before_outputs':True,'interpretation':'pretrained image-to-3D quality trial; Map10 is DEV screenshot, not a learned map-control or geography claim',
         'export':'retain raw vertices/faces/PBR O-Voxel data and neutral GLB; bake native PBR with geometry cleanup, decimation and hole filling disabled',
         'raw_native_to_GLTF':[[1,0,0],[0,0,1],[0,-1,0]],'human_review':'PENDING'})
    freeze = {str(JOB/name):sha(JOB/name) for name in ['trellis2_asset_trial_v3.py','trellis2_trial_owner_v3.py','trellis2_saved_mesh_render_v2.py']}
    freeze.update({x['input']:x['input_sha256'] for x in inputs})
    freeze[str(OUT/'runtime/pipeline.json')]=sha(OUT/'runtime/pipeline.json')
    freeze[str(DINO/'config.json')]=sha(DINO/'config.json')
    save(OUT/'SOURCE_FREEZE.json',freeze)
    print('FIXED_TRIAL_REGISTERED', SEEDS, flush=True)

def export_without_geometry_changes(mesh):
    """Reuse official UV/baking, omit every cleanup operation that changes geometry."""
    import o_voxel
    original = o_voxel.postprocess.to_glb
    tree = ast.parse(inspect.getsource(original))
    skipped = []
    forbidden = {'fill_holes','simplify','remove_duplicate_faces','repair_non_manifold_edges',
                 'remove_small_connected_components','unify_face_orientations'}
    class Preserve(ast.NodeTransformer):
        def visit_Expr(self, node):
            value = node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr in forbidden:
                skipped.append(value.func.attr)
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    tree=ast.fix_missing_locations(Preserve().visit(tree))
    scope=dict(original.__globals__)
    exec(compile(tree,'task_geometry_preserving_glb_export','exec'),scope)
    result=scope['to_glb'](vertices=mesh.vertices,faces=mesh.faces,attr_volume=mesh.attrs,
            coords=mesh.coords,attr_layout=mesh.layout,voxel_size=mesh.voxel_size,
            aabb=[[-.5,-.5,-.5],[.5,.5,.5]],decimation_target=int(mesh.faces.shape[0])+1,
            texture_size=4096,remesh=False,verbose=True)
    return result,skipped

def generate(seed):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    os.environ['CUDA_HOME']=str(DEST/'environment')
    os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
    os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
    sys.path.insert(0,str(REPO))
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg
    rembg.UnusedBackgroundRemoval = lambda: None  # full scene inputs, official run(preprocess_image=False)
    for path,digest in read(OUT/'SOURCE_FREEZE.json').items():
        assert sha(path)==digest,path
    plan=read(OUT/'PLAN.json')
    assert seed in SEEDS
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(OUT/'runtime'))
    pipeline.cuda()
    save(OUT/f'seed{seed}/LOAD.json',{'seed':seed,'pipeline_type':plan['pipeline_type'],
         'model_parameter_dtypes':{k:sorted({str(p.dtype) for p in m.parameters()}) for k,m in pipeline.models.items()},
         'image_encoder_dtype':str(next(pipeline.image_cond_model.model.parameters()).dtype),
         'low_vram':pipeline.low_vram,'sampler_parameters':{
             'sparse_structure':pipeline.sparse_structure_sampler_params,
             'shape':pipeline.shape_slat_sampler_params,'texture':pipeline.tex_slat_sampler_params}})
    for record in plan['inputs']:
        start=time.time()
        target=OUT/f'seed{seed}'/record['id'];target.mkdir(exist_ok=False,parents=True)
        image=Image.open(record['input']).convert('RGB')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        mesh_list,latents=pipeline.run(image,seed=seed,pipeline_type='1024_cascade',
                    preprocess_image=False,return_latent=True,max_num_tokens=49152)
        mesh=mesh_list[0]
        raw_v=mesh.vertices.detach().float().cpu().numpy().copy()
        raw_f=mesh.faces.detach().cpu().numpy().copy()
        assert len(raw_f)>0 and np.isfinite(raw_v).all()
        np.savez_compressed(target/'RAW_MESH.npz',vertices=raw_v,faces=raw_f)
        torch.save({'vertices':mesh.vertices.detach().cpu(),'faces':mesh.faces.detach().cpu(),
                    'attrs':mesh.attrs.detach().cpu(),'coords':mesh.coords.detach().cpu(),
                    'layout':mesh.layout,'voxel_size':mesh.voxel_size,
                    'shape_slat':latents[0].feats.detach().cpu(),'shape_coords':latents[0].coords.detach().cpu(),
                    'tex_slat':latents[1].feats.detach().cpu(),'resolution':latents[2]}, target/'RAW_NATIVE.pt')
        zup_to_yup=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=np.float32)
        neutral=trimesh.Trimesh(vertices=raw_v@zup_to_yup.T,faces=raw_f,process=False)
        neutral.export(target/'generated_geometry.glb')
        print('RAW_MESH_SAVED',seed,record['id'],len(raw_v),len(raw_f),flush=True)
        textured,skipped=export_without_geometry_changes(mesh)
        from scipy.spatial import cKDTree
        distance,indices=cKDTree(raw_v@zup_to_yup.T).query(textured.vertices)
        assert len(textured.faces)==len(raw_f), 'EXPORT_CHANGED_FACE_COUNT'
        assert float(distance.max())<1e-6,'EXPORT_MOVED_VERTICES'
        textured.export(target/'world.glb')
        save(target/'ASSET.json',{'input':record,'seed':seed,'seconds':time.time()-start,
            'raw_vertices':len(raw_v),'raw_faces':len(raw_f),'export_vertices':len(textured.vertices),
            'export_faces':len(textured.faces),'export_vertex_max_distance':float(distance.max()),
            'disabled_geometry_operations':skipped,'generated_geometry':True,'generated_materials':True,
            'asset_sha256':sha(target/'world.glb'),'raw_geometry_sha256':sha(target/'RAW_MESH.npz'),
            'material':'native basecolor/roughness/metallic/opacity; official opaque GLB mode',
            'max_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
            'map_to_world_claim':False,'meters_per_unit_unknown':True})
        print('TEXTURED_ASSET_READY',seed,record['id'],flush=True)
        torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.register:register()
    else:generate(args.seed)

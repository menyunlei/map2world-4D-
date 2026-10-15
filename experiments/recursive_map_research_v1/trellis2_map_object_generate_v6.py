"""Frozen native fine geometry/paint from PREVIOUSLY INSPECTED object Structure.

Each object is generated in its map-derived frame. The block is an unmodified
concatenation under registered fixed transforms, not learned global placement.
No building is replaced by a procedural mesh. No ground/roads added to asset.
"""
from trellis2_map_learning_generate_v1 import base,Trellis2ImageTo3DPipeline,rembg,export_native
from trellis2_map_geometry_tokens_v3_generate import AddressedNoise
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from paths import JOB,read,save,sha
import argparse,random,time,numpy as np,torch,trimesh
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    gate=read(OUT/'MATERIAL_GATE.json');assert gate['actual_per_object_structure_images_inspected']
    assert gate['run_native_fine_geometry_and_texture']
    torch.set_num_threads(4);plan=read(OUT/'PLAN.json')
    ck=read(OUT/a.arm/f'seed{a.seed}'/'CHECKPOINT_000300.json');assert sha(ck['path'])==ck['sha256']
    rembg.UnusedBackgroundRemoval=lambda:None
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'));pipeline.cuda()
    branch=ObjectContractConditioner(reading_mode=a.arm).cuda().eval().requires_grad_(False)
    branch.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'])
    image=Image.open(plan['appearance_condition']).convert('RGB')
    appearance=None
    inputs=read(OUT/'REAL_OBJECT_INPUTS.json')['records'];axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
    for mode in ['correct','edit']:
        root=OUT/a.arm/f'seed{a.seed}'/'assets'/mode;root.mkdir(parents=True,exist_ok=False)
        world=trimesh.Scene();neutral=trimesh.Scene();records=[]
        for r in [r for r in inputs if r['mode']==mode]:
            j=r['object_index'];source=OUT/a.arm/f'seed{a.seed}'/'structure'/f'{mode}_object{j}'/'SS_GENERATED.npz'
            with np.load(source) as z:coords=torch.from_numpy(z['native_coords64'].copy()).int().cuda()
            d=root/f'object{j}';d.mkdir()
            if not len(coords):
                save(d/'EMPTY_STRUCTURE.json',{'object':j,'no_mesh':True,'no_retry_or_repair':True});continue
            with np.load(r['map_input']) as z:m={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
            branch.cuda()
            with torch.no_grad():features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
            fine={'cond':features,'neg_cond':torch.zeros_like(features)}
            gen_seed=953201+a.seed-953101+j*1000
            random.seed(gen_seed);np.random.seed(gen_seed);torch.manual_seed(gen_seed)
            pipeline.shape_slat_sampler=AddressedNoise(pipeline.shape_slat_sampler.sampler if isinstance(pipeline.shape_slat_sampler,AddressedNoise) else pipeline.shape_slat_sampler,gen_seed,1)
            pipeline.tex_slat_sampler=AddressedNoise(pipeline.tex_slat_sampler.sampler if isinstance(pipeline.tex_slat_sampler,AddressedNoise) else pipeline.tex_slat_sampler,gen_seed,2)
            tick=time.time()
            with torch.no_grad():
                shape=pipeline.sample_shape_slat(fine,pipeline.models['shape_slat_flow_model_1024'],coords)
                if appearance is None:appearance=pipeline.get_cond([image],1024)
                texture=pipeline.sample_tex_slat(appearance,pipeline.models['tex_slat_flow_model_1024'],shape)
                mesh=pipeline.decode_latent(shape,texture,1024)[0]
                export_native(mesh,(shape,texture,1024),d)
            # Only fixed map pose/scale. Neither predicted bounds nor targets
            # participate in these transforms; the edit reuses the original pose.
            local=trimesh.load(d/'world_basecolor.glb',force='scene',process=False)
            local_geom=list(local.geometry.values());assert len(local_geom)==1
            colored=local_geom[0]
            f=r['frame'];center=np.array([*f['center_xy'],f['center_z']],np.float32)
            T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=center@axis.T
            world.add_geometry(colored,node_name=f'generated_object{j}',geom_name=f'generated_object{j}',transform=T)
            neutral.add_geometry(trimesh.Trimesh(vertices=colored.vertices,faces=colored.faces,process=False),
                node_name=f'generated_object{j}',geom_name=f'generated_object{j}',transform=T)
            row={**r,'native_asset':str(d/'RAW_NATIVE.pt'),'native_sha256':sha(d/'RAW_NATIVE.pt'),
                'preview_glb':str(d/'world_basecolor.glb'),'vertices':len(colored.vertices),
                'faces':len(colored.faces),'seconds':time.time()-tick,'generation_seed':gen_seed,
                'structure_sha256':sha(source),'fixed_transform_glTF':T.tolist(),
                'geometry_input':'trained map adapter only','appearance_input':'fixed stylized reference, not actual Map10 facade',
                'native_standard_decode':'unchanged official decode_latent including its standard hole filling',
                'manual_repair_or_output_geometry_fitting':False}
            save(d/'GENERATION.json',row);records.append(row)
            print('GENERATED_MAPPED_OBJECT',a.arm,a.seed,mode,j,len(colored.faces),flush=True)
            del shape,texture,mesh,coords;torch.cuda.empty_cache()
        asset=root/'block_basecolor.glb';world.export(asset)
        neutral.export(root/'block_geometry.glb')
        save(root/'BLOCK_ASSET.json',{'asset':str(asset),'sha256':sha(asset),'objects':records,
             'requested_objects':4,'generated_objects':len(records),
             'placement_source':'fixed input-map object frames; NOT learned global placement',
             'background_source':'none in asset; road/ground may be shown as labeled map diagnostic only',
             'no_procedural_building_geometry':True,'no_geometry_repair':True,
             'full_PBR':'per-object native RAW_NATIVE.pt; GLB is basecolor/alpha vertex preview',
             'model_generated_geometry_and_appearance':True,'quality_acceptance':'PENDING actual independent render'})
    save(OUT/a.arm/f'seed{a.seed}'/'GENERATION_COMPLETE.json',{'correct_and_fixed_pose_edit_generated':True,'no_more_seeds_or_training':True})

if __name__=='__main__':main()

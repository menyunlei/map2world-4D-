"""Official texture prior on saved model-generated shapes, geometry preserved exactly."""
from paths import JOB,read,save,sha
from trellis2_street_v15_prepare import OUT,SEEDS
from trellis2_map_geometry_tokens_v3_generate import AddressedNoise
import argparse,gc,numpy as np,torch,trimesh
from PIL import Image

AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)

def run(seed):
    from trellis2.pipelines import Trellis2ImageTo3DPipeline,rembg
    from trellis2.modules.sparse import SparseTensor
    from trellis2.representations import MeshWithVoxel
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    p=read(OUT/'PLAN.json');assert sha(p['appearance_condition'])==p['appearance_sha256']
    root=OUT/f'seed{seed}/correct';assert read(root/'LAYOUT.json')['generated_count']==4
    rembg.UnusedBackgroundRemoval=lambda:None
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'))
    pipe.cuda();image=Image.open(p['appearance_condition']).convert('RGB')
    with torch.no_grad():appearance=pipe.get_cond([image],1024)
    save(root/'PAINT_CONTRACT.json',{'appearance_reference':p['appearance_condition'],
        'appearance_sha256':p['appearance_sha256'],'reference_not_verified_Map10_facade':True,
        'resolution':1024,'texture_sampler':pipe.tex_slat_sampler_params,'texture_noise_base':990201,
        'shape_and_mesh_geometry_fixed':True,'fill_holes_or_repair':False,
        'weights_dtype':{k:sorted({str(v.dtype) for v in m.parameters()}) for k,m in pipe.models.items()},
        'basecolor_preview_only_full_PBR_saved':True})
    world=trimesh.Scene();rows=[]
    for i,r in enumerate(p['records']):
        d=root/f'object{i}';old=read(d/'ASSET.json');assert sha(d/'geometry.glb')==old['sha256']
        with np.load(d/'SHAPE_SLAT.npz') as z:
            shape=SparseTensor(feats=torch.from_numpy(z['feats'].copy()).float().cuda(),
                               coords=torch.from_numpy(z['coords'].copy()).int().cuda())
        with np.load(d/'NATIVE_MESH.npz') as z:v=z['vertices'].copy();f=z['faces'].copy()
        td=d/'appearance';td.mkdir(exist_ok=False)
        tex_seed=990201+i*1000;torch.manual_seed(tex_seed)
        sampler=pipe.tex_slat_sampler.sampler if isinstance(pipe.tex_slat_sampler,AddressedNoise) else pipe.tex_slat_sampler
        pipe.tex_slat_sampler=AddressedNoise(sampler,tex_seed,2)
        with torch.no_grad():
            texture=pipe.sample_tex_slat(appearance,pipe.models['tex_slat_flow_model_1024'],shape)
            decoded,subs=pipe.decode_shape_slat(shape,1024)
            vox=pipe.decode_tex_slat(texture,subs)[0]
            # Use original vertices/faces, bypassing decode_latent's fill_holes entirely.
            mesh=MeshWithVoxel(torch.from_numpy(v).cuda(),torch.from_numpy(f).cuda(),
                origin=[-.5,-.5,-.5],voxel_size=1/1024,coords=vox.coords[:,1:],attrs=vox.feats,
                voxel_shape=torch.Size([*vox.shape,*vox.spatial_shape]),layout=pipe.pbr_attr_layout)
            colors=[]
            for start in range(0,len(v),250000):
                a=mesh.query_attrs(mesh.vertices[start:start+250000])
                rgb=a[:,mesh.layout['base_color']].clamp(0,1)
                colors.append((rgb*255).byte().cpu().numpy())
            torch.save({'vertices':torch.from_numpy(v),'faces':torch.from_numpy(f),
                'attrs':vox.feats.cpu(),'coords':vox.coords.cpu(),'layout':pipe.pbr_attr_layout,
                'voxel_size':1/1024,'texture_feats':texture.feats.cpu(),
                'geometry_source':str(d/'NATIVE_MESH.npz'),'geometry_source_sha256':sha(d/'NATIVE_MESH.npz')},td/'RAW_PBR.pt')
        rgba=np.c_[np.concatenate(colors),np.full(len(v),255,np.uint8)]
        colored=trimesh.Trimesh(vertices=v@AXIS.T,faces=f,vertex_colors=rgba,process=False)
        asset=td/'building_basecolor.glb';trimesh.Scene(colored).export(asset)
        fresh=trimesh.load(asset,force='scene',process=False);check=list(fresh.geometry.values())[0]
        assert len(check.vertices)==len(v) and np.array_equal(check.faces,f)
        assert np.allclose(check.vertices,v@AXIS.T,atol=1e-7)
        frame=r['frame'];T=np.eye(4);T[:3,:3]*=frame['scale']
        T[:3,3]=np.array([*frame['center_xy'],frame['center_z']])@AXIS.T
        world.add_geometry(colored,node_name=r['object_id'],geom_name=r['object_id'],transform=T)
        row={'object_id':r['object_id'],'asset':str(asset),'sha256':sha(asset),'frame':frame,
             'vertices':len(v),'faces':len(f),'original_geometry_preserved':True,
             'appearance':'Official texture prior, fixed reference, generated hypotheses',
             'display':'opaque vertex basecolor; native PBR including alpha/roughness/metallic retained separately'}
        save(td/'ASSET.json',row);rows.append(row)
        print('PAINTED_MAP_BUILDING',seed,i,len(v),len(f),flush=True)
        del shape,texture,decoded,subs,vox,mesh,fresh,check;gc.collect();torch.cuda.empty_cache()
    world.metadata={'source':'AI generated building geometry and basecolor, fixed map frames',
                    'roads_and_ground':'not included in this asset','real_facade_recovery':False}
    asset=root/'street_buildings.glb';world.export(asset)
    save(root/'PAINT_COMPLETE.json',{'asset':str(asset),'sha256':sha(asset),'objects':rows,
         'no_training':True,'no_geometry_repair':True,'generated_buildings':4})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    run(p.parse_args().seed)

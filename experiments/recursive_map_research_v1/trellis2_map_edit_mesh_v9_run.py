"""Existing weights only; save SS then raw native decoded meshes, no materials."""
from trellis2_structure_learnability_v8_train import setup,load_lora_state_dict
from trellis2_map_edit_mesh_v9_prepare import OUT,V8,V6,SEEDS
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow,AddressedNoise
from trellis2_object_confidence_fix_v7 import correct_object_confidence
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_learning_train_fix1 import base
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from paths import JOB,read,save,sha
import argparse,numpy as np,torch,trimesh,time,gc

AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
def maps(plan,mode,index,correct_confidence):
    r=next(r for r in plan['maps'] if r['mode']==mode and r['index']==index)
    assert sha(r['map_input'])==r['sha256']
    with np.load(r['map_input']) as z:m={k:z[k].copy() for k in z.files}
    if correct_confidence:m=correct_object_confidence(m)
    return {k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in m.items()}

def export(mesh,shape,folder,info):
    v=mesh.vertices.detach().float().cpu().numpy();f=mesh.faces.detach().cpu().numpy()
    assert np.isfinite(v).all() and len(f)>0
    with (folder/'NATIVE_MESH.npz').open('xb') as h:np.savez_compressed(h,vertices=v,faces=f)
    with (folder/'SHAPE_SLAT.npz').open('xb') as h:np.savez_compressed(h,coords=shape.coords.cpu().numpy(),feats=shape.feats.float().cpu().numpy())
    obj=trimesh.Trimesh(vertices=v@AXIS.T,faces=f,process=False)
    trimesh.Scene(obj).export(folder/'geometry.glb')
    # Exact parser reload is checked again in a separate rendering process.
    fresh=trimesh.load(folder/'geometry.glb',force='scene',process=False)
    g=list(fresh.geometry.values())[0];assert len(g.faces)==len(f) and np.allclose(g.vertices,v@AXIS.T,atol=1e-7)
    save(folder/'ASSET.json',{**info,'mesh':str(folder/'geometry.glb'),'sha256':sha(folder/'geometry.glb'),
      'vertices':len(v),'faces':len(f),'native_bounds':[v.min(0).tolist(),v.max(0).tolist()],
      'watertight':bool(obj.is_watertight),'winding_consistent':bool(obj.is_winding_consistent),
      'has_material_generation':False,'neutral_display_only':True,'mesh_repair':False,
      'decoder':'original decode_shape_slat, before optional pipeline fill_holes','canonical_up':'Z','glTF_up':'Y'})
    print('ACTUAL_MESH_SAVED',folder,len(v),len(f),flush=True)

def run(seed):
    plan=read(OUT/'PLAN.json');root=OUT/f'seed{seed}';root.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    old,native,branch,reader,_=setup('structure_lora',seed)
    ck=plan['SS_checkpoints'][str(seed)];assert sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=False)
    branch.load_state_dict(state['branch_state']);load_lora_state_dict(native,state['lora_state']);del state
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False)
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=old['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(plan['structure_generation_seed'])).cuda()
    from trellis2_structure_learnability_v8_eval import tile,iou
    tiles=[]
    for mode in ['correct','edit']:
        for i,r in enumerate(plan['records']):
            d=root/mode/f'object{i}';d.mkdir(parents=True)
            if mode=='correct':
                src=V8/'small/structure_lora'/f'seed{seed}/eval_001000/noise970201'/f'object{i:03d}/SS_GENERATED.npz'
                with np.load(src) as z:latent=torch.from_numpy(z['latent'].copy());support=z['native_support64'].copy()
                info={'reused_original':str(src),'reused_sha256':sha(src)}
            else:
                m=maps(plan,mode,i,True);wrapper.mapbatch=m
                with torch.no_grad():
                    features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
                    latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'v8_existing_map_only'},
                      neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**cfg['params'],verbose=False).samples
                    support=(decoder(latent)>0)[0,0].cpu().numpy()
                info={'new_map_intervention':True,'pure_noise':True}
            with (d/'SS_GENERATED.npz').open('xb') as h:np.savez_compressed(h,latent=latent.float().cpu().numpy(),native_support64=support)
            fp=maps(plan,mode,i,True)['footprints'][0,r['target_object']].cpu().numpy()>.5
            with np.load(r['native_target']) as z:target=z['support64'][0,0].astype(bool)
            row={**info,'map_projection_IoU':iou(support.any(2).T,fp),'target_support_IoU_diagnostic':iou(support,target),
              'cells':int(support.sum()),'training_seed':seed,'mode':mode,'object_index':i,'frame':r['frame']}
            save(d/'SS_METRICS.json',row)
            t=tile(f'{seed} object{i} {mode} XY={row["map_projection_IoU"]:.3f}',support,target,fp);t.save(d/'STRUCTURE.png');tiles.append(t)
            print('STRUCTURE_READY',seed,mode,i,row['cells'],flush=True)
    from PIL import Image
    contact=Image.new('RGB',(1536,564),'black')
    for k,t in enumerate(tiles):contact.paste(t,((k%2)*768,(k//2)*282))
    contact.save(root/'STRUCTURE_EDIT.png')
    del wrapper,native,branch,reader,decoder,noise,latent;gc.collect();torch.cuda.empty_cache()
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    parent=plan['Shape_condition_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    fine.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'])
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    scfg=cfg['shape_slat_sampler']
    pipe=Trellis2ImageTo3DPipeline(models=mods,shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**scfg['args']),plan['shape_address_seed'],1),
      shape_slat_sampler_params=scfg['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True)
    pipe.cuda()
    save(root/'PIPELINE_CONTRACT.json',{'shape_condition_checkpoint':parent,'SS_checkpoint':ck,
      'v8_head_not_used_for_untrained_Shape_interface':True,'shape_map_contract':'original v6 raw confidence',
      'precision':{k:sorted({str(p.dtype) for p in v.parameters()}) for k,v in mods.items()},
      'sampler':scfg,'resolution':1024,'image_encoder_or_material_model_loaded':False})
    # Original/edit first, so the first actual asset is available immediately.
    for mode in ['correct','edit','oracleQ_diagnostic','codec_reference_diagnostic']:
        for i,r in enumerate(plan['records']):
            if mode=='codec_reference_diagnostic' and seed!=SEEDS[0]:continue
            d=root/mode/f'object{i}';d.mkdir(parents=True,exist_ok=mode in ['correct','edit'])
            tick=time.time()
            with torch.no_grad():
                if mode=='codec_reference_diagnostic':
                    with np.load(r['native_target']) as z:shape=SparseTensor(feats=torch.from_numpy(z['feats'].copy()).float().cuda(),coords=torch.from_numpy(z['coords'].copy()).int().cuda())
                else:
                    if mode=='oracleQ_diagnostic':
                        with np.load(r['native_target']) as z:support=z['support64'][0,0].astype(bool)
                    else:
                        with np.load(d/'SS_GENERATED.npz') as z:support=z['native_support64']
                    if not support.any():save(d/'EMPTY.json',{'no_repair_or_retry':True});continue
                    coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda()
                    m=maps(plan,'edit' if mode=='edit' else 'correct',i,False)
                    features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
                    torch.manual_seed(plan['shape_address_seed'])
                    shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
                meshes,subs=pipe.decode_shape_slat(shape,1024)
                export(meshes[0],shape,d,{'mode':mode,'object_index':i,'training_seed':seed,'frame':r['frame'],
                  'diagnostic_only':mode.endswith('diagnostic'),'reference_not_in_normal_inference':True,'seconds':time.time()-tick})
                del meshes,subs,shape
            gc.collect();torch.cuda.empty_cache()
    for mode in ['correct','edit']:
        scene=trimesh.Scene()
        for i,r in enumerate(plan['records']):
            p=root/mode/f'object{i}/geometry.glb'
            if not p.exists():continue
            obj=trimesh.load(p,force='scene',process=False);g=list(obj.geometry.values())[0]
            f=r['frame'];T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=np.array([*f['center_xy'],f['center_z']])@AXIS.T
            scene.add_geometry(g,geom_name=f'object{i}',node_name=f'object{i}',transform=T)
        scene.export(root/mode/'two_object_world.glb')
    save(root/'COMPLETE.json',{'training':False,'two_objects':True,'textures':False,'history_preserved':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);run(p.parse_args().seed)

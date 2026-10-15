"""Fixed paired edit inference; weights, noise, spatial inputs and fine chain fixed."""
from trellis2_map_edit_mesh_v9_run import maps,export,AXIS
from trellis2_map_edit_mesh_v9_prepare import OUT as V9,V8,SEEDS
from trellis2_edit_identity_v10_prepare import OUT
from trellis2_edit_identity_anchor_v10 import anchor_original_descriptor
from trellis2_structure_learnability_v8_train import setup,load_lora_state_dict
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow,AddressedNoise
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_learning_train_fix1 import base
from trellis2 import models
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from paths import JOB,read,save,sha
import argparse,numpy as np,torch,trimesh,gc,time

def run(seed):
    p=read(OUT/'PLAN.json');root=OUT/f'seed{seed}';root.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    old,native,branch,reader,_=setup('structure_lora',seed)
    ck=p['SS_checkpoints'][str(seed)];assert sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=False);branch.load_state_dict(state['branch_state']);load_lora_state_dict(native,state['lora_state']);del state
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False);wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=old['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(p['structure_generation_seed'])).cuda()
    from trellis2_structure_learnability_v8_eval import tile,iou
    for i,r in enumerate(p['records']):
        d=root/'edit'/f'object{i}';d.mkdir(parents=True)
        old_map=maps(p,'correct',i,True);edited=maps(p,'edit',i,True);wrapper.mapbatch=edited
        with torch.no_grad(),anchor_original_descriptor(branch,old_map):
            features=branch(branch.prepare(edited,native_world_grid(device='cuda'))).features
            latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'SS_original_map_descriptor_edited_spatial_map'},
              neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**cfg['params'],verbose=False).samples
            support=(decoder(latent)>0)[0,0].cpu().numpy()
        with (d/'SS_GENERATED.npz').open('xb') as h:np.savez_compressed(h,latent=latent.float().cpu().numpy(),native_support64=support)
        fp=edited['footprints'][0,r['target_object']].cpu().numpy()>.5
        # Targets only for retained diagnostic display, after generation finished.
        with np.load(r['native_target']) as z:gt=z['support64'][0,0].astype(bool)
        tile(f'v10 seed{seed} object{i} edit',support,gt,fp).save(d/'STRUCTURE.png')
        save(d/'SS_METRICS.json',{'cells':int(support.sum()),'map_projection_IoU':iou(support.any(2).T,fp),
          'zero_training':True,'reference_Q_or_mesh_used_by_generation':False,'original_map_descriptor_only':True})
        print('ANCHORED_SS_READY',seed,i,int(support.sum()),flush=True)
    del wrapper,native,branch,reader,decoder,noise,latent;gc.collect();torch.cuda.empty_cache()
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    parent=p['Shape_condition_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    fine.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'])
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    s=cfg['shape_slat_sampler']
    pipe=Trellis2ImageTo3DPipeline(models=mods,shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**s['args']),p['shape_address_seed'],1),
      shape_slat_sampler_params=s['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True);pipe.cuda()
    for i,r in enumerate(p['records']):
        d=root/'edit'/f'object{i}'
        with np.load(d/'SS_GENERATED.npz') as z:support=z['native_support64'].copy()
        if not support.any():save(d/'EMPTY.json',{'no_repair_or_retry':True});continue
        coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda().contiguous()
        m=maps(p,'edit',i,False);tick=time.time()
        with torch.no_grad():
            features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
            torch.manual_seed(p['shape_address_seed'])
            shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
            meshes,subs=pipe.decode_shape_slat(shape,1024)
            export(meshes[0],shape,d,{'mode':'edit','intervention':'SS descriptor anchor only','object_index':i,'training_seed':seed,'frame':r['frame'],
              'diagnostic_only':False,'reference_not_in_normal_inference':True,'seconds':time.time()-tick,'fine_shape_unchanged':True})
            del meshes,subs,shape
        gc.collect();torch.cuda.empty_cache()
    scene=trimesh.Scene()
    for i,r in enumerate(p['records']):
        path=root/'edit'/f'object{i}/geometry.glb'
        if not path.exists():continue
        obj=trimesh.load(path,force='scene',process=False);g=list(obj.geometry.values())[0]
        f=r['frame'];T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=np.array([*f['center_xy'],f['center_z']])@AXIS.T
        scene.add_geometry(g,node_name=f'object{i}',geom_name=f'object{i}',transform=T)
    scene.export(root/'edit/two_object_world.glb')
    save(root/'COMPLETE.json',{'new_training':False,'new_meshes':2,'no_material_generation':True,'no_output_repair':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=SEEDS,required=True);run(a.parse_args().seed)

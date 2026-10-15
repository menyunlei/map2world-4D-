"""Existing v9 weights/contract, pure-noise generation for every fixed Map10 object."""
from trellis2_map_edit_mesh_v9_run import maps,export,AXIS
from trellis2_map10_block_v11_prepare import OUT,SEEDS
from trellis2_structure_learnability_v8_train import setup,load_lora_state_dict
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow,AddressedNoise
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_learning_train_fix1 import base
from trellis2 import models
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from paths import JOB,read,save,sha
import argparse,numpy as np,torch,trimesh,gc,time,traceback

def run(seed):
    p=read(OUT/'PLAN.json');root=OUT/f'seed{seed}';root.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    old,native,branch,reader,_=setup('structure_lora',seed)
    ck=p['SS_checkpoints'][str(seed)];assert sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=False)
    branch.load_state_dict(state['branch_state']);load_lora_state_dict(native,state['lora_state']);del state
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False);wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=old['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(p['structure_generation_seed'])).cuda()
    for mode in ['correct','edit']:
        for i,r in enumerate(p['records']):
            d=root/mode/f'object{i}';d.mkdir(parents=True)
            try:
                m=maps(p,mode,i,True);wrapper.mapbatch=m
                with torch.no_grad():
                    features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
                    latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'v9_map_only'},neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**cfg['params'],verbose=False).samples
                    support=(decoder(latent)>0)[0,0].cpu().numpy()
                with (d/'SS_GENERATED.npz').open('xb') as h:np.savez_compressed(h,latent=latent.float().cpu().numpy(),native_support64=support)
                save(d/'SS_METRICS.json',{'cells':int(support.sum()),'pure_noise':True,'no_reference_geometry':True})
                print('MAP10_SS_READY',seed,mode,i,int(support.sum()),flush=True)
            except Exception as e:
                save(d/'FAILURE.json',{'stage':'Structure','error':str(e),'trace':traceback.format_exc(),'slot_preserved':True});torch.cuda.empty_cache()
    del wrapper,native,branch,reader,decoder,noise;gc.collect();torch.cuda.empty_cache()
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    parent=p['Shape_condition_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    fine.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'])
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    s=cfg['shape_slat_sampler'];pipe=Trellis2ImageTo3DPipeline(models=mods,shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**s['args']),p['shape_address_seed'],1),shape_slat_sampler_params=s['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True);pipe.cuda()
    for mode in ['correct','edit']:
        entries=[];scene=trimesh.Scene()
        for i,r in enumerate(p['records']):
            d=root/mode/f'object{i}';entry={'object_index':i,'object_id':r['object_id'],'frame':r['frame']}
            if not (d/'SS_GENERATED.npz').exists():entry['status']='STRUCTURE_FAILED'
            else:
                with np.load(d/'SS_GENERATED.npz') as z:support=z['native_support64'].copy()
                if not support.any():entry['status']='EMPTY_STRUCTURE'
                else:
                    try:
                        coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda().contiguous()
                        m=maps(p,mode,i,False);tick=time.time()
                        with torch.no_grad():
                            features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
                            torch.manual_seed(p['shape_address_seed'])
                            shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
                            meshes,subs=pipe.decode_shape_slat(shape,1024)
                            export(meshes[0],shape,d,{'mode':mode,'object_index':i,'object_id':r['object_id'],'training_seed':seed,'frame':r['frame'],'seconds':time.time()-tick,'v9_contract':True,'real_map_development_only':True})
                            del meshes,subs,shape
                        entry['status']='MESH_GENERATED'
                    except Exception as e:
                        entry['status']='SHAPE_FAILED';save(d/'FAILURE.json',{'stage':'Shape','error':str(e),'trace':traceback.format_exc(),'slot_preserved':True})
                    gc.collect();torch.cuda.empty_cache()
            f=r['frame'];T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=np.array([*f['center_xy'],f['center_z']])@AXIS.T
            if entry['status']=='MESH_GENERATED':
                sc=trimesh.load(d/'geometry.glb',force='scene',process=False);g=list(sc.geometry.values())[0]
                scene.add_geometry(g,node_name=r['object_id'],geom_name=r['object_id'],transform=T)
            else:scene.graph.update(frame_to=r['object_id']+'__'+entry['status'],matrix=T,metadata={'failure':entry['status'],'no_substitute_geometry':True})
            entries.append(entry)
        scene.metadata={'generated_geometry_only':True,'object_slots':entries,'background':'none','placement':'fixed input-map transforms, not learned'}
        scene.export(root/mode/'layout.glb')
        save(root/mode/'LAYOUT.json',{'objects':entries,'expected_count':4,'generated_count':sum(e['status']=='MESH_GENERATED' for e in entries),'asset':str(root/mode/'layout.glb'),'sha256':sha(root/mode/'layout.glb'),'all_object_slots_preserved':True})
    save(root/'COMPLETE.json',{'training_updates':0,'expected_objects':8,'no_material_or_repair':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=SEEDS,required=True);run(a.parse_args().seed)

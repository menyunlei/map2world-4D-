"""One object per GPU; same Map10 input/noise, repaired SS only, Shape frozen."""
from trellis2_stage_v16 import OUT,V15,SEEDS
from trellis2_map_edit_mesh_v9_run import maps,export,AXIS
from trellis2_structure_learnability_v8_train import setup,load_lora_state_dict
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow,AddressedNoise
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_learning_train_fix1 import base
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import argparse,torch,numpy as np,gc,time

def run(seed,index):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    p=read(V15/'PLAN.json');r=p['records'][index];d=OUT/'generated'/f'seed{seed}'/f'object{index}'
    d.mkdir(parents=True,exist_ok=False);start=time.time()
    old,native,branch,reader,_=setup('structure_lora',seed)
    ck=read(OUT/'repair'/f'seed{seed}/CHECKPOINT_000100.json');assert sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=False)
    branch.load_state_dict(state['branch_state'],strict=True);load_lora_state_dict(native,state['lora_state']);del state
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False);wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    sampler_cfg=old['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**sampler_cfg['args'])
    m=maps(p,'correct',index,True);wrapper.mapbatch=m
    noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(p['structure_generation_seed'])).cuda()
    with torch.no_grad():
        features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
        latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'v16_same_map_contract'},
            neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**sampler_cfg['params'],verbose=False).samples
        support=(decoder(latent)>0)[0,0].cpu().numpy()
    with (d/'SS_GENERATED.npz').open('xb') as f:np.savez_compressed(f,latent=latent.float().cpu().numpy(),native_support64=support)
    fp=m['footprints'][0,r['target_object']].cpu().numpy()>.5
    xy=support.any(2).T
    save(d/'SS_METRICS.json',{'cells':int(support.sum()),'pure_noise':True,'native_target_read':False,
        'map_projection_IoU':float((xy&fp).sum()/max(1,(xy|fp).sum())),
        'required_projection_coverage':float((xy&fp).sum()/max(1,fp.sum())),
        'outside_footprint_fraction':float((xy&~fp).sum()/max(1,xy.sum())),
        'repair_checkpoint':ck,'baseline_support':str(V15/f'seed{seed}/correct/object{index}/SS_GENERATED.npz')})
    del wrapper,native,branch,reader,decoder,noise,latent,m,features;gc.collect();torch.cuda.empty_cache()
    if not support.any():
        save(d/'EMPTY.json',{'stage':'Structure','no_substitute':True,'no_retry':True});return
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    parent=p['Shape_condition_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    fine.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'],strict=True)
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    scfg=cfg['shape_slat_sampler']
    pipe=Trellis2ImageTo3DPipeline(models=mods,shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**scfg['args']),p['shape_address_seed'],1),
        shape_slat_sampler_params=scfg['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True)
    pipe.cuda();m=maps(p,'correct',index,False)
    coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda().contiguous()
    with torch.no_grad():
        features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
        torch.manual_seed(p['shape_address_seed'])
        shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
        meshes,subs=pipe.decode_shape_slat(shape,1024)
        export(meshes[0],shape,d,{'mode':'correct','object_index':index,'object_id':r['object_id'],
            'training_seed':seed,'frame':r['frame'],'seconds':time.time()-start,
            'repair_checkpoint':ck,'Shape_checkpoint_unchanged':parent,'generation_only_local_geometry':True,
            'placement_not_in_model_output':'original fixed map frame, applied only in layout export'})
    save(d/'COMPLETE.json',{'seed':seed,'index':index,'geometry_and_placement_separate':True,'no_materials':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--index',type=int,choices=range(4),required=True)
    a=p.parse_args();run(a.seed,a.index)

"""Matched four-card SS test of one coordinate-validity correction, no optimizer."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_address_validity_v5 import AddressValidMapConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import argparse,numpy as np,torch
from PIL import Image
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5'
OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','address_valid'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);native=base.native_model()
    cls=Trellis2MapGeometryTokenConditioner if a.arm=='baseline' else AddressValidMapConditioner
    branch=cls().cuda().eval().requires_grad_(False)
    ckpt=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    branch.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['branch_state'])
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    config=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']['sparse_structure_sampler']
    sampler=FlowEulerGuidanceIntervalSampler(**config['args']);rows=[]
    pairs=read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records'];cases=[]
    for sid in ['graph_0111_a0','graph_0112_a0']:
        r=next(r for r in pairs if r['scene_id']==sid);cases.append((sid,r['map_input'],r))
    cases.extend([(name,str(OLD/'real_input'/filename),None) for name,filename in
        [('correct','map.npz'),('edit','explicit_remove_last_building_map.npz')]])
    for name,path,r in cases:
        directory=OUT/a.arm/f'seed{a.seed}'/'support'/name;directory.mkdir(parents=True,exist_ok=False)
        with np.load(path) as f:m={k:torch.from_numpy(f[k].copy()).unsqueeze(0).cuda() for k in f.files}
        wrapper.mapbatch=m
        with torch.no_grad():
            ctx=branch.prepare(m,native_world_grid(device='cuda'));features=branch(ctx,arm='separated').features
            cond={'features':features,'mapped':True,'source':a.arm}
            neg={'features':torch.zeros_like(features),'mapped':False,'source':'unconditional_zeros'}
            torch.manual_seed(953201+a.seed-953101);noise=torch.randn((1,8,16,16,16),device='cuda')
            latent=sampler.sample(wrapper,noise,cond=cond,neg_cond=neg,**config['params'],verbose=False).samples
            support=(decoder(latent)>0).cpu().numpy();coords=np.argwhere(support)[:,[0,2,3,4]].astype(np.int32)
        with (directory/'SS_GENERATED.npz').open('xb') as f:np.savez_compressed(f,latent=latent.float().cpu().numpy(),native_support64=support,native_coords64=coords)
        active=support[0,0];projected=active[:,:,6:].any(-1).T
        with np.load(path) as f:
            valid=f['object_valid'];union=f['footprints'][valid].max(0)>0;roads=f['roads'][0]
        row={'case':name,'native_support_cells':len(coords),'floor_layer4_fraction':float(active[:,:,4].mean()),
            'proxy_projection_IoU_DIAGNOSTIC':float((projected&union).sum()/max(1,(projected|union).sum())),
            'road_above_floor_support_fraction_DIAGNOSTIC':float(projected[roads>.5].mean()) if (roads>.5).any() else None,
            'geometry_condition_RMS':float(features.square().mean().sqrt()),'split':r['split'] if r else 'real_map10_demo'}
        if r:
            _,target=base.clean(r);gt=target[0,0].cpu().numpy().astype(bool);pg=active[:,:,6:];tg=gt[:,:,6:]
            row['non_ground_surface_IoU']=float((pg&tg).sum()/max(1,(pg|tg).sum()))
        colors=np.zeros((64,64,3),np.uint8);colors[projected]=150;colors[union]=[230,75,80];colors[projected&union]=[240,210,120]
        Image.fromarray(colors[::-1]).resize((512,512),Image.Resampling.NEAREST).save(directory/'SUPPORT_OVERLAY.png')
        rows.append(row);print('ADDRESS_VALIDITY_FIXED_WEIGHT',a.arm,a.seed,row,flush=True)
    save(OUT/a.arm/f'seed{a.seed}'/'PROBE_COMPLETE.json',{'rows':rows,'checkpoint_sha256':sha(ckpt),'optimizer_updates':0,
        'no_normalization_change':True,'map_inputs_known_masks_and_parameters_unchanged':True})
if __name__=='__main__':main()

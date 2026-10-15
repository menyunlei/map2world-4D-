"""Inspect per-object native Structure before allocating any texture models."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_object_module_v1 import native_world_grid
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import argparse,numpy as np,torch
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    torch.set_num_threads(4);native=base.native_model()
    branch=ObjectContractConditioner(reading_mode=a.arm).cuda().eval().requires_grad_(False)
    ck=read(OUT/a.arm/f'seed{a.seed}'/'CHECKPOINT_000300.json');assert sha(ck['path'])==ck['sha256']
    branch.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'])
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']['sparse_structure_sampler']
    sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    cases=[];seen=set()
    for r in read(OUT/'PAIRED_NATIVE_MANIFEST.json')['records']:
        if r['split']=='adapter_HELDOUT' and r['family_id'] not in seen:
            seen.add(r['family_id']);cases.append((r['family_id'],r['map_input'],r,len(cases)+10))
    for r in read(OUT/'REAL_OBJECT_INPUTS.json')['records']:
        cases.append((r['mode']+'_object'+str(r['object_index']),r['map_input'],None,r['object_index']))
    rows=[];tiles=[]
    for name,path,record,index in cases:
        d=OUT/a.arm/f'seed{a.seed}'/'structure'/name;d.mkdir(parents=True,exist_ok=False)
        with np.load(path) as z:m={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
        wrapper.mapbatch=m
        with torch.no_grad():
            ctx=branch.prepare(m,native_world_grid(device='cuda'));features=branch(ctx).features
            cond={'features':features,'mapped':True,'source':'trained_'+a.arm}
            neg={'features':torch.zeros_like(features),'mapped':False,'source':'unconditional_zeros'}
            gen_seed=953201+a.seed-953101+index*1000
            # Exact native pipeline policy: CPU noise then transfer. Same objects
            # and edited objects use the same complete latent tensor.
            noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(gen_seed)).cuda()
            latent=sampler.sample(wrapper,noise,cond=cond,neg_cond=neg,**cfg['params'],verbose=False).samples
            support=(decoder(latent)>0).cpu().numpy()
        coords=np.argwhere(support)[:,[0,2,3,4]].astype(np.int32)
        with (d/'SS_GENERATED.npz').open('xb') as f:np.savez_compressed(f,latent=latent.float().cpu().numpy(),native_support64=support,native_coords64=coords)
        active=support[0,0];xy=active.any(2).T;xz=active.any(1).T;yz=active.any(0).T
        target_idx=int(m['target_object']);fp=m['footprints'][0,target_idx].cpu().numpy()>.5
        overlay=np.zeros((64,64,3),np.uint8);overlay[xy]=[145,145,145];overlay[fp]=[230,75,80];overlay[xy&fp]=[240,210,100]
        row={'case':name,'generation_seed':gen_seed,'model_support_cells':len(coords),
            'target_projection_IoU_DIAGNOSTIC':float((xy&fp).sum()/max(1,(xy|fp).sum())),
            'outside_target_projection_fraction':float((xy&~fp).sum()/max(1,xy.sum())),
            'native_z_min':float((coords[:,3].min()+.5)/64-.5) if len(coords) else None,
            'native_z_max':float((coords[:,3].max()+.5)/64-.5) if len(coords) else None,
            'split':'heldout_layout' if record else 'real_map10',
            'Structure_is_separate_object_output':True,'geometry_image_condition':False}
        if record:
            with np.load(record['native_target']) as z:gt=z['support64'][0,0].astype(bool)
            row['native_support_IoU']=float((active&gt).sum()/max(1,(active|gt).sum()))
            row['family_id']=record['family_id'];row['object_id']=record['object_id']
        tile=Image.new('RGB',(768,280),'#15202b');draw=ImageDraw.Draw(tile)
        draw.text((4,4),name+' '+str(len(coords))+' cells; IoUxy '+str(round(row['target_projection_IoU_DIAGNOSTIC'],3)),fill='white')
        for j,im in enumerate([overlay,np.repeat(xz[:,:,None],3,2)*220,np.repeat(yz[:,:,None],3,2)*220]):
            tile.paste(Image.fromarray(im[::-1].astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST),(j*256,24))
        tile.save(d/'STRUCTURE_VIEWS.png');tiles.append(tile);rows.append(row)
        save(d/'STRUCTURE_METRICS.json',row);print('PER_OBJECT_STRUCTURE',a.arm,a.seed,row,flush=True)
    realtiles=tiles[-8:];sheet=Image.new('RGB',(1536,1120),'black')
    for i,t in enumerate(realtiles):sheet.paste(t,((i%2)*768,(i//2)*280))
    root=OUT/a.arm/f'seed{a.seed}';sheet.save(root/'MAP10_STRUCTURE_CONTACT.png')
    save(root/'STRUCTURE_COMPLETE.json',{'rows':rows,'checkpoint':ck,'material_models_loaded':False,
        'coarse_geometry_inspection_required_before_material':True})

if __name__=='__main__':main()

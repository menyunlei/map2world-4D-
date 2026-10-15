"""Fixed complete native SS sampling from CPU Gaussian noise, never target latents."""
from trellis2_structure_learnability_v8_train import setup,load_record,load_lora_state_dict
from trellis2_structure_learnability_v8_prepare import OUT,OLD,ARMS,SEEDS
from trellis2_map_learning_train_fix1 import base
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_object_module_v1 import native_world_grid
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import torch,numpy as np,argparse
from PIL import Image,ImageDraw

def iou(a,b):return float((a&b).sum()/max(1,(a|b).sum()))

def tile(name,support,gt,fp):
    xy=support.any(2).T; overlay=np.zeros((64,64,3),np.uint8)
    overlay[xy]=[130,145,155];overlay[fp]=[205,60,65];overlay[xy&fp]=[220,195,85]
    im=Image.new('RGB',(768,282),'#101720');d=ImageDraw.Draw(im);d.text((4,4),name,fill='white')
    panels=[overlay,np.stack([support.any(1).T*190,gt.any(1).T*140,support.any(1).T*90],-1),
        np.stack([support.any(0).T*190,gt.any(0).T*140,support.any(0).T*90],-1)]
    for j,p in enumerate(panels):im.paste(Image.fromarray(p[::-1].astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST),(j*256,26))
    return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--phase',choices=['small','large'],default='small');p.add_argument('--update',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    c=setup(a.arm,a.seed);plan,native,branch,reader,params=c
    if a.update:
        ck=read(OUT/a.phase/a.arm/f'seed{a.seed}'/'train'/f'CHECKPOINT_{a.update:06d}.json');assert sha(ck['path'])==ck['sha256']
        state=torch.load(ck['path'],map_location='cpu',weights_only=False);branch.load_state_dict(state['branch_state'],strict=True)
        load_lora_state_dict(native,state['lora_state'])
    else:ck=plan['parent_checkpoints'][str(a.seed)]
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False)
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=plan['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    records=plan['selected_records'] if a.phase=='small' else [r for r in read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records'] if r['split']=='adapter_HELDOUT']
    root=OUT/a.phase/a.arm/f'seed{a.seed}'/f'eval_{a.update:06d}';root.mkdir(parents=True,exist_ok=False)
    rows=[];supports={};targets={};footprints={};sheets={}
    for gen_seed in plan['generation_seeds']:
        tiles=[]
        # One common complete noise per case set, making output/map specificity
        # comparable; paired-map changes reuse these actual outputs, not new draws.
        noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(gen_seed)).cuda()
        for index,r in enumerate(records):
            m=load_record(r);wrapper.mapbatch=m
            with torch.no_grad():
                context=branch.prepare(m,native_world_grid(device='cuda'));features=branch(context).features
                positive={'features':features,'mapped':True,'source':'trained_map_only'}
                negative={'features':torch.zeros_like(features),'mapped':False,'source':'native_zero_negative'}
                latent=sampler.sample(wrapper,noise.clone(),cond=positive,neg_cond=negative,**cfg['params'],verbose=False).samples
                support=(decoder(latent)>0)[0,0].cpu().numpy()
            d=root/f'noise{gen_seed}'/f'object{index:03d}';d.mkdir(parents=True,exist_ok=False)
            with (d/'SS_GENERATED.npz').open('xb') as f:np.savez_compressed(f,latent=latent.float().cpu().numpy(),native_support64=support)
            with np.load(r['native_target']) as z:gt=z['support64'][0,0].astype(bool)
            fp=m['footprints'][0,int(m['target_object'])].cpu().numpy()>.5
            coords=np.argwhere(support);extent=(coords.max(0)-coords.min(0)+1).tolist() if len(coords) else [0,0,0]
            row={'index':index,'scene':r['scene_id'],'family_id':r['family_id'],'object_id':r['object_id'],
                'generation_seed':gen_seed,'mode':'correct','pure_noise':True,'target_not_consumed_by_sampling':True,
                'support_cells':int(support.sum()),'extent_cells':extent,'native_support_IoU':iou(support,gt),
                'map_projection_IoU':iou(support.any(2).T,fp),'outside_projection_fraction':float((support.any(2).T&~fp).sum()/max(1,support.any(2).sum())),
                'object_pose':r['frame'],'split':r['split'],'target_sha256':r['native_target_sha256']}
            rows.append(row);supports[index]=support;targets[index]=gt;footprints[index]=fp
            t=tile(f"{index} {r['source_asset_id']} SS={row['native_support_IoU']:.3f} XY={row['map_projection_IoU']:.3f}",support,gt,fp)
            t.save(d/'STRUCTURE_VIEWS.png');tiles.append(t);save(d/'METRICS.json',row)
            print('PURE_NOISE_OBJECT',a.phase,a.arm,a.seed,a.update,row,flush=True)
        for index,r in enumerate(records):
            other=(index+1)%len(records)
            # Same map-B sample in its canonical cube is interpreted at original
            # pose A. No generated bbox alignment, postprocessing, or target input.
            changed=supports[other]
            rows.append({'index':index,'family_id':r['family_id'],'generation_seed':gen_seed,
                'mode':'changed_paired_map','condition_record_index':other,'original_pose_unchanged':r['frame'],
                'native_support_IoU_to_changed_paired_target':iou(changed,targets[other]),
                'native_support_IoU_to_original_target':iou(changed,targets[index]),
                'changed_map_projection_IoU':iou(changed.any(2).T,footprints[other]),
                'binary_support_change_fraction':float((changed^supports[index]).sum()/max(1,(changed|supports[index]).sum())),
                'reuses_correct_map_B_output_not_independent_sample':True})
            rows.append({'index':index,'family_id':r['family_id'],'generation_seed':gen_seed,'mode':'mismatched_paired_map',
                'condition_record_index':other,'native_support_IoU':iou(changed,targets[index]),
                'same_output_as_changed_mode':True,'not_shuffled_route_attention_causal_test':True})
        if a.phase=='small':
            sheet=Image.new('RGB',(1536,3*282),'black')
            for i,t in enumerate(tiles):sheet.paste(t,((i%2)*768,(i//2)*282))
            sheet.save(root/f'PURE_NOISE_CONTACT_{gen_seed}.png')
    save(root/'RESULTS.json',{'arm':a.arm,'seed':a.seed,'phase':a.phase,'update':a.update,'checkpoint':ck,'rows':rows,
        'native_sampler':cfg,'reference_latents_used_for_generation':False,'materials_loaded':False,
        'exact_support_metric_is_TRAIN_reconstruction_diagnostic':a.phase=='small','no_world_quality_claim':True})

if __name__=='__main__':main()

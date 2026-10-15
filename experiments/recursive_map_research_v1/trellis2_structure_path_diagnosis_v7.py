"""Small fixed-weight causal trace of the existing Structure computation.

No training, appearance images, fine geometry, test data or output repair.
Reference latents are used ONLY for labeled TRAIN/held-out recovery diagnostics.
Normal inference traces start from exactly the original native CPU noise.
"""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_object_data_v6_fix1 import OUT as OLD, ARMS, SEEDS
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import argparse,numpy as np,torch
from PIL import Image,ImageDraw

OUT=JOB/'map_object_world_v1/trellis2_structure_path_diagnosis_v7'
TAPS=[0,4,14,24,29]
MODES=['correct','reflected','tokens_only','residual_only','none']

def rms(x):return float(x.float().square().mean().sqrt())
def iou(a,b):return float((a&b).sum()/max(1,(a|b).sum()))
def mirrored(m):
    wrong={k:(v.flip(-1) if v.ndim==4 else v.clone()) for k,v in m.items()}
    wrong['object_summary'][...,0]*=-1
    return wrong

class TraceFlow(MapSSFlow):
    def __init__(self,native,branch):
        super().__init__(native,branch);self.calls=[];self.capture=False
    def forward(self,x,t,cond,**kwargs):
        pred=super().forward(x,t,cond,**kwargs)
        if self.capture:
            self.calls.append({'t':float(t[0])/1000,'positive':cond['mapped'],
                'x':x.detach().float().cpu(),'v':pred.detach().float().cpu()})
        return pred

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    native=base.native_model();branch=ObjectContractConditioner(reading_mode=a.arm).cuda().eval().requires_grad_(False)
    ck=read(OLD/a.arm/f'seed{a.seed}'/'CHECKPOINT_000300.json');assert sha(ck['path'])==ck['sha256']
    branch.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'])
    wrapper=TraceFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']['sparse_structure_sampler']
    sampler=FlowEulerGuidanceIntervalSampler(**cfg['args']);sigma=cfg['args']['sigma_min']
    directory=OUT/a.arm/f'seed{a.seed}';directory.mkdir(parents=True,exist_ok=False)
    stats={};handles=[]
    for j in TAPS:
        def capture_block(_module,_arguments,value,index=j):
            stats.setdefault(str(index),{})['hidden_RMS']=rms(value)
        def capture_cross(_module,_arguments,value,index=j):
            row=stats.setdefault(str(index),{});row['cross_RMS']=rms(value)
            row['cross_spatial_RMS']=rms(value-value.mean(1,keepdim=True))
        handles.extend([native.blocks[j].register_forward_hook(capture_block),
                        native.blocks[j].cross_attn.register_forward_hook(capture_cross)])
    records=read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records'];chosen=[];seen=set()
    for split,count in [('adapter_TRAIN',2),('adapter_HELDOUT',1)]:
        n=0
        for r in records:
            if r['split']==split and r['family_id'] not in seen:
                chosen.append(r);seen.add(r['family_id']);n+=1
                if n==count:break
    recovery=[];signals=[];consistency=[]
    with torch.no_grad():
        for idx,r in enumerate(chosen):
            m=base.load_map(r);wrong=mirrored(m)
            contexts={'correct':branch.prepare(m,native_world_grid(device='cuda')),
                      'reflected':branch.prepare(wrong,native_world_grid(device='cuda'))}
            features={k:branch(c).features for k,c in contexts.items()}
            with np.load(r['native_target']) as z:x0=torch.from_numpy(z['S0'].copy()).float().cuda();gt=z['support64'][0,0].astype(bool)
            target_idx=int(m['target_object']);fp=m['footprints'][0,target_idx].cpu().numpy()>.5
            noise=torch.randn(x0.shape,generator=torch.Generator().manual_seed(997701+idx)).cuda()
            signal={'case':r['scene_id']+':'+str(r['object_id']),'split':r['split'],
                'map_target_projection_IoU':iou(gt.any(2).T,fp),
                'native_encoder_decoder_target_IoU':iou((decoder(x0)>0)[0,0].cpu().numpy(),gt),
                'condition_RMS':rms(features['correct']),
                'condition_spatial_RMS':rms(features['correct']-features['correct'].mean(1,keepdim=True)),
                'condition_map_reflection_delta_RMS':rms(features['correct']-features['reflected']),
                'residual_RMS':rms(branch.Wo(contexts['correct'].token_features)),
                'residual_reflection_delta_RMS':rms(branch.Wo(contexts['correct'].token_features-contexts['reflected'].token_features)),
                'native_latent_RMS':rms(x0)}
            signals.append(signal)
            for tv in [1.,.9,.6,.2]:
                x=(1-tv)*x0+(sigma+(1-sigma)*tv)*noise;t=torch.tensor([tv*1000],device='cuda')
                target=(1-sigma)*noise-x0;predictions={};block_stats={}
                for mode in MODES:
                    key='reflected' if mode=='reflected' else 'correct'
                    wrapper.reader.context=contexts[key] if mode in ['correct','reflected','residual_only'] else None
                    cond=features[key] if mode in ['correct','reflected','tokens_only'] else torch.zeros_like(features[key])
                    stats.clear();v=native(x,t,cond).float();predictions[mode]=v
                    block_stats[mode]={k:dict(vv) for k,vv in stats.items()}
                    clean=(1-sigma)*x-(sigma+(1-sigma)*tv)*v
                    support=(decoder(clean)>0)[0,0].cpu().numpy()
                    recovery.append({'case':signal['case'],'split':r['split'],'t':tv,'mode':mode,
                        'flow_MSE':float((v-target).square().mean()),'clean_structure_IoU':iou(support,gt),
                        'clean_structure_cells':int(support.sum()),'clean_latent_RMS':rms(clean),
                        'block_stats':block_stats[mode]})
                recovery[-5].update({'correct_reflected_velocity_delta_RMS':rms(predictions['correct']-predictions['reflected']),
                    'correct_none_velocity_delta_RMS':rms(predictions['correct']-predictions['none']),
                    'tokens_only_none_delta_RMS':rms(predictions['tokens_only']-predictions['none']),
                    'residual_only_none_delta_RMS':rms(predictions['residual_only']-predictions['none'])})
                # Direct TRAIN-style call versus actual inference wrapper: same x/t/map.
                wrapper.mapbatch=m
                direct=predictions['correct']
                actual=wrapper(x,t,{'features':features['correct'],'mapped':True,'source':'contract_check'})
                consistency.append({'case':signal['case'],'t':tv,'train_infer_max_abs':float((direct-actual.float()).abs().max())})
            print('RECOVERY_DIAG',a.arm,a.seed,signal,flush=True)
        # Three fixed original failed outputs: do not select a nice case or noise.
        trace_rows=[];tiles=[]
        for name,index,mode in [('correct_object0',0,'correct'),('correct_object3',3,'correct'),('edit_object3',3,'edit')]:
            input_record=next(r for r in read(OLD/'REAL_OBJECT_INPUTS.json')['records'] if r['mode']==mode and r['object_index']==index)
            with np.load(input_record['map_input']) as z:m={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
            wrapper.mapbatch=m;ctx=branch.prepare(m,native_world_grid(device='cuda'));feat=branch(ctx).features
            pos={'features':feat,'mapped':True,'source':'existing_trained_map'}
            neg={'features':torch.zeros_like(feat),'mapped':False,'source':'native_unconditional'}
            gen_seed=953201+a.seed-953101+index*1000
            noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(gen_seed)).cuda()
            wrapper.capture=True;wrapper.calls=[]
            sample=sampler.sample(wrapper,noise,cond=pos,neg_cond=neg,**cfg['params'],verbose=False)
            wrapper.capture=False
            old_path=OLD/a.arm/f'seed{a.seed}'/'structure'/name/'SS_GENERATED.npz'
            with np.load(old_path) as z:old=z['latent'];old_support=z['native_support64']
            final=(decoder(sample.samples)>0).cpu().numpy()
            exact={'case':name,'old_vs_replay_max_abs':float(np.max(np.abs(old-sample.samples.cpu().numpy()))),
                   'same_support':bool(np.array_equal(final,old_support)),
                   'map_condition_RMS':rms(feat),'map_condition_spatial_RMS':rms(feat-feat.mean(1,keepdim=True)),
                   'map_residual_RMS':rms(branch.Wo(ctx.token_features))}
            consistency.append(exact);assert exact['old_vs_replay_max_abs']==0 and exact['same_support'],exact
            fp=m['footprints'][0,index].cpu().numpy()>.5
            for step,clean in enumerate(sample.pred_x_0):
                positive=[c for c in wrapper.calls if c['positive']][step];tv=positive['t']
                positive_clean=(1-sigma)*positive['x']-(sigma+(1-sigma)*tv)*positive['v']
                sp=(decoder(positive_clean.cuda())>0)[0,0].cpu().numpy()
                sg=(decoder(clean)>0)[0,0].cpu().numpy()
                row={'case':name,'step':step,'t':tv,'positive_cells':int(sp.sum()),'guided_cells':int(sg.sum()),
                    'positive_proxy_IoU_DIAGNOSTIC':iou(sp.any(2).T,fp),
                    'guided_proxy_IoU_DIAGNOSTIC':iou(sg.any(2).T,fp),
                    'guided_positive_clean_delta_RMS':rms(clean.cpu()-positive_clean),
                    'positive_clean_RMS':rms(positive_clean),'guided_clean_RMS':rms(clean)}
                trace_rows.append(row)
                with (directory/f'{name}_clean_step{step:02d}.npz').open('xb') as f:
                    np.savez_compressed(f,positive_support=sp,guided_support=sg,
                        positive_clean=positive_clean.numpy(),guided_clean=clean.cpu().numpy())
                for label,support in [('positive',sp),('guided',sg)]:
                    xy=support.any(2).T;rgb=np.zeros((64,64,3),np.uint8);rgb[xy]=150;rgb[fp]=[230,75,80];rgb[xy&fp]=[240,210,100]
                    tile=Image.new('RGB',(192,216),'#15202b');tile.paste(Image.fromarray(rgb[::-1]).resize((192,192),Image.Resampling.NEAREST),(0,24))
                    ImageDraw.Draw(tile).text((2,2),f'{name} {step} {label}',fill='white');tiles.append(tile)
            print('NATIVE_PATH_REPLAY',a.arm,a.seed,exact,flush=True)
        sheet=Image.new('RGB',(2304,1296),'black')
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%12)*192,(i//12)*216))
        sheet.save(directory/'DENOISING_CONTACT.png')
    for h in handles:h.remove()
    wrapper.reader.context=None
    save(directory/'PATH_DIAGNOSIS.json',{'checkpoint':ck,'updates':0,'signals':signals,
        'teacher_forced_diagnostics':recovery,'train_infer_and_replay_checks':consistency,
        'native_Euler_clean_traces':trace_rows,'sampler_unchanged':cfg,
        'no_model_or_normalization_or_geometry_change':True,'no_material_generation':True})

if __name__=='__main__':main()

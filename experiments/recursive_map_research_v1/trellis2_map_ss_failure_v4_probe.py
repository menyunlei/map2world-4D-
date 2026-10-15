"""Fixed-weight SS attribution, not model training or a candidate search."""
import argparse
import numpy as np
import torch
from paths import JOB, read, save, sha
from trellis2_map_learning_train_fix1 import base
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2 import models

OUT = JOB/'map_object_world_v1/trellis2_map_ss_failure_v4'
OLD = JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'

def rms(x): return float(x.float().square().mean().sqrt())

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    native=base.native_model()
    branch=Trellis2MapGeometryTokenConditioner().cuda().eval()
    ckpt=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    branch.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['branch_state'])
    reader=base.BoundMapReader(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    pairs=read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records']
    records=[next(r for r in pairs if r['scene_id']==sid) for sid in ['graph_0064_a0','graph_0065_a0','graph_0111_a0','graph_0112_a0']]
    rows=[];scale_rows=[]
    with torch.no_grad():
        for index,record in enumerate(records):
            x0,target_support=base.clean(record)
            m=base.load_map(record)
            wrong={k:(v.flip(-1,-2) if v.ndim==4 else v) for k,v in m.items()}
            contexts={name:branch.prepare(mm,native_world_grid(device='cuda')) for name,mm in [('correct',m),('shuffled',wrong)]}
            tokens={name:branch(ctx,arm='separated').features for name,ctx in contexts.items()}
            noise=torch.randn(x0.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(994000+index))
            residual=branch.Wo(contexts['correct'].token_features)
            scale_rows.append({'scene_id':record['scene_id'],'latent_RMS':rms(x0),'noise_RMS':rms(noise),
                'map_token_RMS':rms(tokens['correct']),
                'map_token_spatial_deviation_RMS':rms(tokens['correct']-tokens['correct'].mean(1,keepdim=True)),
                'map_token_shuffle_delta_RMS':rms(tokens['correct']-tokens['shuffled']),
                'residual_RMS':rms(residual),
                'residual_spatial_deviation_RMS':rms(residual-residual.mean(1,keepdim=True))})
            for tval in [1.0,.9,.6,.2]:
                t=torch.tensor([tval],device='cuda');tb=t[:,None,None,None,None]
                x=(1-tb)*x0+(1e-5+(1-1e-5)*tb)*noise
                target=(1-1e-5)*noise-x0
                predictions={}
                for mode in ['correct','shuffled','tokens_only','residual_only','none']:
                    features=tokens['shuffled' if mode=='shuffled' else 'correct']
                    reader.context=contexts['shuffled' if mode=='shuffled' else 'correct'] if mode in ['correct','shuffled','residual_only'] else None
                    if mode in ['residual_only','none']:features=torch.zeros_like(features)
                    pred=native(x,t*1000,features).float()
                    clean=(1-1e-5)*x-(1e-5+(1-1e-5)*tb)*pred
                    decoded=decoder(clean)>0
                    intersection=(decoded&target_support.bool()).sum()
                    union=(decoded|target_support.bool()).sum().clamp_min(1)
                    row={'scene_id':record['scene_id'],'family_id':record['family_id'],'split':record['split'],
                         't':tval,'mode':mode,'flow_MSE':float((pred-target).square().mean()),
                         'decoded_clean_surface_IoU':float(intersection/union),
                         'decoded_surface_cells':int(decoded.sum()),'predicted_clean_RMS':rms(clean)}
                    predictions[mode]=pred
                    rows.append(row)
                rows[-5]['correct_shuffle_velocity_delta_RMS']=rms(predictions['correct']-predictions['shuffled'])
                rows[-5]['correct_none_velocity_delta_RMS']=rms(predictions['correct']-predictions['none'])
                rows[-5]['tokens_only_none_velocity_delta_RMS']=rms(predictions['tokens_only']-predictions['none'])
                rows[-5]['residual_only_none_velocity_delta_RMS']=rms(predictions['residual_only']-predictions['none'])
                print('FIXED_WEIGHT_PROBE',a.seed,record['scene_id'],tval,rows[-5],flush=True)
    reader.context=None
    save(OUT/f'FIXED_WEIGHT_SEED{a.seed}.json',{'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),
        'optimizer_updates':0,'target_conditioned_recovery_diagnostic_not_normal_inference':True,
        'pure_noise_t1_has_no_target_in_input':True,'scale_rows':scale_rows,'rows':rows})

if __name__=='__main__':main()

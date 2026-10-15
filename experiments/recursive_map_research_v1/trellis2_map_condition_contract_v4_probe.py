"""Single frozen-weight interface intervention: native condition LayerNorm.

No optimizer; no geometry manipulation or sampler changes. This tests a
specific mismatch, not a declaration that scale explains the whole failure.
"""
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_learning_generate_v1 import base,Trellis2ImageTo3DPipeline,rembg
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid
from paths import JOB,read,save,sha
import argparse, numpy as np,torch
from torch.nn import functional as F
from PIL import Image
OUT=JOB/'map_object_world_v1/trellis2_map_condition_contract_v4'
OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','native_norm'],required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);rembg.UnusedBackgroundRemoval=lambda:None
    plan=read(OLD/'PLAN.json')
    checkpoint=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'))
    pipe.cuda()
    branch=Trellis2MapGeometryTokenConditioner().cuda().eval().requires_grad_(False)
    branch.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['branch_state'])
    wrapper=MapSSFlow(pipe.models['sparse_structure_flow_model'],branch)
    pipe.models['sparse_structure_flow_model']=wrapper
    # Source-native contract is measured, never used as geometry conditioning.
    reference=pipe.get_cond([Image.open(plan['appearance_condition']).convert('RGB')],512)['cond']
    contract={'native_condition_RMS':float(reference.square().mean().sqrt()),
              'native_condition_channel_mean_abs':float(reference.mean(-1).abs().mean())}
    del reference
    records=read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records']
    cases=[]
    for sid in ['graph_0064_a0','graph_0065_a0','graph_0111_a0','graph_0112_a0']:
        r=next(r for r in records if r['scene_id']==sid)
        cases.append((sid,r['map_input'],r))
    cases.extend([(name,str(OLD/'real_input'/filename),None) for name,filename in
        [('map10_correct','map.npz'),('map10_edit','explicit_remove_last_building_map.npz')]])
    rows=[]
    for name,path,record in cases:
        directory=OUT/a.arm/f'seed{a.seed}'/name;directory.mkdir(parents=True,exist_ok=False)
        with np.load(path) as f:m={k:torch.from_numpy(f[k].copy()).unsqueeze(0).cuda() for k in f.files}
        branch.cuda();wrapper.mapbatch=m
        with torch.no_grad():
            context=branch.prepare(m,native_world_grid(device='cuda'))
            raw=branch(context,arm='separated').features
            features=F.layer_norm(raw,raw.shape[-1:]) if a.arm=='native_norm' else raw
        condition={'cond':{'features':features,'mapped':True,'source':a.arm},
                   'neg_cond':{'features':torch.zeros_like(features),'mapped':False,'source':'unconditional_zeros'}}
        torch.manual_seed(953201+a.seed-953101)
        trace={}
        def capture(_module,inputs,output):
            trace['latent']=inputs[0].float().cpu().numpy()
            trace['support']=(output>0).cpu().numpy()
        hook=pipe.models['sparse_structure_decoder'].register_forward_hook(capture)
        with torch.no_grad():coords=pipe.sample_sparse_structure(condition,64)
        hook.remove()
        with (directory/'SS_GENERATED.npz').open('xb') as f:np.savez_compressed(f,**trace,coords=coords.cpu().numpy())
        support=trace['support'][0,0].astype(bool)
        projected=support[:,:,6:].any(-1).T
        with np.load(path) as f:
            footprints=f['footprints'];valid=f['object_valid'];roads=f['roads'][0]
        union=footprints[valid].max(0)>0
        row={'case':name,'split':record['split'] if record else 'real_map10_unregistered_demo',
             'native_support_cells':int(support.sum()),'above_ground_cells':int(support[:,:,6:].sum()),
             'canonical_floor_layer4_fraction':float(support[:,:,4].mean()),
             'map_proxy_projection_IoU_DIAGNOSTIC':float((projected&union).sum()/max(1,(projected|union).sum())),
             'road_above_ground_support_fraction_DIAGNOSTIC':float(projected[roads>.5].mean()) if (roads>.5).any() else None,
             'raw_condition_RMS':float(raw.square().mean().sqrt()),'used_condition_RMS':float(features.square().mean().sqrt())}
        if record:
            _,target=base.clean(record);target=target[0,0].cpu().numpy().astype(bool)
            for label,slice_z in [('whole',slice(None)),('non_ground',slice(6,None))]:
                pred=support[:,:,slice_z];gt=target[:,:,slice_z]
                row[label+'_surface_IoU']=float((pred&gt).sum()/max(1,(pred|gt).sum()))
                row[label+'_surface_recall']=float((pred&gt).sum()/max(1,gt.sum()))
        colors=np.zeros((64,64,3),np.uint8);colors[projected]=150;colors[union]=[230,75,80]
        colors[projected&union]=[240,210,120]
        Image.fromarray(colors[::-1]).resize((512,512),Image.Resampling.NEAREST).save(directory/'SUPPORT_OVERLAY.png')
        save(directory/'MEASUREMENTS.json',row);rows.append(row)
        print('CONDITION_CONTRACT_PROBE',a.arm,a.seed,row,flush=True)
    save(OUT/a.arm/f'seed{a.seed}'/'PROBE_COMPLETE.json',{'optimizer_updates':0,'single_intervention':'condition feature native LayerNorm',
        'checkpoint_sha256':sha(checkpoint),'samplers_unchanged':True,'native_contract':contract,'rows':rows,
        'not_final_geometry_or_quality_success':True})

if __name__=='__main__':main()

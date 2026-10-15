"""Fixed-weight coverage attribution: remove knowledge, not map objects or GT."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import JOB,read,save,sha
import argparse,numpy as np,torch
OUT=JOB/'map_object_world_v1/trellis2_map_ss_failure_v4'
OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'
CONTROL=JOB/'map_object_world_v1/trellis2_map_condition_contract_v4'

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    native=base.native_model();branch=Trellis2MapGeometryTokenConditioner().cuda().eval().requires_grad_(False)
    checkpoint=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    branch.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['branch_state'])
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval()
    config=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']['sparse_structure_sampler']
    sampler=FlowEulerGuidanceIntervalSampler(**config['args']);rows=[]
    pairs=read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records']
    for sid in ['graph_0064_a0','graph_0065_a0','graph_0111_a0','graph_0112_a0']:
        r=next(r for r in pairs if r['scene_id']==sid)
        full=base.load_map(r)
        partial={k:v.clone() for k,v in full.items()}
        partial['footprint_known']=partial['footprints'].clone()
        partial['roads_known']=partial['roads'].clone()
        partial['open_space_known']=torch.zeros_like(partial['open_space_known'])
        assert all(torch.equal(full[k],partial[k]) for k in ['footprints','roads','open_space','object_summary','object_valid'])
        wrapper.mapbatch=partial
        with torch.no_grad():
            context=branch.prepare(partial,native_world_grid(device='cuda'))
            features=branch(context,arm='separated').features
            full_tokens=branch(branch.prepare(full,native_world_grid(device='cuda')),arm='separated').features
            positive={'features':features,'mapped':True,'source':'partial_known_mask_diagnostic'}
            negative={'features':torch.zeros_like(features),'mapped':False,'source':'unconditional_zeros'}
            torch.manual_seed(953201+a.seed-953101)
            noise=torch.randn((1,8,16,16,16),device='cuda')
            latent=sampler.sample(wrapper,noise,cond=positive,neg_cond=negative,**config['params'],verbose=False).samples
            support=(decoder(latent)>0)[0,0].cpu().numpy()
        baseline_path=CONTROL/'raw'/f'seed{a.seed}'/sid/'SS_GENERATED.npz'
        with np.load(baseline_path) as f:baseline=f['support'][0,0].astype(bool)
        _,target=base.clean(r);gt=target[0,0].cpu().numpy().astype(bool)
        row={'case':sid,'family':r['family_id'],'split':r['split'],'values_and_objects_unchanged':True,
            'full_known_condition_RMS':float(full_tokens.square().mean().sqrt()),
            'partial_known_condition_RMS':float(features.square().mean().sqrt()),
            'full_support_cells':int(baseline.sum()),'partial_support_cells':int(support.sum()),
            'support_changed_fraction':float((support!=baseline).mean()),
            'partial_XY_unknown_fraction':float((context.token_features.reshape(1,16,16,16,128).abs().sum((-1,-2))==0).float().mean())}
        for name,pred in [('full_known',baseline),('partial_known',support)]:
            pg=pred[:,:,6:];tg=gt[:,:,6:]
            row[name+'_non_ground_surface_IoU']=float((pg&tg).sum()/max(1,(pg|tg).sum()))
            row[name+'_floor_layer4_fraction']=float(pred[:,:,4].mean())
        rows.append(row);print('KNOWN_COVERAGE_CAUSAL_DIAGNOSTIC',a.seed,row,flush=True)
    save(OUT/f'KNOWN_COVERAGE_SEED{a.seed}.json',{'rows':rows,'checkpoint_sha256':sha(checkpoint),
        'optimizer_updates':0,'diagnostic_only_not_a_new_intervention_or_adopted_fix':True,
        'map_values_objects_world_coordinates_noise_checkpoint_sampler_unchanged':True,
        'only_known_observation_mask_reduced':True,'training_coverage_mismatch_not_proven_sole_cause':True})

if __name__=='__main__':main()

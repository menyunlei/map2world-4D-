"""Narrow CPU attribution and frozen plan, existing two objects/two seeds only."""
from paths import JOB,read,save,sha
from trellis2_map_edit_mesh_v9_prepare import OUT as V9,V8,SEEDS
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_object_confidence_fix_v7 import correct_object_confidence
from trellis2_edit_identity_anchor_v10 import anchor_original_descriptor
import numpy as np,torch
OUT=JOB/'map_object_world_v1/trellis2_edit_identity_v10'

def cpu_map(p,mode,i):
    r=next(x for x in p['maps'] if x['mode']==mode and x['index']==i)
    assert sha(r['map_input'])==r['sha256']
    with np.load(r['map_input']) as z:m=correct_object_confidence({k:z[k].copy() for k in z.files})
    return {k:torch.from_numpy(v).unsqueeze(0) for k,v in m.items()}

def prepare():
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    assert not (OUT/'PLAN.json').exists()
    p=read(V9/'PLAN.json');rows=[]
    for seed in SEEDS:
        b=ObjectContractConditioner(reading_mode='object_grounded').eval().requires_grad_(False)
        portable=V8/'small/structure_lora'/f'seed{seed}/train/checkpoint_001000.npz'
        with np.load(portable) as z:b.load_state_dict({k.split('::',1)[1]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('branch::')})
        for i in range(2):
            old,new=cpu_map(p,'correct',i),cpu_map(p,'edit',i);j=int(old['target_object'][0]);grid=native_world_grid()
            captured=[]
            def grab(_m,args,out):captured.append(args[0].detach().clone())
            hook=b.token_mlp.register_forward_hook(grab)
            with torch.no_grad():
                a=b.prepare(old,grid);fa=b(a).features;old_input=captured[-1]
                e=b.prepare(new,grid);fe=b(e).features;new_input=captured[-1]
                with anchor_original_descriptor(b,old):
                    same=b.prepare(old,grid);fs=b(same).features
                    c=b.prepare(new,grid);fc=b(c).features;candidate_input=captured[-1]
                    neg=b(c,negative_cfg=True)
            hook.remove()
            assert torch.equal(fs,fa),'Nonidentity on unchanged map'
            assert torch.equal(candidate_input[...,128:],new_input[...,128:]),'Spatial geometry path changed'
            assert torch.equal(c.object_tokens,a.object_tokens),'Descriptor was not anchored'
            assert torch.equal(c.routing_weights,e.routing_weights),'Edited routing was not preserved'
            assert neg.context is None and torch.count_nonzero(neg.features)==0
            stable=(a.routing_weights[0,:,j]>.5)&(e.routing_weights[0,:,j]>.5)
            rms=lambda x:float(x.float().square().mean().sqrt())
            row={'seed':seed,'object_index':i,'original_summary':old['object_summary'][0,j].tolist(),
              'edited_summary':new['object_summary'][0,j].tolist(),
              'object_descriptor_RMS_change':rms(e.object_tokens-a.object_tokens),
              'retained_queries':int(stable.sum()),
              'baseline_retained_feature_RMS_change':rms((e.token_features-a.token_features)[0,stable]),
              'anchored_retained_feature_RMS_change':rms((c.token_features-a.token_features)[0,stable]),
              'global_condition_RMS_change_baseline':rms(fe-fa),'global_condition_RMS_change_anchored':rms(fc-fa),
              'unchanged_input_exact_identity':True,'edited_spatial_channels_bitwise_unchanged':True,
              'edited_routing_bitwise_unchanged':True,'original_descriptor_reused':True,'negative_CFG_zero':True}
            rows.append(row)
    assert all(r['object_descriptor_RMS_change']>0 for r in rows if r['object_index']==0)
    save(OUT/'CPU_ATTRIBUTION.json',{'rows':rows,'interpretation':'Editing footprint changes object descriptor globally as well as spatial controls. This establishes an entangled path, not proof that it alone causes height drift. One paired inference test follows.',
      'zero_training':True,'no_backbone_sampling':True})
    save(OUT/'PLAN.json',{**p,'version':'v10','sole_intervention':'Reuse original-map object_mlp output in SS only during footprint edit; keep edited geometry/coverage/relative features. Shape stage stays v9.',
      'baseline_root':str(V9),'new_samples':'2 objects x 2 training seeds, same v9 noise970201, only edited condition; original and old edited assets reused',
      'inputs_required':'original map and edited map; no original generated Q/mesh is fed into sampling',
      'no_new_training':True,'not_new_learned_identity_representation':True,'old_descriptor_contains_geometry':'May conflict with edited dimensions; paired test can fail, no guarantee of benefit.',
      'GPU_occupancy_cap_seconds':1800,
      'fixed_diagnostic_decision':{'both_seeds_required':True,'edited_map_projection_IoU_drop_max':.02,'removed_region_projection_occupancy_max':.02,
        'retained_footprint_coverage_drop_max':.02,'both_height_change_and_retained_SS_change_must_improve':True,
        'no_new_visual_failure':True,'not_historical_gate_replacement':True},
      'no_automatic_next_candidate_or_training':True})
    files=['trellis2_edit_identity_anchor_v10.py','trellis2_edit_identity_v10_prepare.py','trellis2_edit_identity_v10_run.py','trellis2_edit_identity_v10_views.py','trellis2_edit_identity_v10_owner.py','trellis2_edit_identity_v10_score.py']
    save(OUT/'SOURCE_FREEZE.json',{**{str(JOB/f):sha(JOB/f) for f in files},str(OUT/'PLAN.json'):sha(OUT/'PLAN.json')})
    print('CPU_ATTRIBUTION',rows,flush=True)

if __name__=='__main__':prepare()

"""Six registered-condition identifiability check, CPU-only; no new sampling.

Compare effective map embeddings, not metadata/file hashes that include numeric
IDs. Targets are diagnostics only. This cannot edit the running experiment.
"""
from paths import read,save,sha
from trellis2_structure_learnability_v8_prepare import OUT
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_object_confidence_fix_v7 import correct_object_confidence
import numpy as np,torch,itertools

def main():
    assert not torch.cuda.is_available(),'CPU-only diagnostic expected'
    torch.set_num_threads(4);p=read(OUT/'PLAN.json');records=p['selected_records']
    branch=ObjectContractConditioner(reading_mode='object_grounded').eval().requires_grad_(False)
    ck=p['parent_checkpoints'][str(p['training_seeds'][0])];assert sha(ck['path'])==ck['sha256']
    branch.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'],strict=True)
    conditions=[];supports=[];rows=[]
    for i,r in enumerate(records):
        with np.load(r['map_input']) as z:m=correct_object_confidence({k:z[k].copy() for k in z.files})
        target=int(m['target_object']);rows.append({'index':i,'asset':r['source_asset_id'],
            'active_objects_in_canonical_crop':int(m['object_valid'].sum()),
            'known_road_mass':float((m['roads']*m['roads_known']).sum()),
            'height_known':float(m['object_summary'][target,6]),'numeric_object_ID_not_a_learned_identity_input':True})
        mt={k:torch.from_numpy(v).unsqueeze(0) for k,v in m.items()}
        with torch.no_grad():ctx=branch.prepare(mt,native_world_grid());cond=branch(ctx)
        conditions.append((cond.features,ctx.token_features))
        with np.load(r['native_target']) as z:supports.append(z['support64'][0,0].astype(bool))
    pairs=[]
    for a,b in itertools.combinations(range(6),2):
        cf=float((conditions[a][0]-conditions[b][0]).abs().max());tf=float((conditions[a][1]-conditions[b][1]).abs().max())
        ga,gb=supports[a],supports[b]
        pairs.append({'a':a,'b':b,'condition_max_abs_delta':cf,'query_features_max_abs_delta':tf,
            'identical_effective_map_condition':cf==0 and tf==0,
            'native_target_pair_IoU':float((ga&gb).sum()/max(1,(ga|gb).sum()))})
    save(OUT/'CONDITION_INFORMATION.json',{'objects':rows,'pairs':pairs,'CPU_only':True,
        'model_generation_or_training_performed':False,'parent':ck,'no_change_to_targets_or_inputs':True,
        'identical_condition_different_target_pairs':[r for r in pairs if r['identical_effective_map_condition'] and r['native_target_pair_IoU']<.99],
        'unknown_roof_and_height_are_not_observed':'native-target memorization gate is not universal map-fidelity metric'})
    print('CONDITION_INFORMATION',rows,pairs,flush=True)

if __name__=='__main__':main()

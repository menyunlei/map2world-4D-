"""Measure the ONE corrected map field through the frozen branch, on CPU."""
from paths import read,save
from trellis2_map_object_data_v6_fix1 import OUT as OLD,ARMS,SEEDS
from trellis2_object_confidence_prepare_v7 import OUT
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
import torch,numpy as np
def rms(x):return float(x.square().mean().sqrt())
if __name__=='__main__':
    torch.set_num_threads(2);rows=[]
    original=read(OLD/'REAL_OBJECT_INPUTS.json')['records']
    corrected=read(OUT/'REAL_OBJECT_INPUTS.json')['records']
    for arm in ARMS:
        for seed in SEEDS:
            ck=read(OLD/arm/f'seed{seed}'/'CHECKPOINT_000300.json')
            branch=ObjectContractConditioner(reading_mode=arm).cpu().eval().requires_grad_(False)
            branch.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'])
            with torch.no_grad():
                for r in corrected:
                    old=next(x for x in original if x['mode']==r['mode'] and x['object_index']==r['object_index'])
                    row={'arm':arm,'training_seed':seed,'mode':r['mode'],'object_index':r['object_index']}
                    for name,record in [('old',old),('fixed',r)]:
                        with np.load(record['map_input']) as z:m={k:torch.from_numpy(z[k].copy()).unsqueeze(0) for k in z.files}
                        context=branch.prepare(m,native_world_grid())
                        features=branch(context).features
                        row[name+'_condition_RMS']=rms(features)
                        row[name+'_condition_spatial_RMS']=rms(features-features.mean(1,keepdim=True))
                        row[name+'_residual_RMS']=rms(branch.Wo(context.token_features))
                    rows.append(row)
    save(OUT/'FROZEN_BRANCH_SIGNAL_COMPARISON.json',{'rows':rows,'device':'CPU',
        'optimizer_updates':0,'same_weights_and_all_other_inputs':True,
        'signal_amplitude_is_not_geometry_success':True})
    print('ONE_INPUT_FIX_SIGNAL_MEASURED',len(rows),flush=True)

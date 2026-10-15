"""Portable trained branches, per-family raw results and exact owned GPU costs."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
import torch,numpy as np,csv

if __name__=='__main__':
    rows=[];flow=[];portable=[]
    for arm in ARMS:
        for seed in SEEDS:
            d=OUT/arm/f'seed{seed}'
            ck=read(d/'CHECKPOINT_000300.json');assert sha(ck['path'])==ck['sha256']
            state=torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state']
            path=d/'trained_adapter_000300.npz'
            with path.open('xb') as f:np.savez_compressed(f,**{k:v.numpy() for k,v in state.items()})
            portable.append({'arm':arm,'seed':seed,'file':str(path),'sha256':sha(path),
                 'source_checkpoint':ck,'parameter_count':sum(v.numel() for v in state.values()),
                 'contract':'ObjectContractConditioner same trained/inference object frames; map-only geometry'})
            for r in read(d/'STRUCTURE_COMPLETE.json')['rows']:rows.append({'arm':arm,'training_seed':seed,**r})
            for update in [0,20,100,300]:
                for r in read(d/f'HELDOUT_{update:06d}.json')['units']:flow.append({'arm':arm,'training_seed':seed,'update':update,**r})
    def write_csv(path,data):
        cols=list(dict.fromkeys(k for r in data for k in r))
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(data)
    write_csv(OUT/'PER_FAMILY_STRUCTURE.csv',rows)
    write_csv(OUT/'PER_FAMILY_FLOW.csv',flow)
    save(OUT/'PORTABLE_CHECKPOINTS.json',{'records':portable,'not_final_success':True})
    costs=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))]
    write_csv(OUT/'GPU_COST_STAGE1.csv',costs)
    save(OUT/'STAGE1_EVIDENCE.json',{'facts':read(OUT/'CONTRACT_FACTS.json'),
         'training_complete':read(OUT/'TRAINING_COMPLETE.json'),
         'portable_checkpoints':portable,'structure_rows':rows,'heldout_flow':flow,
         'gpu_seconds_stage1':sum(r['gpu_seconds'] for r in costs),
         'includes_encoding_preflight_load_idle_training_and_structure':True,
         'no_model_switch_or_base_weight_training':True,'historical_v5_unchanged':True})

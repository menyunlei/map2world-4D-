"""Archive actual small-set measurements; no new training or sampling."""
from paths import JOB,read,save,sha
from trellis2_structure_learnability_v8_prepare import OUT,ARMS,SEEDS
import json,csv,numpy as np

def main():
    decision=read(OUT/'SMALL_DECISION.json');assert (OUT/'SMALL_COMPLETE.json').exists()
    prior=read(OUT/'native_prior_check/RESULTS.json');plan=read(OUT/'PLAN.json')
    rows=[];training=[];pairs=[];freeze={str(OUT/'PLAN.json'):sha(OUT/'PLAN.json')}
    for seed in SEEDS:
        logs={}
        for arm in ARMS:
            d=OUT/'small'/arm/f'seed{seed}'/'train'
            done=read(d/'TRAIN_COMPLETE.json');assert done['actual_updates']==1000
            ck=read(d/'CHECKPOINT_001000.json');assert sha(ck['path'])==ck['sha256'];freeze[ck['path']]=ck['sha256']
            logs[arm]=[json.loads(line) for line in (d/'train_log.jsonl').read_text().splitlines()]
            assert len(logs[arm])==1000
            training.append({k:done[k] for k in ['arm','seed','parent','initial_branch_sha256','trainable_parameters',
                'lora_parameters','actual_updates','actual_seconds','exposure','geometry_images_read']})
            for update in [100,1000]:
                result=read(OUT/'small'/arm/f'seed{seed}'/f'eval_{update:06d}/RESULTS.json')
                rows += [{**r,'arm':arm,'training_seed':seed,'update':update} for r in result['rows']]
        same_slots=all(a['slots']==b['slots'] for a,b in zip(logs[ARMS[0]],logs[ARMS[1]]))
        same_parent=training[-2]['parent']==training[-1]['parent']
        same_initial=training[-2]['initial_branch_sha256']==training[-1]['initial_branch_sha256']
        assert same_slots and same_parent and same_initial
        pairs.append({'seed':seed,'identical_all_2000_TRAIN_slots_time_dropout_noise':same_slots,
            'identical_parent_and_adapter_initialization':same_parent and same_initial})
    costs=[read(p) for p in (OUT/'gpu_cost').glob('*.json')]
    total=sum(r['gpu_seconds'] for r in costs);assert total<=plan['gpu_budget']['total_occupancy_seconds']+20
    family=[]
    for arm in ARMS:
        for seed in SEEDS:
            correct=[r for r in rows if r['arm']==arm and r['training_seed']==seed and r['update']==1000 and r['mode']=='correct']
            for name in sorted({r['family_id'] for r in correct}):
                unit=[r for r in correct if r['family_id']==name]
                family.append({'arm':arm,'training_seed':seed,'family_id':name,
                    'objects':len({r['object_id'] for r in unit}),'generation_seeds':len({r['generation_seed'] for r in unit}),
                    'native_support_IoU_mean':float(np.mean([r['native_support_IoU'] for r in unit])),
                    'map_projection_IoU_mean':float(np.mean([r['map_projection_IoU'] for r in unit])),
                    'empty_objects':sum(r['support_cells']==0 for r in unit),
                    'families_not_rays_or_views_are_independent':True,'TRAIN_only':True})
    with (OUT/'PER_FAMILY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(family[0]));w.writeheader();w.writerows(family)
    save(OUT/'ASSET_CHECKPOINT_FREEZE.json',freeze)
    result={'status':decision['status'],'passing_arms':decision['passing_arms'],'pure_generation_summary':decision['summaries'],
        'per_family':family,'per_object_and_control':rows,'training':training,'matched_validation':pairs,
        'native_prior_check':prior,'total_GPU_seconds':total,'total_GPU_hours':total/3600,
        'total_budget_GPU_seconds':plan['gpu_budget']['total_occupancy_seconds'],
        'large_phase_started':(OUT/'LARGE_OWNER.json').exists(),'materials_generated':False,'real_map_world_delivered':False,
        'small_case_assets':'native Structure latents/support NPZ, not textured 3D worlds',
        'data_scope':'6 licensed TRAIN objects from 2 TRAIN map families; 381/102 split retained for conditional phase',
        'real_map_training_pairs_verified':0,'human_review':'PENDING_NOT_REQUESTED',
        'causal_scope':'same noise, map A vs paired map B; B output reused for wrong-target specificity, not independent trials',
        'original_v7_and_failed_owners_preserved':True}
    save(OUT/'FINAL_RESULTS_SMALL.json',result)
    save(OUT/'SESSION_STATE_SMALL_FINAL.json',{'stage':result['status'],'actual_optimizer_jobs':4,'updates_per_job':1000,
        'no_active_small_workers':True,'no_automations':True,'no_more_jobs_queued':True,
        'large_authorized_only_if_gate_passed':bool(decision['passing_arms']),'GPU_seconds':total})
    print('ARCHIVED_LEARNABILITY_RESULT',result['status'],'GPU_hours',total/3600,flush=True)

if __name__=='__main__':main()

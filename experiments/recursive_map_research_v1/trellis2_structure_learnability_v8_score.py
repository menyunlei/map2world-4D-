"""Preregistered pure-generation gate. No training loss enters the decision."""
from trellis2_structure_learnability_v8_prepare import OUT,ARMS,SEEDS
from paths import read,save
import numpy as np,csv

def main():
    plan=read(OUT/'PLAN.json');gate=plan['small_gate'];summaries=[];allrows=[];passed=[]
    for arm in ARMS:
        arm_pass=True
        for seed in SEEDS:
            result=read(OUT/'small'/arm/f'seed{seed}'/'eval_001000/RESULTS.json')
            for noise in plan['generation_seeds']:
                rows=[r for r in result['rows'] if r['generation_seed']==noise and r['mode']=='correct']
                wrong=[r for r in result['rows'] if r['generation_seed']==noise and r['mode']=='mismatched_paired_map']
                assert len(rows)==6 and len(wrong)==6
                successes=[r for r in rows if r['native_support_IoU']>=gate['native_support_IoU_at_least']
                    and r['map_projection_IoU']>=gate['map_projection_IoU_at_least'] and min(r['extent_cells'])>=3]
                correct=float(np.mean([r['native_support_IoU'] for r in rows]));mismatch=float(np.mean([r['native_support_IoU'] for r in wrong]))
                ok=len(successes)>=gate['minimum_successful_objects_per_train_seed_and_gen_seed'] and correct-mismatch>=gate['mean_correct_minus_mismatch_IoU_at_least']
                arm_pass &= ok
                summaries.append({'arm':arm,'training_seed':seed,'generation_seed':noise,'successful_objects':len(successes),
                    'total_objects':6,'correct_native_support_IoU_mean':correct,'mismatched_IoU_mean':mismatch,
                    'correct_minus_mismatch':correct-mismatch,'map_projection_IoU_mean':float(np.mean([r['map_projection_IoU'] for r in rows])),
                    'empty_objects':sum(r['support_cells']==0 for r in rows),'passes_fixed_learnability_gate':ok})
                allrows += [{**r,'arm':arm,'training_seed':seed} for r in result['rows'] if r['generation_seed']==noise]
        if arm_pass:passed.append(arm)
    costs=[read(p) for p in (OUT/'gpu_cost').glob('*.json')]
    decision={'status':'SMALL_TRAIN_PURE_NOISE_LEARNABLE' if passed else 'STOP_SMALL_TRAIN_NOT_LEARNED',
        'passing_arms':passed,'launch_large_matched':bool(passed),'summaries':summaries,'per_object':allrows,
        'TRAIN_only_not_generalization':True,'real_world_success':False,'materials_started':False,
        'large_483_manifest_split':{'TRAIN':381,'HELDOUT':102},'gpu_seconds':sum(r['gpu_seconds'] for r in costs),
        'inference_paired_changed_and_mismatch_reuse':'same generated paired-map B output, not independent evidence',
        'frozen_original_native_weights_and_decoders':True,'human_review':'PENDING_NOT_BLOCKING',
        'next_step':'matched 381-TRAIN/102-HELDOUT phase within same budget after coarse-output inspection' if passed else
            'no larger training or Map10/material generation; distinguish limited adaptation capacity from insufficient observable target information'}
    save(OUT/'SMALL_DECISION.json',decision)
    keys=list(summaries[0])
    with (OUT/'SMALL_SUMMARY.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(summaries)
    with (OUT/'GPU_COST.csv').open('x',newline='') as f:
        fields=['name','gpu_index','start_epoch','end_epoch','gpu_seconds','includes_load_idle','returncode']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in costs:w.writerow({k:(r['gpu']['index'] if k=='gpu_index' else r[k]) for k in fields})
    print('SMALL_PURE_GENERATION_DECISION',decision['status'],summaries,flush=True)

if __name__=='__main__':main()

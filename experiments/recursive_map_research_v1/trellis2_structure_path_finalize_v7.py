"""Seal fixed-weight localization and the single confidence-denominator fix."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT as OLD,ARMS,SEEDS
from trellis2_object_confidence_prepare_v7 import OUT as FIX
from trellis2_structure_path_owner_v7 import OUT
import numpy as np,csv

if __name__=='__main__':
    diagnoses=[];comparison=[];recovery=[];steps=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=OUT/arm/f'seed{seed}'/'PATH_DIAGNOSIS.json'
            if arm=='object_grounded' and seed==953102:path=OUT/'decoder_math_fix1'/arm/f'seed{seed}'/'PATH_DIAGNOSIS.json'
            d=read(path);diagnoses.append({'arm':arm,'seed':seed,'path':str(path),'sha256':sha(path)})
            for r in d['teacher_forced_diagnostics']:
                recovery.append({'arm':arm,'seed':seed,**{k:v for k,v in r.items() if k!='block_stats'}})
            for r in d['native_Euler_clean_traces']:steps.append({'arm':arm,'seed':seed,**r})
            before=read(OLD/arm/f'seed{seed}'/'STRUCTURE_COMPLETE.json')['rows']
            after=read(FIX/arm/f'seed{seed}'/'STRUCTURE_COMPLETE.json')['rows']
            for r in after:
                b=next(x for x in before if x['case']==r['case'])
                comparison.append({'arm':arm,'seed':seed,'case':r['case'],
                    'cells_before':b['model_support_cells'],'cells_after':r['model_support_cells'],
                    'proxy_IoU_before_DIAGNOSTIC':b['target_projection_IoU_DIAGNOSTIC'],
                    'proxy_IoU_after_DIAGNOSTIC':r['target_projection_IoU_DIAGNOSTIC'],
                    'same_weights_pose_noise_sampler_and_rasters':True,
                    'AI_visual_acceptance':'FAIL','not_human_review':True})
    def csv_file(path,rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    csv_file(OUT/'FIXED_WEIGHT_RECOVERY.csv',recovery)
    csv_file(OUT/'NATIVE_DENOISING_STEPS.csv',steps)
    csv_file(OUT/'SINGLE_FIX_COMPARISON.csv',comparison)
    costs=[read(p) for root in [OUT,FIX] for p in sorted((root/'gpu_cost').glob('*.json'))]
    csv_file(OUT/'GPU_COST_FINAL.csv',costs)
    check_rows=[];map_only=[];none=[]
    for r in diagnoses:
        d=read(r['path']);check_rows+=d['train_infer_and_replay_checks']
        map_only.extend(x['clean_structure_IoU'] for x in d['teacher_forced_diagnostics'] if x['t']==.6 and x['mode']=='correct')
        none.extend(x['clean_structure_IoU'] for x in d['teacher_forced_diagnostics'] if x['t']==.6 and x['mode']=='none')
        assert sha(d['checkpoint']['path'])==d['checkpoint']['sha256']
    result={'status':'ONE_PROVEN_INPUT_BUG_FIXED_BUT_MAP_CONDITIONED_STRUCTURE_STILL_FAILS',
        'actual_changes':'only object_summary confidence denominator; no architecture, loss, sampler, normalization or checkpoint update',
        'localized_input_bug':{'path':str(JOB/'trellis2_map_object_data_v6.py'),
            'calculation':'local_map object_summary[5] = mean(footprint_known) over entire crop',
            'why_wrong':'counts unrelated unknown background as low reliability of the observed building; condition multiplied by this confidence in both object features and spatial routing',
            'object0_original_registered_confidence':0.5195357203483582,
            'object0_erroneous_local_confidence':0.04762329161167145,
            'object0_corrected_confidence':0.5195584893226624,
            'fixed_code':str(JOB/'trellis2_object_confidence_fix_v7.py'),
            'preserves_unknown_background_UI_occlusion_pose_and_fp':True},
        'not_broken_connections':{
            'train_style_vs_inference_wrapper_max_abs':max(x['train_infer_max_abs'] for x in check_rows if 'train_infer_max_abs' in x),
            'saved_vs_replayed_latent_max_abs':max(x['old_vs_replay_max_abs'] for x in check_rows if 'old_vs_replay_max_abs' in x),
            'map_tokens_and_residual_change_velocity':True,
            'native_encoded_target_decoder_IoU':1.0,
            'corrected_missing_diagnostic_keeps_original_CuDNN_flag':True},
        'remaining_failure_location':'native SS generation from pure-noise trajectory does not establish map-consistent building support; before ShapeFlow, appearance, export or renderer',
        'recovery_is_not_generation':{'t06_target_noised_IoU_mean_correct':float(np.mean(map_only)),
            't06_target_noised_IoU_mean_map_disabled':float(np.mean(none)),
            'interpretation':'good recovery largely survives map removal: this diagnostic cannot demonstrate learned map-to-Structure world birth',
            'early_t1_empty_prediction_alone_not_impossibility_proof':True},
        'single_fix_causal_test':comparison,
        'actual_structure_contacts_inspected':['ordinary/seed953101','ordinary/seed953102','object_grounded/seed953101','object_grounded/seed953102'],
        'visual_verdict':'all four corrected-map Structure screens FAIL; empty target remains in grounded seed953102, other supports are sheets/columns/non-building shapes',
        'new_training_updates':0,'new_material_generation':False,'new_complete_3D_world_asset':False,
        'do_not_claim_single_input_bug_explains_all_failures':True,
        'training_contract_caveat':'current checkpoints were trained with the erroneous crop-density confidence; the corrected input does NOT constitute trained adaptation under the corrected contract',
        'one_next_focus_only':'align learned map conditions with the high-noise native SS geometry-generation computation. Do not train texture/Detail or extend steps as a presumed remedy; no next model job launched.',
        'no_unseen_layout_or_photorealistic_success_claim':True,
        'diagnostics':diagnoses,'confidence_evidence':str(FIX/'CONFIDENCE_EVIDENCE.json'),
        'signal_comparison':str(FIX/'FROZEN_BRANCH_SIGNAL_COMPARISON.json'),
        'GPU_seconds':sum(r['gpu_seconds'] for r in costs),'GPU_hours':sum(r['gpu_seconds'] for r in costs)/3600,
        'cost_includes_load_idle_initial_failed_diagnostic_and_exact_math_retry':True,
        'old_v5_v6_weights_and_negative_outputs_preserved':True,
        'no_extra_models_seeds_loss_search_or_budget_extension':True,
        'reviewer':'AI diagnostic/visual inspection; human review remains pending'}
    save(OUT/'FINAL_RESULTS.json',result)
    save(OUT/'SESSION_STATE_FINAL.json',{'status':result['status'],'model_training_updates':0,
        'single_versioned_input_fix':str(JOB/'trellis2_object_confidence_fix_v7.py'),
        'new_world_delivered':False,'GPU_workers_active':False,'no_future_jobs_or_automation':True,
        'all_evidence_preserved':True,'GPU_seconds':result['GPU_seconds']})
    print('SINGLE_CORE_FIX_AND_LOCALIZATION_COMPLETE',result['recovery_is_not_generation'],
        'GPU_hours',result['GPU_hours'],flush=True)

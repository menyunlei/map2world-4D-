"""Save all fixed outcomes and costs; no success inference from export counts."""
from paths import JOB,read,save,sha
import csv,collections,numpy as np
ROOT=JOB/'map_object_world_v1';OUT=ROOT/'trellis2_map_address_validity_v5/cpu_noise_fix1'
def main():
    delivery=read(OUT/'DELIVERY_COMPLETE.json');units=[];assets=[];edits=[]
    for arm in ['baseline','address_valid']:
        for seed in [953101,953102]:
            for row in read(OUT/arm/f'seed{seed}'/'PROBE_COMPLETE.json')['rows']:units.append({'arm':arm,'seed':seed,**row})
    for seed in [953101,953102]:
        for mode in ['correct','edit']:
            folder=OUT/'separated'/f'seed{seed}'/'assets'/mode
            record={'seed':seed,'mode':mode}
            if (folder/'ASSET.json').exists():record.update(asset=read(folder/'ASSET.json'),reload=read(folder/'views/RELOAD.json') if (folder/'views/RELOAD.json').exists() else None)
            elif (folder/'EMPTY_NATIVE_SUPPORT.json').exists():record.update(asset=None,failure='EMPTY_NATIVE_SUPPORT')
            else:record.update(asset=None,failure='DECODE_OR_RELOAD_FAILURE_PRESERVED',delivery_case=next(r for r in delivery['cases'] if r['seed']==seed and r['mode']==mode))
            assets.append(record)
        for arm in ['baseline','address_valid']:
            supports=[]
            for mode in ['correct','edit']:
                with np.load(OUT/arm/f'seed{seed}'/'support'/mode/'SS_GENERATED.npz') as f:supports.append(f['native_support64'][0,0].astype(bool))
            with np.load(OUT/'real_input/map.npz') as f:target=f['footprints'][-1]>0
            changed=supports[0]^supports[1]
            edits.append({'arm':arm,'seed':seed,'changed_surface_cells':int(changed.sum()),
                'changed_cells_in_last_proxy_column':int(changed[target.T].sum()),
                'changed_cells_outside_last_proxy_column':int(changed[~target.T].sum()),
                'only_SS_diagnostic_not_final_mesh_edit_success':True})
    fields=sorted({k for r in units for k in r})
    with (OUT/'PER_FAMILY_RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(units)
    phases=[ROOT/'trellis2_map_ss_failure_v4',ROOT/'trellis2_map_ss_failure_v4/cpu_noise_fix1',ROOT/'trellis2_map_condition_contract_v4',ROOT/'trellis2_map_address_validity_v5',OUT]
    costs=[{**read(p),'phase':str(phase.relative_to(ROOT))} for phase in phases for p in (phase/'gpu_cost').glob('*.json')]
    by_gpu=collections.defaultdict(list)
    for r in costs:by_gpu[r['gpu']['index']].append(r)
    for records in by_gpu.values():
        records.sort(key=lambda r:r['start_epoch']);assert all(a['end_epoch']<=b['start_epoch'] for a,b in zip(records,records[1:])), 'Double counting physical occupancy'
    with (OUT/'GPU_COST.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','name','gpu','GPU_seconds','returncode','includes_load_idle']);writer.writeheader()
        for r in costs:writer.writerow({'phase':r['phase'],'name':r['name'],'gpu':r['gpu']['index'],'GPU_seconds':r['gpu_seconds'],'returncode':r['returncode'],'includes_load_idle':r['includes_load_idle']})
    corrected=ROOT/'trellis2_map_ss_failure_v4/cpu_noise_fix1'
    coverage=[{'seed':seed,**r} for seed in [953101,953102] for r in read(corrected/f'KNOWN_COVERAGE_SEED{seed}.json')['rows']] if (corrected/'COVERAGE_COMPLETE.json').exists() else None
    save(OUT/'EVIDENCE_EXPORT.json',{'units':units,'assets':assets,'local_edit_SS_diagnostics':edits,
        'corrected_official_noise_coverage_diagnostics':coverage,'historical_baseline_exact':read(OUT/'BASELINE_REUSE_VERIFIED.json'),
        'trained_checkpoint_source':'existing v3 separated branch_000300 for each seed; unchanged weights',
        'trainable_parameter_count_existing':611584,'optimizer_updates_this_request':0,
        'single_model_intervention':'separate query-address validity from content confidence; all map confidence channels retained',
        'normalization_hypothesis_rejected':True,'earlier_CUDA_noise_coverage_comparisons_not_causal_evidence':True,
        'old_results_source_hashes_and_thresholds_preserved':True,'four_matched_GPU_jobs_executed':True,
        'actual_GPU_seconds_including_all_diagnosis_failed_jobs_load_idle':sum(r['gpu_seconds'] for r in costs),
        'real_map_input_limit':'10.jpg oblique screenshot four roof proxies, not measured true footprints/GPS',
        'appearance_reference':'unchanged stylized reference used only by texture stage, not real facade evidence',
        'visual_verdict_recorded_separately_after_actual_image_inspection':True,'no_more_model_changes_or_training':True})
    print('ADDRESS_NATIVE_EVIDENCE_EXPORTED',len(assets),sum(r['gpu_seconds'] for r in costs),flush=True)
if __name__=='__main__':main()

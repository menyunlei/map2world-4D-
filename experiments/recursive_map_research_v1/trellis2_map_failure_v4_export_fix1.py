"""Finalize negative outputs faithfully, including an empty scene and decode OOM."""
from paths import JOB,read,save
import csv,collections
ROOT=JOB/'map_object_world_v1';DIAG=ROOT/'trellis2_map_ss_failure_v4';TEST=ROOT/'trellis2_map_condition_contract_v4'
def main():
    assert (TEST/'separated/seed953102/assets/correct/views/RELOAD.json').exists()
    assert (DIAG/'COVERAGE_COMPLETE.json').exists()
    units=[];recovery=[];coverage=[];assets=[]
    for seed in [953101,953102]:
        probe=read(DIAG/f'FIXED_WEIGHT_SEED{seed}.json')
        for t in [1,.9,.6,.2]:
            correct=[r for r in probe['rows'] if r['mode']=='correct' and r['t']==t]
            recovery.append({'seed':seed,'t':t,'independent_families':len(correct),
                'decoded_clean_surface_IoU_mean':sum(r['decoded_clean_surface_IoU'] for r in correct)/len(correct),
                'target_information_in_input':t<1,'not_final_generated_world_metric':True})
        for r in read(DIAG/f'KNOWN_COVERAGE_SEED{seed}.json')['rows']:coverage.append({'seed':seed,**r})
        for arm in ['raw','native_norm']:
            for r in read(TEST/arm/f'seed{seed}'/'PROBE_COMPLETE.json')['rows']:units.append({'arm':arm,'seed':seed,**r})
        for mode in ['correct','edit']:
            folder=TEST/'separated'/f'seed{seed}'/'assets'/mode
            if (folder/'EMPTY_NATIVE_SUPPORT.json').exists():
                assets.append({'seed':seed,'mode':mode,'native_asset':None,'empty_support':True})
            elif (folder/'ASSET.json').exists():
                assets.append({'seed':seed,'mode':mode,'asset':read(folder/'ASSET.json'),
                    'reload':read(folder/'views/RELOAD.json'),'generation':read(folder/'GENERATION.json')})
            else:
                assets.append({'seed':seed,'mode':mode,'native_asset':None,'failure':'NATIVE_DECODER_CUDA_OOM',
                    'log':str(TEST/'logs'/f'subcell_norm_asset_{mode}_{seed}_native_decode.log'),
                    'sampled_SS_preserved':str(TEST/'native_norm'/f'seed{seed}'/('map10_'+mode)/'SS_GENERATED.npz'),
                    'no_pruning_resolution_change_or_retry':True})
    fields=sorted({k for r in units for k in r})
    with (TEST/'PER_FAMILY_SUPPORT_DIAGNOSTIC.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(units)
    costs=[read(p) for d in [DIAG,TEST] for p in (d/'gpu_cost').glob('*.json')]
    by_gpu=collections.defaultdict(list)
    for r in costs:by_gpu[r['gpu']['index']].append(r)
    for rows in by_gpu.values():
        rows.sort(key=lambda r:r['start_epoch'])
        assert all(a['end_epoch']<=b['start_epoch'] for a,b in zip(rows,rows[1:])), 'Occupancy overlap'
    with (TEST/'GPU_COST.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','gpu','GPU_seconds','returncode','includes_load_idle']);writer.writeheader()
        for r in costs:writer.writerow({'name':r['name'],'gpu':r['gpu']['index'],'GPU_seconds':r['gpu_seconds'],'returncode':r['returncode'],'includes_load_idle':r['includes_load_idle']})
    save(TEST/'EVIDENCE_EXPORT.json',{'recovery':recovery,'full_sampling_support_units':units,
        'known_mask_causal_diagnostic':coverage,'assets':assets,'GPU_seconds':sum(r['gpu_seconds'] for r in costs),
        'optimizer_updates_this_request':0,'historical_thresholds_and_outputs_unchanged':True,
        'single_interface_normalization_test_rejected':True,'visual_findings_separately_after_actual_inspection':True,
        'real_map': '10.jpg approximate roof proxies, not geodetically surveyed footprints',
        'no_qualified_world_claim':True,'no_more_training_or_candidate_launches':True})
    save(TEST/'DELIVERY_FINALIZED_WITH_FAILURES.json',{'assets_complete':1,'empty_structures':2,'native_decode_OOM':1,
        'no_fabricated_edit_asset':True,'human_review_pending_not_blocking':True,'no_active_jobs_from_this_delivery':True})
    print('NEGATIVE_EVIDENCE_EXPORTED',len(assets),sum(r['gpu_seconds'] for r in costs),flush=True)
if __name__=='__main__':main()

"""Append independent reload evidence and complete GPU receipts without overwrites."""
from paths import JOB,read,save
import collections,csv
ROOT=JOB/'map_object_world_v1';OUT=ROOT/'trellis2_map_address_validity_v5/cpu_noise_fix1'
def main():
    assert (OUT/'INDEPENDENT_RELOADS_COMPLETE.json').exists()
    evidence=read(OUT/'EVIDENCE_EXPORT.json')
    for record in evidence['assets']:
        folder=OUT/'separated'/f'seed{record["seed"]}'/'assets'/record['mode']
        if (folder/'views/RELOAD.json').exists():record['reload']=read(folder/'views/RELOAD.json')
    phases=[ROOT/'trellis2_map_ss_failure_v4',ROOT/'trellis2_map_ss_failure_v4/cpu_noise_fix1',ROOT/'trellis2_map_condition_contract_v4',ROOT/'trellis2_map_address_validity_v5',OUT]
    costs=[{**read(p),'phase':str(phase.relative_to(ROOT))} for phase in phases for p in (phase/'gpu_cost').glob('*.json')]
    by_gpu=collections.defaultdict(list)
    for r in costs:by_gpu[r['gpu']['index']].append(r)
    for rows in by_gpu.values():
        rows.sort(key=lambda r:r['start_epoch']);assert all(a['end_epoch']<=b['start_epoch'] for a,b in zip(rows,rows[1:])), 'Physical occupancy overlap'
    with (OUT/'GPU_COST_FINAL.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','name','gpu','GPU_seconds','returncode','includes_load_idle']);writer.writeheader()
        for r in costs:writer.writerow({'phase':r['phase'],'name':r['name'],'gpu':r['gpu']['index'],'GPU_seconds':r['gpu_seconds'],'returncode':r['returncode'],'includes_load_idle':r['includes_load_idle']})
    evidence['actual_GPU_seconds_including_all_diagnosis_failed_jobs_load_idle']=sum(r['gpu_seconds'] for r in costs)
    evidence['all_nonempty_assets_independently_reloaded']=True
    save(OUT/'EVIDENCE_FINAL.json',evidence)
    print('FINAL_COST_AND_RELOAD_EVIDENCE',evidence['actual_GPU_seconds_including_all_diagnosis_failed_jobs_load_idle'],flush=True)
if __name__=='__main__':main()

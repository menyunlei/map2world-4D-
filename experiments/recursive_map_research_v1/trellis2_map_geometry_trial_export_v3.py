"""CPU-only trained-weight and evidence export; never infers success from files."""
from paths import JOB,read,save,sha
import argparse
import collections
import csv
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['entry_v2','tokens_v3'],required=True)
    args=parser.parse_args()
    name='trellis2_map_geometry_entry_v2' if args.phase=='entry_v2' else 'trellis2_map_geometry_tokens_v3'
    root=JOB/'map_object_world_v1'/name
    assert (root/'TRAINING_COMPLETE.json').exists()
    assert not (root/'TRAINED_DELIVERY.json').exists()
    flow=[];checkpoints=[]
    for arm in ['original','separated']:
        for seed in [953101,953102]:
            directory=root/arm/f'seed{seed}'
            record=read(directory/'CHECKPOINT_000300.json')
            assert sha(record['path'])==record['sha256']
            checkpoint=torch.load(record['path'],map_location='cpu',weights_only=False)
            path=directory/'trained_map_condition.npz'
            values={k:v.detach().cpu().numpy() for k,v in checkpoint['branch_state'].items()}
            with path.open('xb') as handle:np.savez_compressed(handle,**values)
            with np.load(path) as loaded:
                assert set(loaded.files)==set(values)
                assert all(np.array_equal(loaded[k],values[k]) for k in values)
            portable={'arm':arm,'seed':seed,'path':str(path),'sha256':sha(path),
                'actual_learned_parameter_count':sum(v.size for v in values.values()),
                'additional_updates':300,'starting_parent_updates':100,
                'load':'branch.load_state_dict({k:torch.from_numpy(a[k]) for k in np.load(path).files},strict=True)',
                'original_checkpoint':record,'optimizer_and_rng_in_original_PT':True}
            save(directory/'PORTABLE_CHECKPOINT.json',portable);checkpoints.append(portable)
            units=read(directory/'HELDOUT_000300.json')['units']
            families=collections.defaultdict(list)
            for unit in units:families[unit['family_id']].append(unit)
            for family,cases in sorted(families.items()):
                row={'arm':arm,'seed':seed,'family':family,'paired_cases':len(cases)}
                for key in cases[0]:
                    if key.endswith('MSE'):
                        row[key]=sum(case[key] for case in cases)/len(cases)
                row['diagnostic_not_final_asset_success']=True;flow.append(row)
    with (root/'FLOW_PER_FAMILY_FINAL.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flow[0]));writer.writeheader();writer.writerows(flow)
    costs=[read(path) for path in sorted((root/'gpu_cost').glob('*.json'))]
    with (root/'GPU_COST_EXPORT.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['name','gpu','uuid','GPU_seconds','returncode','includes_load_idle'])
        writer.writeheader()
        for cost in costs:
            writer.writerow({'name':cost['name'],'gpu':cost['gpu']['index'],'uuid':cost['gpu']['uuid'],
                'GPU_seconds':cost['gpu_seconds'],'returncode':cost['returncode'],'includes_load_idle':cost['includes_load_idle']})
    assets=[]
    for arm in ['original','separated']:
        for seed in [953101,953102]:
            for mode in ['correct','edit']+(['shuffled'] if arm=='separated' else []):
                directory=root/arm/f'seed{seed}/assets'/mode
                if (directory/'ASSET.json').exists():
                    assets.append({'arm':arm,'seed':seed,'mode':mode,'asset':read(directory/'ASSET.json'),
                        'generation':read(directory/'GENERATION.json'),
                        'reload':read(directory/'views/RELOAD.json') if (directory/'views/RELOAD.json').exists() else None})
                else:
                    assets.append({'arm':arm,'seed':seed,'mode':mode,'asset':None,
                        'empty_support':read(directory/'EMPTY_NATIVE_SUPPORT.json') if (directory/'EMPTY_NATIVE_SUPPORT.json').exists() else None,
                        'not_fabricated_or_retried':True})
    save(root/'TRAINED_DELIVERY.json',{'phase':args.phase,'checkpoints':checkpoints,'assets':assets,
         'TRAIN_families':32,'TRAIN_cases':96,'heldout_assessed_families':4,'heldout_assessed_cases_each':12,
         'total_available_adapter_HELDOUT_families':8,'real_paired_TRAIN_verified':0,
         'real_map_eval':'10.jpg fixed four roof proxies, normalized layout not surveyed footprints',
         'appearance_reference':'fixed stylized building, not recovered real facades',
         'human_review':'PENDING_NOT_A_BLOCKER',
         'actual_GPU_seconds_receipted':sum(cost['gpu_seconds'] for cost in costs),
         'visual_verdict':'NOT_AUTOMATICALLY_INFERRED_FROM_EXPORT_OR_FLOW_MSE',
         'historical_results_and_thresholds_preserved':True})
    print('TRAINED_CHECKPOINTS_AND_NATIVE_EVIDENCE_EXPORTED',args.phase,flush=True)


if __name__=='__main__':main()

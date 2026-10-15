"""CPU portable trained weights and receipts, no new training/model sampling."""
from paths import JOB, read, save, sha
from trellis2_map_learning_prepare_fix1 import OUT
import torch, numpy as np, csv, collections

for seed in [953101,953102]:
    meta=read(OUT/f'seed{seed}/CHECKPOINT_000100.json')
    assert sha(meta['path'])==meta['sha256']
    ckpt=torch.load(meta['path'],map_location='cpu',weights_only=False)
    path=OUT/f'seed{seed}/trained_map_branch.npz'
    with path.open('xb') as f:
        np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in ckpt['branch_state'].items()})
    with np.load(path) as loaded:
        assert set(loaded.files)==set(ckpt['branch_state'])
        assert all(np.array_equal(loaded[k],ckpt['branch_state'][k].numpy()) for k in loaded.files)
    save(OUT/f'seed{seed}/PORTABLE_CHECKPOINT.json',{'file':str(path),'sha256':sha(path),
        'format':'actual NumPy named parameter arrays, weight-only; optimizer and RNG checkpoint remains original PT',
        'updates':100,'learned':True,'parameter_count':sum(v.numel() for v in ckpt['branch_state'].values()),
        'load':'branch.load_state_dict({k:torch.from_numpy(a[k]) for k in np.load(path).files},strict=True)',
        'geometry_correction_strength':'learned, no post-hoc manual boost','real_map_world_success':False})
rows=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))]
with (OUT/'TRAIN_GPU_COST.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['name','physical_gpu','uuid','GPU_seconds','returncode','includes_load_idle'])
    w.writeheader()
    for r in rows:
        w.writerow({'name':r['name'],'physical_gpu':r['gpu']['index'],'uuid':r['gpu']['uuid'],
            'GPU_seconds':r['gpu_seconds'],'returncode':r['returncode'],'includes_load_idle':r['includes_load_idle']})
save(OUT/'TRAIN_DELIVERY.json',{'training_seeds':[953101,953102],'updates_each':100,
    'TRAIN_records':96,'TRAIN_families':32,'available_adapter_heldout_records':24,
    'teacher_forced_assessed_records_per_seed':12,'teacher_forced_assessed_independent_families_per_seed':4,
    'actual_real_map_assets_pending':True,'training_GPU_seconds_including_failed_preflight':sum(r['gpu_seconds'] for r in rows),
    'real_paired_training_data_verified':0,'no_Map10_training':True,'no_frozen_prior_changes':True})
print('PORTABLE_CHECKPOINTS_AND_TRAIN_RECEIPTS_READY',flush=True)

"""One preregistered Structure-only experiment; historical files are read-only."""
from paths import JOB, read, save, sha
import numpy as np, hashlib, json

OLD=JOB/'map_object_world_v1/trellis2_map_object_contract_v6'
OUT=JOB/'map_object_world_v1/trellis2_structure_learnability_v8'
ARMS=['frozen_structure','structure_lora']
SEEDS=[953101,953102]
GEN_SEEDS=[970201,970202]

def prepare():
    assert not (OUT/'PLAN.json').exists(), 'Existing experiment must be preserved'
    manifest=read(OLD/'PAIRED_NATIVE_MANIFEST.json')
    records=manifest['records']; train=[r for r in records if r['split']=='adapter_TRAIN']
    held=[r for r in records if r['split']=='adapter_HELDOUT']
    assert len(records)==483 and len(train)==381 and len(held)==102
    assert not ({r['family_id'] for r in train}&{r['family_id'] for r in held})
    geoms={r['key']:r for r in read(OLD/'SOURCE_GEOMETRIES.json')['records']}
    selected=[];seen=set();audit=[]
    for r in train:
        if r['source_asset_id'] in seen:continue
        seen.add(r['source_asset_id']); rr={**r,**{'source_geometry_record':geoms[r['geometry_key']]}}
        for p,h in [(r['map_input'],r['map_sha256']),(r['native_target'],r['native_target_sha256']),
                    (r['source_manifest'],r['manifest_sha256'])]:assert sha(p)==h
        with np.load(r['native_target']) as z:
            support=z['support64'][0,0].astype(bool); latent=z['S0']
            assert latent.shape==(1,8,16,16,16) and np.isfinite(latent).all() and support.any()
        with np.load(r['map_input']) as z:
            fp=z['footprints'][int(z['target_object'])]>.5
            identity=hashlib.sha256(b''.join(z[k].tobytes() for k in sorted(z.files))).hexdigest()
        proj=support.any(2).T
        audit.append({'scene':r['scene_id'],'object_id':r['object_id'],'source_asset':r['source_asset_id'],
            'map_tensor_sha256':identity,'support_cells':int(support.sum()),
            'target_footprint_projection_IoU':float((fp&proj).sum()/max(1,(fp|proj).sum())),
            'geometry_key':r['geometry_key'],'frame':r['frame']})
        selected.append(rr)
        if len(selected)==6:break
    assert len(selected)==6 and len({a['map_tensor_sha256'] for a in audit})==6
    parents={str(s):read(OLD/'object_grounded'/f'seed{s}'/'CHECKPOINT_000300.json') for s in SEEDS}
    for r in parents.values():assert sha(r['path'])==r['sha256']
    sampler=JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json'
    files=['trellis2_structure_learnability_v8_prepare.py','trellis2_structure_learnability_v8_train.py',
        'trellis2_structure_learnability_v8_eval.py','trellis2_structure_learnability_v8_prior.py',
        'trellis2_structure_learnability_v8_owner.py','trellis2_structure_learnability_v8_score.py',
        'trellis2_structure_lora_v8.py','test_trellis2_structure_lora_v8.py',
        'trellis2_map_object_contract_v6.py','trellis2_object_confidence_fix_v7.py',
        'trellis2_map_geometry_token_module_v3.py','trellis2_map_object_module_v1.py',
        'trellis2_map_learning_train_v1.py','trellis2_map_learning_train_fix1.py']
    freeze={str(JOB/f):sha(JOB/f) for f in files}
    freeze[str(sampler)]=sha(sampler);freeze[str(OLD/'PAIRED_NATIVE_MANIFEST.json')]=sha(OLD/'PAIRED_NATIVE_MANIFEST.json')
    for r in selected:
        for p,h in [(r['map_input'],r['map_sha256']),(r['native_target'],r['native_target_sha256']),
                (r['source_geometry_record']['source_geometry'],r['source_geometry_record']['sha256']),
                (r['source_geometry_record']['source_mesh'],r['source_geometry_record']['source_mesh_sha256'])]:freeze[p]=h
    for p in parents.values():freeze[p['path']]=p['sha256']
    plan={'version':'v8','arms':ARMS,'training_seeds':SEEDS,'generation_seeds':GEN_SEEDS,
        'selected_records':selected,'selected_audit':audit,'parent_checkpoints':parents,
        'selection':'first six unique authored assets in legitimate TRAIN manifest order; before generated outputs',
        'fixed_backbone':'original TRELLIS.2 SS Flow; frozen original weights, frozen native SS decoder',
        'adapter':'existing 611584-parameter object_grounded adapter; v7 confidence preprocessing at TRAIN and inference',
        'sole_arm_difference':{'Structure_cross_attention_LoRA':{'blocks':[4,14,24],
            'projections':['to_q','to_kv','to_out'],'rank':8,'alpha':8,'dropout':0,'B_initialization':'zero'}},
        'small':{'updates':1000,'checkpoints':[20,100,1000],'effective_batch':2,
            'objects':6,'primary_checkpoint':1000,'train_seed_count':2},
        'large_conditional':{'updates':500,'checkpoints':[20,100,500],'primary_checkpoint':500,
            'all_manifest_records':483,'TRAIN_objects':381,'HELDOUT_objects':102,
            'restart':'same registered v6 parent, NOT selected small-set checkpoint',
            'requires':'pure-noise small-set learnability gate; both training seeds and both generation seeds'},
        'optimizer':{'name':'AdamW','lr':1e-4,'weight_decay':.01,'betas':[.9,.95],'eps':1e-8,'clip_grad':1.},
        'native_objective':{'loss':'flow MSE only','sigma_min':1e-5,'time':'sigmoid(N(1,1))',
            'condition_dropout':.1,'noise_policy':'identical independent CUDA generator per TRAIN slot',
            'dropped_condition':'backprop if graph has trainable dependency; LoRA learns native unconditional slots'},
        'evaluation':{'pure_noise':True,'noise_policy':'native CPU torch.randn then CUDA transfer',
            'sampler':read(sampler)['args']['sparse_structure_sampler'],
            'modes':['correct','changed_paired_map','mismatched_paired_map'],
            'changed':'cyclic next of six registered paired maps, original output pose held fixed; target B in local coordinates',
            'mismatched':'same generated B scored against original target A; reused output, NOT independent trials or shuffled-routing test',
            'unknown_heights':'exact native support IoU is TRAIN memorization diagnostic, not map-observable truth'},
        'small_gate':{'minimum_successful_objects_per_train_seed_and_gen_seed':5,'total_objects':6,
            'native_support_IoU_at_least':.50,'map_projection_IoU_at_least':.65,
            'mean_correct_minus_mismatch_IoU_at_least':.10,
            'applies_separately_to_each_arm':'at least one arm must pass all four seed/noise combinations',
            'no_empty_or_grossly_sheetlike_success':'generated support must occupy >=3 cells in X,Y,Z',
            'not_historical_gate_change':True},
        'gpu_budget':{'total_occupancy_seconds':21600,'includes':'loading, idle, preflight, failures, sampling, conditional large phase',
            'small_job_limit_seconds':2700,'large_job_limit_seconds':2000,'preflight_job_limit_seconds':600,
            'prior_job_limit_seconds':600,'eval_job_limit_seconds':900,'no_automatic_extension':True},
        'no_material_generation':True,'no_new_losses':True,'no_new_target_meshes':True,
        'TRAIN_map10':False,'real_city_paired_supervision':0,'v7_preserved':True,'human_review':'PENDING_NOT_BLOCKING'}
    save(OUT/'PLAN.json',plan);save(OUT/'SOURCE_FREEZE.json',freeze)
    print('REGISTERED',len(selected),len(train),len(held),plan['gpu_budget'],flush=True)

if __name__=='__main__':prepare()

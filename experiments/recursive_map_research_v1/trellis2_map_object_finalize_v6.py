"""Seal a failed object-contract screen without manufacturing a block asset."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
import csv,numpy as np

if __name__=='__main__':
    gate=read(OUT/'MATERIAL_GATE.json');assert not gate['run_native_fine_geometry_and_texture']
    rows=[];reports=[];edit_rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            d=OUT/arm/f'seed{seed}';r=read(d/'STRUCTURE_COMPLETE.json')['rows']
            heldout=[x for x in r if x['split']=='heldout_layout'];real=[x for x in r if x['case'].startswith('correct')]
            flows={u:read(d/f'HELDOUT_{u:06d}.json')['units'] for u in [0,20,100,300]}
            report={'arm':arm,'training_seed':seed,'additional_updates':300,
                'heldout_layout_families':len(heldout),'heldout_objects_per_family':1,
                'heldout_support_IoU_mean':float(np.mean([x['native_support_IoU'] for x in heldout])),
                'heldout_empty_objects':sum(x['model_support_cells']==0 for x in heldout),
                'real_objects_nonempty':sum(x['model_support_cells']>0 for x in real),
                'real_target_projection_IoU_DIAGNOSTIC':float(np.mean([x['target_projection_IoU_DIAGNOSTIC'] for x in real])),
                'real_complete_block':False,'per_object_visual_gate':'FAIL',
                'flow_at0_SS':float(np.mean([x['SS_flow_MSE'] for x in flows[0]])),
                'flow_at300_SS':float(np.mean([x['SS_flow_MSE'] for x in flows[300]])),
                'flow_at0_shape':float(np.mean([x['shape_flow_MSE'] for x in flows[0]])),
                'flow_at300_shape':float(np.mean([x['shape_flow_MSE'] for x in flows[300]])),
                'material_generated':False,'final_asset_quality_not_measured':True}
            reports.append(report)
            for x in r:rows.append({'arm':arm,'training_seed':seed,**x})
            for j in range(4):
                before=d/'structure'/f'correct_object{j}'/'SS_GENERATED.npz'
                after=d/'structure'/f'edit_object{j}'/'SS_GENERATED.npz'
                with np.load(before) as z:a=z['native_support64'][0,0].astype(bool)
                with np.load(after) as z:b=z['native_support64'][0,0].astype(bool)
                edit_rows.append({'arm':arm,'training_seed':seed,'object_index':j,
                    'edited_target':j==3,'pose_unchanged':True,
                    'Structure_symmetric_difference_fraction':float((a^b).sum()/max(1,(a|b).sum())),
                    'cells_before':int(a.sum()),'cells_after':int(b.sum()),
                    'same_object_noise':True,'correct_generated_building':False})
    def write_csv(path,data):
        fields=list(dict.fromkeys(k for r in data for k in r))
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    write_csv(OUT/'PER_SEED_RESULTS.csv',reports);write_csv(OUT/'FOOTPRINT_EDIT_RESULTS.csv',edit_rows)
    costs=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))]
    write_csv(OUT/'GPU_COST_FINAL.csv',costs)
    parents=read(OUT/'PLAN.json')['parent_checkpoints']
    assert all(sha(p['path'])==p['sha256'] for p in parents.values())
    source={str(JOB/n):sha(JOB/n) for n in [
        'trellis2_map_object_contract_v6.py','trellis2_map_object_train_v6.py',
        'trellis2_map_object_structure_v6.py','trellis2_map_object_data_v6.py','trellis2_map_object_data_v6_fix1.py']}
    result={'verdict':'OBJECT_CONTRACT_CORRECTED_AND_TRAINED_BUT_STRUCTURE_GENERATION_FAILED',
        'three_facts':read(OUT/'CONTRACT_FACTS.json'),'training':read(OUT/'TRAINING_COMPLETE.json'),
        'budget':read(OUT/'TRAIN_BUDGET_FREEZE.json'),'per_seed':reports,'footprint_edits':edit_rows,
        'material_gate':gate,'gpu_seconds':sum(x['gpu_seconds'] for x in costs),
        'gpu_hours':sum(x['gpu_seconds'] for x in costs)/3600,
        'portable_checkpoints':read(OUT/'PORTABLE_CHECKPOINTS.json')['records'],
        'model_failure_location':'native SS denoising under trained map-only object conditions; wrong support exists before fine geometry or texture',
        'next_step_recommendation':'Do not paint or assemble these invalid supports. Any next learned-condition alignment/capacity experiment needs a separately fixed budget; none launched.',
        'no_proven_unique_root_cause_or_backbone_impossibility':True,
        'real_map_training_pairs_verified':0,'actual_data':'legitimate synthetic object meshes; real screenshot used only for demonstration evaluation',
        'fixed_placement':'Map-only anchor/scale/orientation and registered ground; NOT learned',
        'background':'No generated road/ground/world background. Fixed diagnostic raster maps only.',
        'complete_3D_block_and_street_views_delivered':False,
        'unexecuted_material_and_asset_code':'Provided but locked by MATERIAL_GATE false; no new block GLB claimed',
        'historical_v5_preserved':True,'parent_hashes_verified_unchanged':True,
        'source_hashes':source,'human_review':'PENDING, not blocking',
        'no_training_extension_or_new_model_or_seed_search':True}
    save(OUT/'FINAL_RESULTS.json',result)
    save(OUT/'SESSION_STATE_FINAL.json',{'status':'BOUNDED_SCREEN_STOPPED_AFTER_MANDATORY_STRUCTURE_GATE_FAILURE',
         'training_updates_each':300,'arms':ARMS,'seeds':SEEDS,'GPU_jobs_running':False,
         'future_jobs_queued':False,'automation_created':False,'native_assets_complete':False,
         'outputs_preserved':True,'GPU_seconds':result['gpu_seconds'],'next_action':'No automatic additional GPU work'})
    print('FINAL_OBJECT_CONTRACT_RESULTS',reports,'GPU_seconds',result['gpu_seconds'],flush=True)

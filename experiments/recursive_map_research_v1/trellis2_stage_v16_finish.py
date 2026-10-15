"""Freeze negative result and exact evidence; CPU only, no additional jobs."""
from trellis2_stage_v16 import OUT,SEEDS,V15
from paths import JOB,read,save,sha
import tarfile,collections

assert not read(OUT/'POST_COMPLETE.json')['errors']
receipts=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))]
cost=sum(r['gpu_seconds'] for r in receipts);assert cost<=28800
families=[]
for seed in SEEDS:
    rows=read(OUT/'preservation'/f'seed{seed}/RESULTS.json')['rows']
    for (update,group,family),rs in __import__('itertools').groupby(
            sorted(rows,key=lambda r:(r['update'],r['group'],r['family'])),
            key=lambda r:(r['update'],r['group'],r['family'])):
        rs=list(rs);families.append({'seed':seed,'update':update,'group':group,'family':family,
            'objects':len(rs),'map_IoU':sum(r['map_IoU'] for r in rs)/len(rs),
            'native_target_IoU_DIAGNOSTIC_ONLY':sum(r['native_target_IoU_diagnostic'] for r in rs)/len(rs),
            'empty_objects':sum(r['empty'] for r in rs)})
scene=[]
for seed in SEEDS:
    for arm in ['baseline','repair']:
        row=read(OUT/'streets'/f'seed{seed}'/arm/'LAYOUT.json')
        row['agent_visual_verdict']='FAIL'
        row['agent_visual_reason']=(
            'Nonbuilding bodies, long disconnected roof-like sheets and floating geometry remain'
            if seed==953101 else 'Small/near-degenerate objects and oversized distorted mass; not a usable street')
        scene.append(row)
result={'verdict':'SINGLE_STRUCTURE_ADAPTATION_FAILED_STOP',
    'goal_achieved':False,'goal':'Four usable map-corresponding buildings forming a street on Map10',
    'attribution':{
        'empty_output':'Present in original Structure: seed953102 object3 support has zero cells, before Shape',
        'vehicle_like_form':'Nonbuilding main silhouettes already present in original Structure orthogonal projections',
        'floating_sheets':'Thin/disconnected and wrongly elevated supports already present before Shape; Shape follows them',
        'Shape_evidence':'With identical Shape checkpoint/noise and legitimate target Structure, two fixed TRAIN objects form coherent houses, both seeds',
        'codec_evidence':'Existing encoded target Shape with native decoder forms coherent reference houses for the same paired TRAIN objects',
        'limits':'No real Map10 3D GT exists. These controls do not prove Shape universally correct or identify every fine fragment causal origin.',
        'implementation_bug_proven':False},
    'one_intervention':read(OUT/'REPAIR_PLAN.json'),
    'training_complete':read(OUT/'REPAIR_TRAINING_COMPLETE.json'),
    'family_results':families,'scenes':scene,
    'structural_preservation':'Report actual TRAIN/DEV family metrics below, not training loss or nonempty counts',
    'generated_object_count_not_success':True,
    'map_control_status':'Not repaired: real-footprint Structure remains unstable; no street success claim',
    'decision':'Do not promote candidate; preserve original TRELLIS.2 and prior positive evidence only in its tested scope',
    'automatic_next_training':False,'additional_method_search':False,
    'Shape_or_decoder_modified':False,'new_losses':False,'material_generation':False,
    'geometry_repair_or_replacement':False,'Map10_training':False,
    'placement':'Fixed transforms derived from four input roof-proxy footprints, not learned world placement',
    'background':'Flat ground is a separately labeled render reference, not included in delivered generated GLBs',
    'GPU_seconds':cost,'GPU_hours':cost/3600,'GPU_limit_hours':8,
    'review':'Agent visual inspection, not human acceptance',
    'all_historical_results_preserved':True,'no_more_GPU_jobs':True}
save(OUT/'RESULTS.json',result)
save(OUT/'SESSION_STATE.json',{'status':'COMPLETE_FAILED_STOP','results':'RESULTS.json',
    'training_updates_each':100,'further_training_queued':False,'seed_search':False,
    'native_assets_and_checkpoints_preserved':True,'GPU_seconds':cost})
with (OUT/'GPU_COST.csv').open('x') as f:
    f.write('name,gpu_index,gpu_uuid,gpu_seconds,returncode,includes_load_idle\n')
    for r in receipts:f.write(f"{r['name']},{r['gpu']['index']},{r['gpu']['uuid']},{r['gpu_seconds']},{r['returncode']},true\n")
selected=[]
for path in sorted(OUT.rglob('*')):
    if not path.is_file() or any(x in path.relative_to(OUT).parts for x in ['work','cache','extensions','tmp']):continue
    if path.suffix.lower() in ['.json','.csv','.log','.jsonl','.jpg','.png','.mp4']:
        selected.append(path)
for seed in SEEDS:
    selected.extend([OUT/'repair'/f'seed{seed}'/f'checkpoint_{u:06d}.pt' for u in [20,100]])
asset_paths=list((OUT/'generated').glob('*/object*/geometry.glb'))+list((OUT/'streets').glob('*/*/street_buildings.glb'))
manifest={str(p.relative_to(OUT)):sha(p) for p in selected}
save(OUT/'EVIDENCE_MANIFEST.json',manifest)
save(OUT/'ASSET_MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in asset_paths})
archive=OUT/'evidence_and_checkpoints.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in selected+[OUT/'EVIDENCE_MANIFEST.json',OUT/'ASSET_MANIFEST.json']:
        tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    for p in sorted(JOB.glob('trellis2_stage_v16*.py')):tar.add(p,arcname='code/'+p.name,recursive=False)
print({'verdict':result['verdict'],'GPU_seconds':cost,'archive_bytes':archive.stat().st_size,
       'archive_sha256':sha(archive),'family_results':families},flush=True)

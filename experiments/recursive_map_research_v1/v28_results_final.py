"""One-entry-change result and fixed saved-asset comparisons; never inference."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_road_stage_v28';OLD=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    candidate=OUT/'native64_contiguous'
    before=read(OLD/'attempts/attempt-1/REPORT.json');after=read(candidate/'REPORT.json');plan=read(OUT/'PREGISTRATION.json')
    assert before['contract_sha256']==after['contract_sha256'] and before['verifier_sha256']==after['verifier_sha256']
    assert read(OUT/'views_float/baseline/RELOAD.json')['cameras']==read(OUT/'views_float/candidate/RELOAD.json')['cameras']
    assert sha(candidate/'world.glb')==read(candidate/'COMPLETE.json')['sha256']
    assert sha(OLD/'attempts/attempt-1/world.glb')==plan['immutable_inputs']['old_mesh']
    original_queries=np.load(OLD/'EVALUATION_QUERIES.npz')['road_xy_native']
    measured=np.load(candidate/'RAY_RESULTS.npz');assert np.array_equal(original_queries,measured['xy'])
    m=read(OLD/'map10_scaffold.json')
    panels=[]
    for label,path in [('v27 cascade / Structure32',OLD/'attempts/attempt-1/TOP_NATIVE.png'),('v28 native1024 / Structure64',candidate/'TOP_NATIVE.png')]:
        im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im)
        for features,color in [(m['buildings'],'#ffb76a'),(m['roads'],'#3df3cd')]:
            for f in features:
                p=[((x/100+.5)*im.width,(z/100+.5)*im.height) for x,z in f['polygon']];d.line(p+[p[0]],fill=color,width=4)
        panels.append((label,im.resize((640,640))))
    top=Image.new('RGB',(1280,676),'#17252f');draw=ImageDraw.Draw(top)
    for i,(label,im) in enumerate(panels):top.paste(im,(i*640,36));draw.text((i*640+8,10),label+' / SAME fixed world frame; no asset fitting',fill='white')
    top.save(OUT/'top_map_comparison.png')
    sheet=Image.new('RGB',(4*384,2*410),'#17252f');draw=ImageDraw.Draw(sheet)
    for row,which in enumerate(['baseline','candidate']):
        for col,view in enumerate(['oblique','street','lateral','top']):
            im=Image.open(OUT/f'views_float/{which}/{view}_appearance.png').resize((384,384));sheet.paste(im,(col*384,row*410+26));draw.text((col*384+8,row*410+7),which+' / '+view+' / fixed camera',fill='white')
    sheet.save(OUT/'fixed_views_comparison.png')
    close=Image.new('RGB',(2*384,2*410),'#17252f');draw=ImageDraw.Draw(close)
    for row,which in enumerate(['baseline','candidate']):
        for col,view in enumerate(['street','lateral']):
            close.paste(Image.open(OUT/f'views_float/{which}/{view}_appearance.png').resize((384,384)),(col*384,row*410+26))
            draw.text((col*384+8,row*410+7),which+' / '+view+' / identical world camera',fill='white')
    close.save(OUT/'street_lateral_comparison.png')
    costs=[dict(file=p.name,**read(p)) for p in OUT.glob('GPU_COST*.json')]
    diag=read(OUT/'STAGE_DIAGNOSIS.json');stat=[]
    for name,r in [('v27_cascade32',before),('v28_native64',after)]:
        stat.append(dict(name=name,road_ground_fraction=r['road_surface_near_display_ground_fraction'],
             road_air_obstruction_fraction=r['road_air_column_obstruction_fraction'],valid_native_mesh=r['valid_geometry'],
             vertices=r['vertices'],faces=r['faces'],semantic_building_correspondence=r['semantic_building_correspondence'],
             old_gate_verdict=r['decision']))
    improved=after['road_air_column_obstruction_fraction']<before['road_air_column_obstruction_fraction']
    result=dict(epoch=time.time(),experiment='v28_single_native_Structure64_entry_fix',training_updates=0,candidate_count=1,seed=942271,
        same_map=True,same_saved_Structure_latent=True,same_image=True,same_world_frame=True,same_query_and_verifier=True,
        original_v27_unchanged=True,coordinate_shift=False,geometry_repair=False,
        first_failure_localization=dict(
            Structure64='Saved latent replay matches native pooled32 exactly; positive road ground support exists, no road support in the declared above-ground+2voxel band.',
            interface='64->32 max pooling mixes road ground and free air. Ground bin spans -0.030..3.095 display units after pooling, so it is not a precise surface-height condition.',
            Shape='146/149 baseline blocked queries have top surfaces below3 display units; median1.552 near mixed-cell center1.532. Concrete low-level obstruction appears in resolved Shape geometry, not proof of149 building intrusions.',
            remaining_tall_obstructions='Report separately; not all obstruction is explained by low ground surfaces.'),
        only_fix='Official1024 Shape path retains64 Structure support, replacing1024_cascade32 entry. Frozen weights and decoder unchanged.',
        causal_limit='This bundles removal of LR cascade and finer support entry; fixed seed does not mean identical per-token noise for different sparse dimensions.',
        results=stat,targeted_road_obstruction_improved=improved,
        full_map_world_success=False,semantic_building_metric='UNRESOLVED; no height-derived building IoU',
        visual_review='AI inspected independently reloaded GLB appearance and neutral top/oblique/street/lateral images. Building masses remain recognizable. Both have foliage sheets and foreground obstruction; native64 does not pass street-world quality. Not human acceptance.',
        total_GPU_seconds=sum(x['gpu_seconds'] for x in costs),GPU_cost=costs,
        native_asset='native64_contiguous/world.glb',asset_sha256=sha(candidate/'world.glb'),
        technical_recovery=read(OUT/'ABI_FIX.json'),
        fixed_images=['top_map_comparison.png','fixed_views_comparison.png'],
        decision='STOP_SINGLE_ENTRY_FIX_INSUFFICIENT' if improved else 'STOP_SINGLE_ENTRY_FIX_NOT_EFFECTIVE',
        no_automatic_extension=True,no_more_training_or_retries=True,
        coordinate_caveat='Map10 is oblique manually traced roof/road proxies; arbitrary normalized display units. Display-floor score is not surveyed terrain accuracy.',
        source_hashes={n:sha(JOB/n) for n in ['v28_stage_check.py','v28_native64.py','v28_gpu_fallback.py','v28_contiguous_fix.py','v28_views.py','v28_views_float.py','v27_verify.py','v27_verify_fix1.py']})
    from v22_map_scaffold import contains
    bdata=np.load(OLD/'attempts/attempt-1/RAY_RESULTS.npz');world_xz=original_queries*np.array([100,-100])
    per=[]
    for road in m['roads']:
        mask=np.array([contains(road['polygon'],p) for p in world_xz]);n=int(mask.sum())
        if n:per.append(dict(id=road['id'],queries=n,
            baseline_obstruction_fraction=float(bdata['road_blocked'][mask].mean()),candidate_obstruction_fraction=float(measured['road_blocked'][mask].mean()),
            baseline_near_ground_fraction=float(bdata['near_ground'][mask].mean()),candidate_near_ground_fraction=float(measured['near_ground'][mask].mean())))
    result['per_existing_road_patch']=per
    result['all_road_patches_improved']=all(x['candidate_obstruction_fraction']<=x['baseline_obstruction_fraction'] for x in per)
    largest=max(per,key=lambda x:x['queries'])
    result['largest_sampled_road_patch']=dict(**largest,worsened=largest['candidate_obstruction_fraction']>largest['baseline_obstruction_fraction'])
    result['mean_improvement_is_not_joint_success']=True
    result['one_map_one_seed_only']=True
    old_bad=bdata['road_blocked'];new_bad=measured['road_blocked']
    result['surface_height_quantization_signature']=dict(
        Structure32_ground_cell_center_world=diag['stage']['pooled32_ground_center_world_height'],
        Structure64_ground_cell_center_world=sum(diag['stage']['dense64_ground_bin_world_bounds'])/2,
        baseline_blocked_surface_height_median=float(np.median(bdata['height'][old_bad])),
        candidate_blocked_surface_height_median=float(np.median(measured['height'][new_bad])),
        baseline_blocked_top_at_least_3_units=int((old_bad&(bdata['height']>=3)).sum()),
        candidate_blocked_top_at_least_3_units=int((new_bad&(measured['height']>=3)).sum()),
        interpretation='Consistent with surface placement tied to support-cell scale; not proof that all obstructions are buildings or an isolated pooling-only causal effect.')
    result['remaining_core_problem']='Voxel support is not a continuous road-surface constraint. Preserving64 reduces the observed scale error but still fails the unchanged0.5/8-unit clearance and ground gates; learned map-object layout correspondence also remains unestablished.'
    result['resource_note']='Fixed-card launch refused GPUs1-3 after another live CUDA owner appeared. Only an actually idle authorized GPU was leased; no unrelated process stopped or shared.'
    target=OUT/'RESULTS.json';tmp=target.with_suffix('.next');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(target)
    manifest=dict(candidate_asset=str(candidate/'world.glb'),sha256=sha(candidate/'world.glb'),
        result_sha256=sha(target),actual_camera_equality_checked=True,
        complete_PBR_retained_on_server=True,GLB_appearance='native base color preview, all original generated triangles, no geometry repair',
        scene_is_not_map_aligned=True,world_go_status='FAIL',
        images={n:sha(OUT/n) for n in ['fixed_views_comparison.png','street_lateral_comparison.png','top_map_comparison.png']},
        native_generator_entry=str(JOB/'v28_contiguous_fix.py'),training_updates=0,automatic_followup=False)
    target=OUT/'ASSET_MANIFEST.json';tmp=target.with_suffix('.next');tmp.write_text(json.dumps(manifest,indent=2),encoding='utf8');tmp.replace(target)
    print(json.dumps(dict(decision=result['decision'],results=stat,total_GPU_seconds=result['total_GPU_seconds']),indent=2))
if __name__=='__main__':main()

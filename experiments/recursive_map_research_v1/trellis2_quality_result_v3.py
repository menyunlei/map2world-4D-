"""Compact receipts for the finished fixed-input, fixed-seed, zero-training trial."""
from trellis2_asset_trial_v3 import OUT, DEST, REPO, SEEDS, save, read, sha
import csv
import json
import subprocess

costs=[read(p) for p in (OUT/'gpu_cost').glob('*.json')]
total=sum(r['gpu_seconds'] for r in costs)
with (OUT/'GPU_COST.csv').open('x',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['name','physical_gpu','uuid','GPU_seconds','returncode','includes_load_idle'])
    writer.writeheader()
    for r in costs:writer.writerow({'name':r['name'],'physical_gpu':r['gpu']['index'],'uuid':r['gpu']['uuid'],
          'GPU_seconds':r['gpu_seconds'],'returncode':r['returncode'],'includes_load_idle':True})
cases=[]
for seed in SEEDS:
    for name in ['map10_block','official_building']:
        root=OUT/f'seed{seed}'/name
        record={'seed':seed,'input':name,'raw_geometry':str(root/'RAW_MESH.npz'),
                'native_full_PBR':str(root/'RAW_NATIVE.pt'),'native_full_PBR_sha256':sha(root/'RAW_NATIVE.pt')}
        if name=='map10_block':
            record.update(full_PBR_GLB=str(root/'world.glb'),data=read(root/'ASSET.json'),
                          independent_render=read(root/'views_v4/render_manifest.json'),
                          quality='FAIL_CLEAR_EXPLORABLE_STREET',
                          agent_observations='Recognizable buildings, roofs, roads and tree color; actual geometry has large ground gaps and ragged walls/edges. Seed2 is also incomplete. No map-control PASS.')
        else:
            record.update(basecolor_preview_GLB=str(root/'building_basecolor_preview.glb'),
                          preview=read(root/'COMPATIBLE_GLB.json'),native_reload=read(root/'native_views_final/RELOAD.json'),
                          Blender_reload=read(root/'BLENDER_COMPATIBILITY.json'),
                          quality='POSITIVE_SINGLE_BUILDING_CAPABILITY_FIXED_TWO_SEEDS',
                          agent_observations='Clear visible doors/windows, coherent colored facade, sides/back and rooftop; lateral viewpoints retain the building volume. Stylized source and output. Small edge/plant artifacts and invented unobserved back details remain.',
                          full_PBR_GLB='NOT_COMPLETED_AT40MIN_UV_EXPORT_CAP',
                          renderer_limit='19.6M/21.8M faces exceed16.7M single-call raster limit; all triangles were inspected through official8M-face chunked opaque base-color rendering.')
        cases.append(record)
result={'status':'BOUNDED_ZERO_TRAINING_TRIAL_FINISHED','model':'official TRELLIS.2-4B',
    'repository':str(REPO),'official_source_commit':read(OUT/'SETUP_COMPLETE.json')['source_commit'],
    'official_weights':str(DEST/'weights/TRELLIS.2-4B'),'weight_manifest':str(DEST/'OFFICIAL_WEIGHTS.json'),
    'DINOv3':read(OUT/'SETUP_COMPLETE.json')['dinov3_path'],
    'fixed_inputs_and_sampler':read(OUT/'PLAN.json'),'seeds':SEEDS,'training_updates':0,
    'raw_generated_assets':4,'full_native_PBR_GLBs':2,'basecolor_compatible_GLBs':2,
    'cases':cases,'GPU_seconds':total,'GPU_hours':total/3600,'GPU_cost_includes_load_idle_failed_attempts':True,
    'AI_visual_review':True,'HUMAN_REVIEW_PENDING':True,
    'visual_evidence_inspected':['both Map10 textured/neutral contact sheets, seed942201 closeup and neutral closeup',
          'both building9-view native base-color contacts and closeups; compatible GLBs fresh-loaded in Blender'],
    'native_decoder_standard_step':'Original decode_latent calls Mesh.fill_holes; retained unchanged. No additional geometry cleanup/repair/pruning was applied during export or evaluation.',
    'compatibility_note':'Building GLBs preserve every vertex/face and generated base color/alpha at vertices. They do not contain the spatial roughness/metallic fields; these remain in RAW_NATIVE.pt. Native inspections use unlit base color with opaque visibility, not full PBR lighting or transparency peeling.',
    'decision':'Keep TRELLIS.2 as a promising single-building asset generator; direct Map10 screenshot-to-whole-block output does not meet explorable-world quality. No learned map control or photorealistic city success established.',
    'next_focus_if_authorized':'Map spatial conditioning and scene composition need a learned mechanism that preserves this asset prior; no new model training or seed search launched.',
    'all_GPU_jobs_finished':True,'historical_failed_models_and_results_preserved':True}
save(OUT/'RESULTS.json',result)
save(OUT/'SESSION_STATE.json',{'status':result['status'],'owned_GPU_jobs':'complete','model_training_updates':0,
     'single_building':'positive2/2 visual evidence, stylized','Map10_city':'fails explorable street quality',
     'PBR_export':'two cities completed; two building UV exports stopped at fixed40min; native PBR retained',
     'cost_GPU_seconds':total,'no_automatic_further_training':True})
print('RESULTS_READY',total,flush=True)

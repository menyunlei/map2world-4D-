"""Preserve one native scene test with separate geometry and visual findings."""
from trellis2_native_street_v17 import OUT,REPO
from paths import JOB,read,save,sha
import subprocess,tarfile

assert (OUT/'RELOAD.json').exists()
costs=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))]
cost=sum(r['gpu_seconds'] for r in costs)
commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
result={
    'verdict':'NATIVE_IMAGE_CONDITIONED_BLOCK_GENERATED_QUALITY_LIMITED',
    'model':'Original pretrained TRELLIS.2-4B','actual_source_commit':commit,
    'seed':942217,'model_samples':1,'training_updates':0,'map_adapter_or_LoRA_loaded':False,
    'input':'Official single-image multi-building urban example; stylized, not surveyed real street',
    'generated_together':['buildings','roofs','ground/plaza','stairs and small scene details','appearance'],
    'not_procedural_building_placement':True,'not_an_assembly_of_independent_building_samples':True,
    'asset':read(OUT/'ASSET.json'),
    'agent_findings':{
        'geometry':'Recognizable multi-building masses, roofs, base/ground and stairs from several sides; no large detached wings/vehicle-like main form seen in the fixed views',
        'appearance':'Coherent cream/red-roof palette and visible facade details; near-ground walls have blotchy/soft textures and vegetation is simplified',
        'novel_views':'Front/back/left/right/top and lateral views remain a coherent static asset; back details are generated hypotheses',
        'limits':'Miniature/stylized quality, not photorealism; not certified watertight or collision-navigable; only one fixed sample',
        'assessment':'AI visual inspection, not human acceptance'},
    'map_control_claim':False,'real_city_reconstruction_claim':False,
    'causal_comparison_limit':'This image-conditioned test has richer visual inputs than the map-conditioned tests; it is not a matched architecture ablation',
    'GPU_seconds':cost,'GPU_hours':cost/3600,'cost_includes_load_idle_and_failed_renderer_attempt':True,
    'view_interface_fix':'Original MeshWithVoxel renderer returns base_color, not attr; fixed reader keys only, no model resampling',
    'no_geometry_repair':True,'optional_native_hole_fill_disabled':True,
    'Blender_format':'GLB containing full generated geometry and vertex base color; full PBR remains in native server asset',
    'no_more_jobs':True,'automatic_training_or_additional_seeds':False}
save(OUT/'RESULTS.json',result)
save(OUT/'SESSION_STATE.json',{'status':'COMPLETE_ONE_NATIVE_SAMPLE','results':'RESULTS.json',
    'GPU_seconds':cost,'no_more_GPU_jobs':True,'native_asset_preserved':True,'map_control_unproven':True})
with (OUT/'GPU_COST.csv').open('x') as f:
    f.write('name,gpu_index,gpu_uuid,gpu_seconds,returncode,includes_load_idle\n')
    for r in costs:f.write(f"{r['name']},{r['gpu']['index']},{r['gpu']['uuid']},{r['gpu_seconds']},{r['returncode']},true\n")
selected=[]
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or any(x in p.relative_to(OUT).parts for x in ['work','frames','tmp','cache','extensions']):continue
    if p.suffix in ['.json','.csv','.log','.png','.jpg','.mp4']:selected.append(p)
save(OUT/'EVIDENCE_MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in selected})
archive=OUT/'native_street_evidence.tar.gz'
with tarfile.open(archive,'x:gz') as tar:
    for p in selected+[OUT/'EVIDENCE_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    for p in sorted(JOB.glob('trellis2_native_street_v17*.py')):tar.add(p,arcname='code/'+p.name,recursive=False)
print({'verdict':result['verdict'],'GPU_seconds':cost,'source_commit':commit,
       'evidence_bytes':archive.stat().st_size,'evidence_sha256':sha(archive)},flush=True)

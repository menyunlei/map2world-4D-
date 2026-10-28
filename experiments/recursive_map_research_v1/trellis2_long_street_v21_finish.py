"""Final CPU receipt for one native long street, not map-to-world validation."""
from trellis2_long_street_v21 import OUT
from paths import JOB,read,save,sha
import numpy as np,trimesh,tarfile,json
from PIL import Image,ImageDraw

a=read(OUT/'ASSET.json');scale=read(OUT/'ASSUMED_SCALE.json');tour=read(OUT/'TOUR.json')
scene=trimesh.load(OUT/'large_street_100m.glb',force='scene',process=False)
g=list(scene.geometry.values())[0];assert len(g.faces)==a['faces'] and len(g.vertices)==a['vertices']
bounds=scene.bounds;assert np.isclose(max(np.ptp(bounds,axis=0)[[0,2]]),100,atol=.001)
save(OUT/'METRIC_RELOAD.json',{'independent_CPU_process':True,'loaded_GLTF_root_transform':True,
    'vertices':len(g.vertices),'faces':len(g.faces),'vertex_color_shape':list(g.visual.vertex_colors.shape),
    'world_bounds_assumed_m':bounds.tolist(),'all_original_faces_preserved':True,'sha256':sha(OUT/'large_street_100m.glb')})
del scene,g
costs=[read(p) for p in sorted((OUT/'gpu_cost').glob('*.json'))];cost=sum(p['gpu_seconds'] for p in costs)
finding=read(OUT/'VISUAL_ASSESSMENT.json')
result={'verdict':'LARGER_NATIVE_STREET_GENERATED_QUALITY_LIMITED','one_new_3D_sample':True,
    'model':'Original pretrained TRELLIS.2, same native pipeline/weights/settings/seed942217 asv17',
    'reference':'One new imagegen-generated longer-street condition, v17 style reference; image is not a 3D result',
    'map_control':False,'training_updates':0,'model_switch':False,'old_assets_modified':False,
    'generated_as_one_scene':True,'not_repeated_v17_tiles':True,'not_procedural_building_assembly':True,
    'asset':'large_street_100m.glb','asset_sha256':scale['sha256'],'native_asset_sha256':a['sha256'],
    'scale':'Longest XY extent uniformly set to100m for viewing; not recovered real-world scale. New content comes from new generated mesh, not this scaling.',
    'dimensions_assumed_m':np.ptp(bounds,axis=0).tolist(),'mesh_vertices':a['vertices'],'mesh_faces':a['faces'],
    'surface_appearance':'native base-color exported as vertex colors; complete native PBR preserved on server inRAW_NATIVE.pt; noUV texture atlas conversion',
    'street_tour':{'seconds':20,'fps':24,'frames':480,'eye_height_assumed_m':1.6,
        'forward_m':tour['forward_distance_assumed_m'],'return_m':10,'map_overlay':'generated-asset top view, not input-map correspondence or RGB tracking'},
    'ground_centerline_check':read(OUT/'STREET_ROUTE.json')['ground_profile'],
    'AI_visual_inspection':finding,'human_acceptance':'NOT_CLAIMED','GPU_seconds_including_load_idle':cost,
    'imagegen_cost':'one external built-in image generation; not included inA800 occupancy; token/monetary cost unavailable',
    'limits':['one fixed sample only','reference-image-conditioned, notMap10-conditioned','rear/hidden details generated hypotheses',
        'stylized and softened close-range facade detail, notphotoreal','centerline clear does not certify all sidewalks/interiors or mesh watertightness'],
    'no_more_jobs':True}
save(OUT/'RESULTS.json',result)
save(OUT/'SESSION_STATE.json',{'status':'COMPLETE_ONE_LARGER_NATIVE_STREET','GPU_seconds':cost,'no_more_GPU_jobs':True,'map_control_unproven':True})
with (OUT/'GPU_COST.csv').open('x') as h:
    h.write('name,gpu_index,gpu_seconds,returncode\n')
    for x in costs:h.write(f"{x['name']},{x['gpu']['index']},{x['gpu_seconds']},{x['returncode']}\n")
old=JOB/'map_object_world_v1/trellis2_native_street_v17/views_keys_fix1/appearance/oblique.png'
new=OUT/'views/appearance/oblique.png';panel=Image.new('RGB',(1024,548),'#252525');d=ImageDraw.Draw(panel)
for i,(p,label) in enumerate([(old,'v17: compact plaza'),(new,'v21: NEW generated continuous street')]):
    panel.paste(Image.open(p).resize((512,512)),(512*i,36));d.text((512*i+8,10),label,fill='white')
panel.save(OUT/'V17_V21_COMPARISON.jpg',quality=95)
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>v21 — Larger generated street</title>
<style>body{{max-width:1200px;margin:28px auto;background:#202329;color:#eee;font:18px/1.6 sans-serif}}a{{color:#8bceff}}img,video{{max-width:100%}}small{{color:#bbb}}</style>
<h1>v21: Longer natively generated street</h1><p>The same TRELLIS.2, one fixed long-street reference image, one fixed seed and zero training. Buildings, roads and appearance were generated together in one model pass, without copying old blocks. All images and videos on this page come from independently loading the saved 3D asset.</p>
<p><a href="large_street_100m.glb">Download complete GLB (about 108MB, importable into Blender)</a> · <a href="street_20s_24fps.mp4">20-second street walkthrough and look back</a> · <a href="RESULTS.json">Results and limitations</a></p>
<video controls src="street_20s_24fps.mp4" poster="tour_views/0000.png"></video>
<p>480 frames / 24fps; the camera moves forward about 90 meters along the generated road, turns around and returns 10 meters, at an eye height of 1.6 meters. These metric values use a uniform display scale: the longest horizontal extent of the generated scene is set to 100 meters, not a surveyed measurement. The inset comes from the generated asset, not the input map.</p>
<img src="street_contact.jpg"><img src="V17_V21_COMPARISON.jpg"><img src="appearance_contact.jpg">
<p>Some windows, plants and walls remain blurry or simplified; this is not a photorealistic block, and Map10 map control has not been demonstrated. Inspection is an AI visual review, not human acceptance.</p>
<p>GPU occupancy was {cost:.1f} seconds (including loading and idle time); one external imagegen reference image is excluded from A800 occupancy. All GPU jobs have ended.</p>
<details><summary>Reference image and generation scope (the reference is not the final 3D result)</summary><img src="input.png"><p>imagegen only creates the conditioning reference image; TRELLIS.2 generates the final geometry and appearance. The source prompt is saved in the reference record in the same directory. No procedural city, output repair or seed selection.</p></details>
'''
with (OUT/'review.html').open('x',encoding='utf8') as h:h.write(page)
selected=[p for p in OUT.rglob('*') if p.is_file() and not any(x in p.relative_to(OUT).parts for x in ['work','frames','tmp','cache','extensions'])
    and p.suffix in ['.json','.csv','.log','.png','.jpg','.mp4','.html']]
save(OUT/'DELIVERY_MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in selected}|{'large_street_100m.glb':scale['sha256']})
with tarfile.open(OUT/'final_evidence.tar.gz','x:gz') as tar:
    for p in selected+[OUT/'DELIVERY_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
print(json.dumps({'verdict':result['verdict'],'GPU_seconds':cost,'archive_sha256':sha(OUT/'final_evidence.tar.gz'),'bytes':(OUT/'final_evidence.tar.gz').stat().st_size}),flush=True)

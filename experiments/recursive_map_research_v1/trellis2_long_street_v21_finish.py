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
page=f'''<!doctype html><meta charset="utf-8"><title>v21 — Larger generated street</title>
<style>body{{max-width:1200px;margin:28px auto;background:#202329;color:#eee;font:18px/1.6 sans-serif}}a{{color:#8bceff}}img,video{{max-width:100%}}small{{color:#bbb}}</style>
<h1>v21：更长的原生生成街道</h1><p>同一个TRELLIS.2，一张固定长街参考图，一个固定seed，零训练。建筑、道路和外观由模型一次生成，不是复制旧街区。此页图片和视频均来自独立加载保存的三维资产。</p>
<p><a href="large_street_100m.glb">下载完整GLB（约108MB，可导入Blender）</a> · <a href="street_20s_24fps.mp4">20秒街内行进与回看</a> · <a href="RESULTS.json">结果与限制</a></p>
<video controls src="street_20s_24fps.mp4" poster="tour_views/0000.png"></video>
<p>480帧／24fps；相机在生成道路内前进约90米、转身、返回10米，视高1.6米。以上米制为统一展示尺度：生成场景的最长水平边设为100米，并非测绘结果。小图来自生成资产，不是输入地图。</p>
<img src="street_contact.jpg"><img src="V17_V21_COMPARISON.jpg"><img src="appearance_contact.jpg">
<p>局部窗户、植物和墙面仍有模糊／简化；这不是照片级街区，也没有证明Map10地图控制。检查是AI视觉审查，不是人工验收。</p>
<p>GPU占用{cost:.1f}秒（含加载与空闲）；一次外部imagegen参考图不计入A800占用。全部GPU任务已结束。</p>
<details><summary>参考图与生成边界（参考图不是最终3D）</summary><img src="input.png"><p>imagegen只创建条件参考图；TRELLIS.2生成最终几何与外观。源提示词保存在同级参考记录中。无程序化城市、无输出修补、无种子筛选。</p></details>'''
with (OUT/'review.html').open('x',encoding='utf8') as h:h.write(page)
selected=[p for p in OUT.rglob('*') if p.is_file() and not any(x in p.relative_to(OUT).parts for x in ['work','frames','tmp','cache','extensions'])
    and p.suffix in ['.json','.csv','.log','.png','.jpg','.mp4','.html']]
save(OUT/'DELIVERY_MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in selected}|{'large_street_100m.glb':scale['sha256']})
with tarfile.open(OUT/'final_evidence.tar.gz','x:gz') as tar:
    for p in selected+[OUT/'DELIVERY_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
print(json.dumps({'verdict':result['verdict'],'GPU_seconds':cost,'archive_sha256':sha(OUT/'final_evidence.tar.gz'),'bytes':(OUT/'final_evidence.tar.gz').stat().st_size}),flush=True)

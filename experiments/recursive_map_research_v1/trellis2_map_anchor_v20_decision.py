"""Close one v20 feasibility test; CPU evidence only, no inference or training."""
from paths import read,save,sha
from trellis2_map_anchor_v20 import OUT
import json,html

p=read(OUT/'PLAN.json');blind=read(OUT/'BLIND_VISUAL_ASSESSMENT.json')
rows=[]
for mode in ['correct','edit']:
    st=read(OUT/mode/'STRUCTURE.json');mesh=read(OUT/mode/'ASSET.json');r=read(OUT/mode/'RELOAD.json')
    rows.append({'mode':mode,'SS_support_projection_IoU_DIAGNOSTIC':st['all_support_projection_IoU_diagnostic'],
        'final_all_geometry_projection_IoU_DIAGNOSTIC':r['all_geometry_above_floor_IoU_DIAGNOSTIC_ONLY'],
        'semantic_building_IoU':None,'semantic_status':'UNRESOLVED: incomplete/fragmented roofs; no target-conditioned face selection',
        'known_road_obstruction':r['known_road_obstruction'],'mesh_faces':mesh['faces'],
        'mesh_vertices':mesh['vertices'],'native_bounds':mesh['native_bounds'],
        'asset_sha256':mesh['sha256'],'asset_reloaded_in_new_process':r['new_process_exact_geometry'],
        'SS_noise_sha256':st['noise_sha256']})
assert rows[0]['SS_noise_sha256']==rows[1]['SS_noise_sha256']
result={'decision':'STOP','scope':'One custom sampler feasibility test; not a learned map-conditioned model',
    'preserved':{'v17':'positive native street-generation reference; untouched','v19':'failed map-control result; untouched, old projection metrics are not building-only'},
    'map_metric_correction':'Old metric kept triangles if any vertex exceeded height; vegetation and scene bases were not semantically excluded. Historical results preserved, not promoted to building scores.',
    'native_interface':'Structure Flow accepts x,t,cond, no native spatial-mask/inpainting contract; texture concat conditioning is not a Structure inpainting interface.',
    'single_intervention':'Within native Euler sampling: decode predicted clean support, enforce selected XY foreground/road-air constraints and a two-layer boundary anchor, re-encode, continue. Final Structure/Shape mesh unmasked and unrepaired.',
    'map_provided':['selected approximate XY roof-proxy masks','positive known road patches','foreground-only scope','base boundary at inherited assumed floor'],
    'model_generated':['remaining vertical Structure support','height/roof hypotheses','all fine mesh surfaces'],
    'not_map_truth':['unknown building heights','reference-image geometry/style bias','unknown/untraced map context','unverified physical meter scale'],
    'placement':'One fixed native normalized world coordinate frame; no fitting and no per-object transforms. GLB changes axes only. The inherited optional global z display offset is recorded, not baked into the neutral native assets.',
    'per_output':rows,
    'AI_visual_finding':{'correct':'Left small-map-object region is cypress-like thin spires, not a building; right mass is an incomplete tower/facade-like object with open exposed structure.',
        'edit':'Narrowing the right footprint changes it into a more skeletal multi-spire/tower arrangement; left remains non-building. No complete two-building street.',
        'height_change':'Reported as a generated hypothesis, not failed against an unknown ground-truth roof/height.',
        'appearance':'No material stage was run; neutral-shaded geometry alone was evaluated.',
        'human_review':'PENDING; these are AI observations, not human approval'},
    'target_blind_assessment_sha256':sha(OUT/'BLIND_VISUAL_ASSESSMENT.json'),
    'gates':{'two_nondegenerate_buildings':'FAIL','semantic_roof_union_IoU':'UNRESOLVED_NOT_PASS',
        'known_road_air':'PASS on registered sparse positive road patches only','local_edit_with_valid_buildings':'FAIL',
        'no_major_skeletal_fragmentation':'FAIL'},
    'architecture_evidence':'Map support can constrain where occupancy survives, but the unchanged image-conditioned native prior does not reliably supply building identity and coherent shape under those anchors. No native trained structural-inpainting interface was found.',
    'limits':'This single constrained-sampler failure is not proof that every future trained TRELLIS.2 extension is impossible. Sparse road patches do not validate a full road network. Two development objects are not held-out generalization.',
    'next_recommendation':'Prefer a pretrained city-layout-conditioned 3D backbone with multi-building paired-data training and explicit footprint/road control; do not proceed with another minor adapter or gradient revision of this frozen object/image prior in this line. No new model or training launched.',
    'implementation_failure_preserved':'First Shape call passed sparse-grid32 where official lr_resolution512 is required. Corrected that caller only, reused exact saved SS; original logs and GPU receipts preserved. Native noise addressing unchanged.',
    'GPU_seconds_including_failure_load_idle':read(OUT/'GPU_COMPLETE.json')['gpu_seconds'],
    'training_updates':0,'additional_seed_search':False,'materials_generated':False,'geometry_repair':False,
    'no_further_jobs':True}
save(OUT/'ASSESSMENT.json',result)
tr=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [r['mode'],f"{r['SS_support_projection_IoU_DIAGNOSTIC']:.3f}",f"{r['final_all_geometry_projection_IoU_DIAGNOSTIC']:.3f}",'UNRESOLVED',r['known_road_obstruction']])+'</tr>' for r in rows)
page=f'''<!doctype html><meta charset="utf-8"><title>v20 — STOP</title>
<style>body{{max-width:1500px;margin:28px auto;background:#14171f;color:#eee;font:17px/1.55 sans-serif}}a{{color:#85caff}}img{{max-width:100%}}video{{width:45%}}td,th{{padding:10px;border:1px solid #526}}small{{color:#bbb}}</style>
<h1>v20 — STOP</h1><p>约束改变了位置支撑，但没有生成两栋完整建筑。小对象成树状细柱，大对象成不完整塔楼／立面骨架。保留全部失败资产，不追加试验。</p>
<p>固定10.jpg两个近似屋顶轮廓，同一原生TRELLIS.2、参考图和噪声；无训练、无材质、无网格修补、无输出对齐拟合。地图提供XY与已知道路空域；模型提供其余高度、屋顶和细表面。高度并无地图真值。</p>
<img src="COMPARISON.jpg"><p>上：原图；下：footprint编辑。红线是固定地图轮廓；中间图是原生资产新进程重载后的中性材质渲染。建筑语义不合格，不能把支撑重合率叫作建筑成功率。</p>
<table><tr><th>布局</th><th>中间支撑IoU（诊断）</th><th>最终全部高处几何IoU（诊断）</th><th>建筑专属IoU</th><th>已登记道路遮挡</th></tr>{tr}</table>
<p>历史v19高度筛面指标混入植被／底座，仍保留为场景投影诊断。本次独立评审仅看生成图、不看目标轮廓；无法可信确定完整屋顶区域，建筑专属数值留空，绝不按目标mask筛面造分数。</p>
<p><a href="correct/geometry.glb">原图 GLB</a> · <a href="edit/geometry.glb">编辑 GLB</a> · <a href="ASSESSMENT.json">完整判定</a> · <a href="BLIND_VISUAL_ASSESSMENT.json">目标盲化AI审查</a> · <a href="PLAN.json">冻结方案</a></p>
<video controls src="correct/lateral_2s_24fps.mp4"></video> <video controls src="edit/lateral_2s_24fps.mp4"></video>
<p>GPU占用 {result['GPU_seconds_including_failure_load_idle']:.1f} 秒（含加载、空闲及失败调用）；所有GPU工作已结束。AI审查不等于人工验收。</p>
<p>结论：停止这一冻结先验的地图锚定修补线。建议下一阶段优先考虑已按多建筑城市布局预训练、具备footprint／道路条件的3D骨干；本轮不自动启动任何新模型或训练。</p>'''
with (OUT/'review.html').open('x',encoding='utf8') as h:h.write(page)
save(OUT/'SESSION_STATE.json',{'status':'COMPLETE_STOP','decision':'STOP','next_action':'None authorized automatically',
    'assets':[r['asset_sha256'] for r in rows],'training_updates':0,'GPU_seconds':result['GPU_seconds_including_failure_load_idle']})
print('V20_DECISION_STOP',flush=True)

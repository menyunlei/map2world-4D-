"""Summarize actual bounded online evidence, never equate render FPS with AI speed."""
from pathlib import Path
import json,hashlib,statistics
JOB=Path(__file__).resolve().parent
OUT=JOB/'map_object_world_v1/trellis2_realtime_v25'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def summary(xs):
    return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs)} if xs else {'n':0}
state=read(OUT/'STATE.json');events=lines(OUT/'EVENTS.jsonl');telemetry=lines(OUT/'BROWSER_TELEMETRY.jsonl')
seen=set();ui=[];samples=[]
export=read(OUT/'BROWSER_EXPORT.json') if (OUT/'BROWSER_EXPORT.json').exists() else None
if export:
    telemetry.append({'payload':{'snapshot':export['current'],'events':export['events']}})
    samples.extend(export.get('samples',[]))
for row in telemetry:
    p=row['payload'];snap=p['snapshot'];samples.append(snap)
    for ev in p.get('events',[]):
        key=(snap['session'],ev.get('type'),ev.get('epoch'),ev.get('t'))
        if key not in seen:seen.add(key);ui.append(dict(session=snap['session'],**ev))
result={'verdict':'INTERACTION_IMPROVED_GENERATION_NOT_REALTIME',
        'not_map10_aligned':True,'not_seamless':True,'training':False,
        'rendering':'Actual local user camera; software WebGL, not A800 frame rendering',
        'input_metric':'Event to requestAnimationFrame render submission, NOT input-to-photon',
        'camera_rtt_ms':summary([e['rtt_ms'] for e in ui if e['type']=='camera_report' and 'rtt_ms'in e]),
        'input_submission_ms':summary([e['ms'] for e in ui if e['type']=='input_render_submission']),
        'visible_FPS_samples':summary([s['fps'] for s in samples if s.get('visible') and s['fps']>0 and s.get('loaded')]),
        'render_resolution_scale_samples':sorted(set(s.get('renderScale',1) for s in samples)),
        'display_distance_default':40,'all_native_triangles_retained':True,
        'fixed_sampling':'1024_cascade, native12steps unchanged; no sample/quality sweep',
        'chunks':[],'gpu_seconds':0,'prelaunch_resource_failure_preserved':True,
        'GPU_availability_change':'GPU1-3 occupied by another PID after initial audit; never stopped or shared',
        'next_single_bottleneck':'Per-process Python/model cold loading; a resident model service is needed. No framewise generation claim.'}
result['browser_evidence_source']='Periodic telemetry plus full UI-exported JSON; periodic telemetry gap is preserved, not interpolated'
result['visual_limitations']='Latency-only implementation, not new world-quality success. Road mounds/artifacts remain in the new chunk, legacy seam failure remains; no geometry repair performed.'
result['completion_scope']='Two actual camera-triggered samples finished. New generation stops at2; exploration/revisit remains usable. Instant/unlimited AI generation is NOT implemented.'
result['metadata_caveat']='Inherited COMPLETE.json id is road-boundary-0--1; actual persistent identity is chunk directory + REQUEST.chunk_id + SHA, recorded separately here. Seed, placement and hashes verified; raw evidence retained unchanged.'
result['FPS_by_browser_session']={session:summary([s['fps'] for s in samples if s['session']==session and s.get('visible') and s['fps']>0 and s.get('loaded')]) for session in set(s['session'] for s in samples)}
for c in state['chunks']:
    p=OUT/'chunks'/c['id'];complete=read(p/'COMPLETE.json') if (p/'COMPLETE.json').exists() else {}
    stages=lines(p/'STAGES.jsonl')
    phases=[dict(stage=a['stage'],seconds=b['epoch']-a['epoch']) for a,b in zip(stages,stages[1:])]
    cost=read(p/'GPU_COST.json') if (p/'GPU_COST.json').exists() else {}
    result['gpu_seconds']+=cost.get('gpu_seconds',0)
    result['chunks'].append(dict(id=c['id'],status=c['status'],seed=complete.get('seed'),
        sha256=c.get('sha256'),position_fixed=c['position'],preview_seconds=c.get('preview_seconds'),
        final_local_ready_seconds=c.get('elapsed_seconds'),phases=phases,GPU=cost,
        final_faces=complete.get('faces'),independent_reload=(p/'LOCAL_RELOAD.json').exists(),
        browser_events=[e for e in ui if e.get('id')==c['id']]))
protected=[JOB/'map_object_world_v1/trellis2_long_street_v21/large_street_100m.glb',
           JOB/'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1/chunk_100m.glb']
result['protected_current_hashes']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in protected}
result['protected_hashes_unchanged']=list(result['protected_current_hashes'].values())==[
    'b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe',
    '5379954ffac716884c6f04f65a1956753af714acd74753692c75f07fca7c8408']
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(result,indent=2,ensure_ascii=False))

"""Measure the actual camera->GPU->asset timeline from saved evidence."""
from pathlib import Path
import json,hashlib,statistics,time
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_online_v23';CHUNK=OUT/'chunks/online-0--1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
state=read(OUT/'STATE.json');complete=read(CHUNK/'COMPLETE.json');cost=read(CHUNK/'GPU_COST.json');attempt=read(OUT/'ATTEMPT.json')
rows=[json.loads(x)['payload'] for x in (OUT/'BROWSER_TELEMETRY.jsonl').read_text().splitlines() if x.strip()]
events={}
for row in rows:
    for e in row['events']:events[(row['snapshot']['session'],e['type'],e['t'])]=e|{'session':row['snapshot']['session']}
ev=list(events.values());arrivals=[e for e in ev if e['type']=='chunk_arrival' and e['id']=='online-0--1'];assert arrivals,'Browser must actually load the generated asset'
generated=[r for r in rows if cost['start_epoch']*1000<=r['snapshot']['epoch']<=cost['end_epoch']*1000 and r['snapshot']['visible']]
positions={tuple(round(x,3) for x in r['snapshot']['position']) for r in generated}
movement=[e for e in ev if cost['start_epoch']*1000<=e['epoch']<=cost['end_epoch']*1000 and e['type'] in ['step','side','turn_back','key_down','key_up','look','home']]
fps=[r['snapshot']['fps'] for r in generated if r['snapshot']['fps']>0]
backend=[json.loads(x) for x in (OUT/'EVENTS.jsonl').read_text().splitlines()]
dispatch=[e for e in backend if e['event']=='remote_dispatch'];assert len(dispatch)==1
assert cost['start_epoch']>=attempt['epoch'] and complete['sha256']==read(CHUNK/'LOCAL_RELOAD.json')['sha256']
result={'verdict':'ONLINE_CAMERA_TRIGGER_AND_CACHE_PASS_CONTINUOUS_STREET_FAIL','generation_jobs':1,'training_updates':0,'seed':942219,
 'timeline':{'camera_trigger_epoch':attempt['epoch'],'gpu_process_start_epoch':cost['start_epoch'],'gpu_process_end_epoch':cost['end_epoch'],
 'asset_completed_remote_epoch':complete['completed_epoch'],'asset_published_local_epoch':state['ready_epoch'],
 'camera_to_local_ready_seconds':state['ready_epoch']-attempt['epoch'],
 'primary_browser_visible_interval_seconds':[state['ready_epoch']-attempt['epoch'],(OUT/'after_online_arrival.png').stat().st_mtime-attempt['epoch']],
 'primary_browser_exact_arrival_missing':True,'primary_browser_evidence':'after_online_arrival.png; automatic arrival was visible before any reload, but original-session telemetry ceased before arrival',
 'first_recorded_browser_arrival_epoch':min(e['epoch'] for e in arrivals)/1000,'first_recorded_arrival_is_revisit':all(e['session']!=attempt['session'] for e in arrivals),
 'GPU_process_seconds':cost['gpu_seconds'],'native_generation_seconds':complete['native_generation_seconds'],'camera_to_gpu_process_seconds':cost['start_epoch']-attempt['epoch']},
 'actual_concurrent_navigation':{'telemetry_samples':len(generated),'distinct_camera_positions':len(positions),'movement_events':movement,'fps_min_median_max':([min(fps),statistics.median(fps),max(fps)] if fps else None),'renderer':rows[0]['renderer'],'viewport':rows[0]['snapshot']['renderSize']},
 'asset':complete,'independent_remote_reload':read(CHUNK/'RELOAD.json'),'independent_local_reload':read(CHUNK/'LOCAL_RELOAD.json'),
 'visual_inspection':{'actual_saved_asset_inspected':True,'continuous_road':False,'observations':['New block road runs mostly across the old street rather than continuing it.','Fixed world placement leaves an approximately 17-display-unit gap between longitudinal bounds.','The forward view faces a new facade, so seamless street traversal is not established.','Both actual generated geometry and appearance load successfully; coarse facade/vegetation artifacts remain.'],'views':['after_online_arrival.png','online_top.png'],'geometry_or_pose_repair':False},
 'persistence':{'chunk_id':'online-0--1','sha256':complete['sha256'],'browser_load_sessions':len({e['session'] for e in arrivals}),'GPU_dispatches':len(dispatch),'durable_attempt_file':True,'camera_reset_on_arrival':False},
 'prelaunch_interface_failure_preserved':'PRELAUNCH_FAILURE.json; zero inference jobs had launched; empty SFTP request preserved on server; corrected write mode before same-seed camera re-trigger',
 'limits':['One finite ahead chunk in the fixed road direction, not unlimited generation.','Uses existing legitimate image conditioning; no native learned map control or neighboring mesh constraint.','No seamless geometry guarantee; no mesh repair or collision certification.','Asset is a full generated scene, with original vertex/index/color buffers retained under a global display transform.','Rendering FPS reflects software WebGL on this browser.','After initial full-distance test, a visible 80/160/300-unit draw-distance control was added to preserve UI responsiveness; complete assets and cached geometry are unchanged.','AI camera/render inspection is not human acceptance.']}
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
hashes={}
for p in OUT.rglob('*'):
    if p.is_file() and p.name not in ['BROWSER_TELEMETRY.jsonl','DELIVERY_HASHES.json','STATE.json','STATE.next','EVENTS.jsonl']:
        with p.open('rb') as f:hashes[str(p.relative_to(OUT)).replace('\\','/')]=hashlib.file_digest(f,'sha256').hexdigest()
(OUT/'DELIVERY_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
print(json.dumps({k:result[k] for k in ['verdict','timeline','actual_concurrent_navigation','persistence']},indent=2))

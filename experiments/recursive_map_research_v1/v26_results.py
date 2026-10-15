"""Read-only measurements of actual walk requests and generated chunks."""
from pathlib import Path
import json,time,hashlib
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_walk_generate_v26'
def read(p):return json.loads(p.read_text())
state=read(OUT/'STATE.json');chunks=[]
for c in state['chunks']:
 d=OUT/'chunks'/c['id'];row=dict(id=c['id'],status=c['status'],request_epoch=c.get('request_epoch'),
    sha256=c.get('sha256'),preview_seconds=c.get('preview_seconds'),end_to_end_seconds=c.get('elapsed_seconds'),
    actual_new_geometry=True if c['status']=='ready' else None)
 if (d/'LAUNCH.json').exists():
  launch=read(d/'LAUNCH.json');row.update(launch=launch,request_to_worker_seconds=launch['epoch']-read(d/'REQUEST.json')['epoch'])
 if (d/'COMPLETE.json').exists():row['complete']=read(d/'COMPLETE.json')
 row['independent_local_reload']=(d/'LOCAL_RELOAD.json').exists()
 chunks.append(row)
protected=[JOB/'map_object_world_v1/trellis2_long_street_v21/large_street_100m.glb',
 JOB/'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1/chunk_100m.glb']
protected.extend(JOB/f'map_object_world_v1/trellis2_realtime_v25/chunks/stream-0--{i}/chunk_100m.glb' for i in (2,3))
actual=[]
for p in protected:
 with p.open('rb') as f:actual.append(hashlib.file_digest(f,'sha256').hexdigest())
expected=['b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe',
 '5379954ffac716884c6f04f65a1956753af714acd74753692c75f07fca7c8408',
 'f4cb2967b8cc737911fdf0bfcc007cd5b6b637cf7e4556a568e6a503bfab2e5c',
 'abc58b5b434bc67c2a279b04b52424073af3fa76673a01a134272c8cc27c1193']
result=dict(epoch=time.time(),scope='Actual camera-driven bounded streaming, not map-control or seamless-world proof',
 decision='WALK_TRIGGER_AND_ASYNC_ARRIVAL_DEMONSTRATED; REQUEST_RACE_REPAIRED; SUSTAINED_PACING_NOT_PROVEN',
 model='Same TRELLIS.2 1024_cascade native12steps, v24 road conditioning unchanged',training=False,
 resident_contract='Pipeline and weights persist in CPU memory; original low_vram GPU transfers retained',
 first_preheat=read(OUT/'MODEL_READY.json') if (OUT/'MODEL_READY.json').exists() else None,
 repair_preheat=read(OUT/'MODEL_READY_FIX1.json') if (OUT/'MODEL_READY_FIX1.json').exists() else None,
 chunks=chunks,worker=state['worker'],budget=state['budget'],protected_hashes_unchanged=actual==expected,
 historical_hashes=actual,queue_failure='Reader saw empty REQUEST during SFTP write. Worker1 exited after first scene. Raw logs and cost preserved.',
 repair='Close REQUEST.upload then atomic rename; worker ignores incomplete legacy JSON; resume unstarted second request, no new seed.',
 timing_caveat='Second end-to-end latency includes failure/reload, not a warm-speed benchmark. Only active sample time is separable.',
 costs={name:read(OUT/name) for name in ['GPU_COST.json','GPU_COST_FIX1.json'] if (OUT/name).exists()},
 physical_cost_bound='269.83808159828186 seconds initial lease + <=630 replacement lease, <=900 total; no double counting per-chunk timings',
 limitations=['Not framewise AI generation','User can outrun prefetch','Road seams/quality remain unvalidated or failed as before','Not Map10 aligned','Third chunk intentionally available for live user testing until timeout or budget consumed'])
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(dict(decision=result['decision'],chunks=chunks,worker=state['worker'],protected_hashes_unchanged=actual==expected),indent=2))

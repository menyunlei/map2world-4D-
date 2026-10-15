"""CPU-only guard test: simultaneous camera requests may claim at most one job."""
import v23_server as server
from pathlib import Path
import tempfile,threading,concurrent.futures,json
with tempfile.TemporaryDirectory(prefix='v23_guard_',dir=Path(__file__).resolve().parent) as directory:
    server.OUT=Path(directory);server.CHUNK=server.OUT/'chunks';started=[]
    server.generate=lambda request:started.append(request)
    base={'session':'guard-test','world':'native','position':[0,2.73,45],'forward':[0,0,-1]}
    assert not server.camera(base)['accepted']
    assert not server.camera(base|{'position':[0,2.73,4],'forward':[0,0,1]})['accepted']
    assert not server.camera(base|{'position':[40,2.73,4]})['accepted']
    assert not server.camera(base|{'position':[0,2.73,4],'world':'map'})['accepted']
    with concurrent.futures.ThreadPoolExecutor(8) as pool:answers=list(pool.map(lambda _:server.camera(base|{'position':[0,2.73,4]}),range(32)))
    assert sum(x['accepted'] for x in answers)==1
    assert len(started)==1
    assert server.state['budget']['used']==1
    assert (server.OUT/'ATTEMPT.json').is_file()
    assert not server.camera(base|{'position':[0,2.73,-80]})['accepted']
    print(json.dumps({'CPU_only':True,'GPU_jobs':0,'eligible_requests':32,'durable_job_claims':1,'background_dispatches':len(started),'wrong_direction_or_world_does_not_trigger':True,'revisit_does_not_trigger':True}))

"""Loopback camera -> bounded SSH inference -> verified asset -> live browser."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
import argparse,getpass,json,os,time,threading,hashlib,shlex,math
import paramiko
JOB=Path(__file__).resolve().parent
OUT=JOB/'map_object_world_v1/trellis2_online_v23';CHUNK=OUT/'chunks/online-0--1'
REMOTE='/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1'
RCHUNK=REMOTE+'/map_object_world_v1/trellis2_online_v23/chunks/online-0--1'
PYTHON=REMOTE+'/trellis2_native_quality_v1/environment/bin/python'
V21=JOB/'map_object_world_v1/trellis2_long_street_v21'
VENDOR=JOB/'matched_backbone_adaptation_v1/asset_delivery_v3/interactive/dist/vendor'
lock=threading.RLock();ssh=None
state={'status':'waiting_camera','message':'Move forward to Z≤5 while facing forward to start the only actual inference run.','budget':{'max_generations':1,'used':0,'max_gpu_seconds':600},
 'v21':{'asset_url':'/assets/v21.glb','sha256':'b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe'},
 'trigger':{'z_lte':5,'x_abs_lte':25,'forward_z_lte':-.25},
 'chunk':{'id':'online-0--1','status':'waiting_camera','asset_url':'/assets/online.glb','sha256':None,'position':[0,0,-100],'model_generation_seconds':None}}
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(event,**kw):
    with lock:
        with (OUT/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'event':event,'epoch':time.time(),**kw})+'\n')
def persist():
    with lock:
        p=OUT/'STATE.next';p.write_text(json.dumps(state,indent=2),encoding='utf8');os.replace(p,OUT/'STATE.json')
def update(status,message,**kw):
    with lock:state.update(status=status,message=message,**kw);state['chunk']['status']=status;persist()
    log('status',status=status,message=message)
def execute(command,timeout=30):
    _,o,e=ssh.exec_command(command,timeout=timeout);a=o.read().decode();b=e.read().decode();code=o.channel.recv_exit_status()
    if code:raise RuntimeError(f'SSH command failed ({code}): {b[-1200:]} {a[-1200:]}')
    return a
def rjson(sftp,name):
    with sftp.open(RCHUNK+'/'+name,'rb') as f:return json.load(f)
def pull(sftp,name,expected=None):
    dest=CHUNK/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if expected:assert digest(dest)==expected
        return
    part=dest.with_suffix(dest.suffix+'.part');sftp.get(RCHUNK+'/'+name,str(part))
    if expected:assert digest(part)==expected,'Remote/local checksum mismatch'
    os.replace(part,dest)
def generate(request):
    try:
        log('asset_absent_before_launch',local_exists=(CHUNK/'chunk_100m.glb').exists())
        with ssh.open_sftp() as s:
            try:s.stat(RCHUNK+'/ASSET.json');raise RuntimeError('Pre-existing generated asset is forbidden')
            except FileNotFoundError:pass
            with s.open(RCHUNK+'/REQUEST.json','wx') as f:f.write(json.dumps(request))
        command=f'cd {shlex.quote(REMOTE)} && {shlex.quote(PYTHON)} -B -u v23_native_job.py --detach'
        launched=json.loads(execute(command));log('remote_dispatch',command=command,owner=launched)
        start=time.time();update('queued','Submitted to A800; loading TRELLIS.2.',generation_started_epoch=start)
        while time.time()-start<840:
            with ssh.open_sftp() as s:remote=rjson(s,'STATUS.json')
            stage=remote['status']
            if stage=='failed':raise RuntimeError(remote.get('error','Native inference failed'))
            if stage=='ready':break
            with lock:
                state['generation_elapsed_seconds']=time.time()-start
                if state['status']!=stage:update(stage,'TRELLIS.2 is generating a new 3D block; movement remains available.' if stage=='generating' else 'Generation complete; exporting native geometry.')
                state['remote_gpu']=remote.get('gpu',state.get('remote_gpu'));persist()
            time.sleep(1)
        else:raise RuntimeError('Bounded wall-clock deadline reached; remote generation has its own 600s hard limit')
        with ssh.open_sftp() as s:
            complete=rjson(s,'COMPLETE.json');update('transferring','Actual inference is complete; automatically downloading and verifying the new asset.')
            pull(s,'chunk_100m.glb',complete['sha256'])
            for name in ['COMPLETE.json','GPU_COST.json','PLAN.json','REQUEST.json','LAUNCH.json','RELOAD.json','ASSET.json','LOAD.json','SOURCE_FREEZE.json','ONE_ATTEMPT.json','logs/generation.log']:
                pull(s,name)
        # Actual local independent file reload; blocks only this background worker.
        import trimesh,numpy as np
        scene=trimesh.load(CHUNK/'chunk_100m.glb',force='scene',process=False)
        assert all(len(g.faces)>0 and np.isfinite(g.vertices).all() for g in scene.geometry.values())
        (CHUNK/'LOCAL_RELOAD.json').write_text(json.dumps({'epoch':time.time(),'bounds':scene.bounds.tolist(),'sha256':digest(CHUNK/'chunk_100m.glb'),'faces':sum(len(g.faces) for g in scene.geometry.values())}),encoding='utf8')
        with lock:
            state['chunk'].update(sha256=complete['sha256'],model_generation_seconds=complete['native_generation_seconds'],gpu_seconds=complete['gpu_seconds'],vertices=complete['vertices'],faces=complete['faces'])
            state['generation_elapsed_seconds']=time.time()-start
        update('ready','The new block has been generated, downloaded and independently loaded. Revisits reuse this cache; the budget for this run is exhausted.',ready_epoch=time.time())
        log('asset_published',sha256=complete['sha256'],camera_request=request,total_seconds=time.time()-start)
    except Exception as e:update('failed','This online inference attempt failed; logs are preserved, with no automatic retry.',error=str(e));log('failed',error=str(e))
def camera(data):
    p=data.get('position');d=data.get('forward')
    if not all(isinstance(a,list) and len(a)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and abs(x)<1e6 for x in a) for a in [p,d]):raise ValueError('Invalid camera')
    with lock:
        state['camera']={'position':p,'forward':d,'session':data.get('session'),'epoch':time.time()}
        eligible=data.get('world')=='native' and p[2]<=5 and abs(p[0])<=25 and d[2]<=-.25
        if not eligible or state['budget']['used']:return {'accepted':False,'status':state['status'],'budget':state['budget']}
        request={**state['camera'],'trigger':state['trigger'],'seed':942219,'chunk_id':'online-0--1','fps':data.get('fps'),'event':data.get('event')}
        # Exclusive durable claim prevents duplicates on repeated frames/revisits/restarts.
        with (OUT/'ATTEMPT.json').open('x',encoding='utf8') as f:json.dump(request,f)
        state['budget']['used']=1;update('queued','The camera entered the trigger region; one actual generation request has been registered.')
        log('camera_trigger',request=request);threading.Thread(target=generate,args=(request,),daemon=True).start()
        return {'accepted':True,'status':'queued','chunk_id':'online-0--1'}
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store' if self.path.startswith('/api/') or self.path.endswith(('.html','.js','.css')) else 'private,max-age=3600');super().end_headers()
    def send_json(self,data,code=200):
        b=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if urlsplit(self.path).path=='/api/state':
            with lock:return self.send_json(state)
        return super().do_GET()
    def do_POST(self):
        if self.path not in ['/api/camera','/api/telemetry']:return self.send_error(404)
        if self.headers.get('Origin')!=f'http://127.0.0.1:{self.server.server_port}':return self.send_error(403)
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<262144:return self.send_error(413)
        try:
            data=json.loads(self.rfile.read(n))
            if self.path=='/api/camera':return self.send_json(camera(data))
            with lock:
                with (OUT/'BROWSER_TELEMETRY.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'received_epoch':time.time(),'payload':data})+'\n')
            return self.send_json({'saved':True})
        except Exception as e:return self.send_json({'error':str(e)},400)
    def translate_path(self,path):
        p=urlsplit(path).path
        fixed={'/assets/v21.glb':V21/'large_street_100m.glb','/assets/v21_top.png':V21/'ORTHO_TOP.png','/assets/online.glb':CHUNK/'chunk_100m.glb'}
        if p in fixed:return str(fixed[p])
        if p.startswith('/vendor/'):
            name=p.removeprefix('/vendor/')
            return str(VENDOR/name) if name in ['three.module.js','OrbitControls.js'] else str(OUT/'__denied__')
        candidate=(OUT/(p.lstrip('/') or 'index.html')).resolve()
        return str(candidate) if candidate.is_relative_to(OUT.resolve()) else str(OUT/'__denied__')
    def list_directory(self,path):return self.send_error(403)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--port',type=int,default=8768);a.add_argument('--recover-unlaunched',action='store_true');args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);CHUNK.mkdir(parents=True,exist_ok=True)
    ssh=paramiko.SSHClient();ssh.load_host_keys(str(JOB.parent.parent/'work_known_hosts'))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=getpass.getpass('A800 SSH password (memory only): ')
    ssh.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=password,timeout=20,allow_agent=False,look_for_keys=False);password=None
    ssh.get_transport().set_keepalive(20)
    log('server_started',preexisting_asset=(CHUNK/'chunk_100m.glb').exists(),generation_not_started=True)
    if args.recover_unlaunched:
        previous=json.loads((OUT/'STATE.json').read_text(encoding='utf8'))
        assert previous.get('error')=='File not open for writing','Only the identified prelaunch write failure can be recovered'
        with ssh.open_sftp() as s:
            for name in ['ONE_ATTEMPT.json','LAUNCH.json','ASSET.json']:
                try:s.stat(RCHUNK+'/'+name)
                except FileNotFoundError:continue
                raise RuntimeError('Inference may already have launched; no automatic second job')
            assert s.stat(RCHUNK+'/REQUEST.json').st_size==0
            s.rename(RCHUNK+'/REQUEST.json',RCHUNK+'/REQUEST_PRELAUNCH_EMPTY.json')
        with (OUT/'PRELAUNCH_FAILURE.json').open('x',encoding='utf8') as f:json.dump(previous,f)
        (OUT/'ATTEMPT.json').rename(OUT/'PRELAUNCH_CAMERA_REQUEST.json')
        log('prelaunch_write_recovered',GPU_jobs_before_recovery=0,same_seed=942219,next_launch_requires_camera=True)
        persist()
    elif (OUT/'ATTEMPT.json').exists():
        state=json.loads((OUT/'STATE.json').read_text(encoding='utf8'))
        if state['status'] not in ['ready','failed']:update('failed','An existing attempt is incomplete; preserving state and preventing a duplicate launch.')
    else:persist()
    print(f'V23_READY http://127.0.0.1:{args.port}/ CAMERA_TRIGGER_ONLY max_generations=1',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

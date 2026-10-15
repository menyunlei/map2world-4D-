"""Loopback predictive camera queue + genuine progressive TRELLIS.2 geometry.

Two new chunks maximum, two idle authorized GPUs, 600s/job. Historical assets
are read-only cache entries. No model generation is needed to turn/revisit.
"""
from pathlib import Path
_old=Path(__file__).with_name('v23_server.py').read_text().split("\nif __name__=='__main__':")[0]
exec(compile(_old,str(Path(__file__).with_name('v23_server.py')),'exec'),globals())
OUT=JOB/'map_object_world_v1/trellis2_realtime_v25'
SCALE=100.0154161373455
PORT=8770
OLD=JOB/'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1'
WORKER=JOB/'map_object_world_v1/trellis2_online_v23/mesh_worker.js'
state={'status':'waiting_camera','message':'移动即请求前方两个区块；几何先到、外观后到。不是逐帧即时生成。',
       'budget':{'max_generations':2,'used':0,'max_gpu_seconds':1200,'max_concurrency':2},
       'lookahead_seconds':150,'map_aligned':False,'seamless_guarantee':False,
       'v21':{'asset_url':'/assets/v21.glb','sha256':'b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe'},
       'cached':[{'id':'road-boundary-0--1','status':'ready','position':[0,0,-SCALE],
                  'asset_url':'/assets/v24.glb','sha256':digest(OLD/'chunk_100m.glb'),'historical_immutable':True}],
       'chunks':[{'id':f'stream-0--{i}','index':i,'status':'waiting_camera','position':[0,0,-SCALE*i],
                   'asset_url':f'/assets/{i}/final.glb','preview_url':f'/assets/{i}/preview.glb'} for i in (2,3)]}


def chunkdir(c): return OUT/'chunks'/c['id']
def remote_dir(c): return REMOTE+'/map_object_world_v1/trellis2_realtime_v25/chunks/'+c['id']


def mark(c,status_name,**kw):
    with lock:
        c.update(status=status_name,**kw)
        state['status']='active' if any(x['status'] not in ('ready','failed','waiting_camera') for x in state['chunks']) else 'idle'
        persist()


def json_remote(s,path):
    with s.open(path,'rb') as f: return json.load(f)


def transfer(s,c,name,expected=None):
    p=chunkdir(c)/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if expected: assert digest(p)==expected
        return
    temp=p.with_suffix(p.suffix+'.part')
    s.get(remote_dir(c)+'/'+name,str(temp))
    if expected: assert digest(temp)==expected,'SHA256 mismatch'
    os.replace(temp,p)


def available_gpu():
    # Check BOTH live CUDA ownership and our existing advisory leases. A worker
    # loading Python may hold its lease before it appears in nvidia-smi.
    code="""import fcntl,json,subprocess
from pathlib import Path
known=['GPU-e0a56bf9-85ef-c658-67fa-457b76585254','GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa','GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71','GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d']
owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
free=[]
for i,u in enumerate(known):
 if u in owners or f'{i}, {u}' not in inventory:continue
 p=Path('gpu_leases')/f'gpu{i}.lock'
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:continue
  free.append(i);fcntl.flock(f,fcntl.LOCK_UN)
print(json.dumps(free))
"""
    return json.loads(execute(f'cd {shlex.quote(REMOTE)} && {shlex.quote(PYTHON)} -c {shlex.quote(code)}'))


def generate(c,request,resume=False):
    started=c.get('request_epoch',time.time());root=remote_dir(c)
    try:
        if not resume:
            waiting=time.time()
            while not available_gpu():
                mark(c,'queued',stage='waiting_authorized_idle_GPU',elapsed_seconds=time.time()-started)
                if time.time()-waiting>600:raise TimeoutError('No idle authorized GPU within600s; no inference launched')
                time.sleep(2)
            with ssh.open_sftp() as s:
                try: s.stat(root+'/ASSET.json'); raise RuntimeError('Existing output prohibited')
                except FileNotFoundError: pass
                with s.open(root+'/REQUEST.json','wx') as f: f.write(json.dumps(request))
            command=f'cd {shlex.quote(REMOTE)} && V25_CHUNK_INDEX={c["index"]} {shlex.quote(PYTHON)} -B -u v25_native_job.py --detach'
            launched=json.loads(execute(command))
            log('remote_dispatch',chunk=c['id'],owner=launched,command=command)
        else:
            log('reattach_existing_owner',chunk=c['id'],new_inference=False)
        while time.time()-started<840:
            with ssh.open_sftp() as s:
                remote=json_remote(s,root+'/STATUS.json')
                if remote['status']=='failed': raise RuntimeError(remote.get('error'))
                progress={}
                try: progress=json_remote(s,root+'/PROGRESS.json')
                except FileNotFoundError: pass
                mark(c,remote['status'],stage=progress.get('stage',remote['status']),gpu=remote.get('gpu',c.get('gpu')),
                     elapsed_seconds=time.time()-started)
                if not c.get('preview_sha256'):
                    try: preview=json_remote(s,root+'/PREVIEW.json')
                    except FileNotFoundError: preview=None
                    if preview:
                        transfer(s,c,'geometry_preview.glb',preview['sha256']); transfer(s,c,'PREVIEW.json')
                        mark(c,remote['status'],preview_sha256=preview['sha256'],preview_ready_epoch=time.time(),
                             preview_seconds=time.time()-started)
                        log('preview_published',chunk=c['id'],seconds=time.time()-started,**preview)
                if remote['status']=='ready':
                    complete=json_remote(s,root+'/COMPLETE.json')
                    mark(c,'transferring',stage='download_final')
                    transfer(s,c,'chunk_100m.glb',complete['sha256'])
                    for name in ['COMPLETE.json','GPU_COST.json','PLAN.json','REQUEST.json','LAUNCH.json',
                                 'RELOAD.json','ASSET.json','LOAD.json','SOURCE_FREEZE.json','ONE_ATTEMPT.json',
                                 'PREVIEW.json','STAGES.jsonl','logs/generation.log']:
                        transfer(s,c,name)
                    # Independent reload, no edits or repair; only background worker blocks.
                    import trimesh,numpy as np
                    scene=trimesh.load(chunkdir(c)/'chunk_100m.glb',force='scene',process=False)
                    assert all(len(g.faces)>0 and np.isfinite(g.vertices).all() for g in scene.geometry.values())
                    reload=dict(epoch=time.time(),sha256=complete['sha256'],bounds=scene.bounds.tolist(),
                                faces=sum(len(g.faces) for g in scene.geometry.values()))
                    (chunkdir(c)/'LOCAL_RELOAD.json').write_text(json.dumps(reload))
                    mark(c,'ready',stage='cached',sha256=complete['sha256'],gpu_seconds=complete['gpu_seconds'],
                         model_generation_seconds=complete['native_generation_seconds'],ready_epoch=time.time(),
                         elapsed_seconds=time.time()-started)
                    log('asset_published',chunk=c['id'],seconds=time.time()-started,sha256=complete['sha256'])
                    return
            time.sleep(.5)
        raise TimeoutError('Bounded deadline; owner has independent 600s hard limit')
    except Exception as e:
        mark(c,'failed',error=str(e),no_auto_retry=True);log('failed',chunk=c['id'],error=str(e))


def camera(data):
    p=data.get('position');d=data.get('forward')
    if not all(isinstance(a,list) and len(a)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and abs(x)<1e6 for x in a) for a in [p,d]):
        raise ValueError('Invalid camera')
    received=time.time();accepted=[]
    with lock:
        state['camera']=dict(position=p,forward=d,session=data.get('session'),epoch=received)
        # Real navigation only: looking at a page or polling does not launch inference.
        eligible=data.get('world')=='native' and p[2]<44.5 and abs(p[0])<30 and d[2]<-.25
        for c in state['chunks']:
            if not eligible or c['status']!='waiting_camera' or state['budget']['used']>=2: continue
            distance=max(0,p[2]-(c['position'][2]+50))
            # Conservative walk speed4; 150s lookahead requests BOTH bounded chunks
            # early, instead of waiting until the user stands at the empty boundary.
            if distance>4*state['lookahead_seconds']: continue
            request={**state['camera'],'fps':data.get('fps'),'seed':942225+c['index']-2,
                     'chunk_id':c['id'],'trigger':'movement_predictive_prefetch','distance_to_near_edge':distance,
                     'lookahead_seconds':state['lookahead_seconds'],'position_fixed':c['position']}
            directory=chunkdir(c);directory.mkdir(parents=True,exist_ok=True)
            with (directory/'ATTEMPT.json').open('x') as f: json.dump(request,f)
            state['budget']['used']+=1;c.update(status='queued',request_epoch=received)
            log('camera_trigger',chunk=c['id'],request=request)
            threading.Thread(target=generate,args=(c,request),daemon=True).start()
            accepted.append(c['id'])
        persist()
    return dict(accepted=accepted,status=state['status'],budget=state['budget'],server_received_epoch=received,
                cached_movement_requires_inference=False)


def monitor_gpu():
    while True:
        try:
            with lock:
                active=[c for c in state['chunks'] if c.get('gpu') and c['status'] not in ('ready','failed')]
            if active:
                raw=execute('nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader,nounits',timeout=10)
                inventory={p[0].strip():dict(utilization_percent=int(p[1]),memory_MiB=int(p[2]),epoch=time.time())
                           for line in raw.splitlines() if len(p:=line.split(','))==3}
                with lock:
                    for c in active:c['gpu_stats']=inventory.get(c['gpu']['uuid'])
                with (OUT/'GPU_LIVE.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=time.time(),chunks={c['id']:c.get('gpu_stats') for c in active}))+'\n')
        except Exception as e:
            log('gpu_monitor_error',error=str(e))
        time.sleep(2)


BaseHandler=Handler
class Handler(BaseHandler):
    def end_headers(self):
        path=urlsplit(self.path).path
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store' if path.startswith('/api/') or path in ('/','/index.html') or path.endswith(('.html','.js','.css')) else 'private,max-age=3600')
        # Bypass the preserved base helper, which incorrectly includes query
        # strings when deciding whether JavaScript can be cached for an hour.
        SimpleHTTPRequestHandler.end_headers(self)
    def translate_path(self,path):
        p=urlsplit(path).path
        if p=='/assets/v24.glb': return str(OLD/'chunk_100m.glb')
        if p=='/mesh_worker.js': return str(WORKER)
        for c in state['chunks']:
            for kind,name in [('final','chunk_100m.glb'),('preview','geometry_preview.glb')]:
                if p==f'/assets/{c["index"]}/{kind}.glb': return str(chunkdir(c)/name)
        return super().translate_path(path)
    def do_GET(self):
        if urlsplit(self.path).path=='/api/state':
            with lock: return self.send_json(state)
        return super().do_GET()


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=OUT/'RUN_CONTRACT.json'
    if not frozen.exists():
        frozen.write_text(json.dumps(dict(max_new_chunks=2,max_gpu_seconds=1200,max_concurrency=2,
            seeds=[942225,942226],pipeline='unchanged1024_cascade',native_sampling_steps='unchanged12',
            geometry_modification=False,training=False,instant_generation_claim=False,
            old_asset_hashes=[state['v21']['sha256'],state['cached'][0]['sha256']]),indent=2))
    if (OUT/'STATE.json').exists():
        state=json.loads((OUT/'STATE.json').read_text())
    ssh=paramiko.SSHClient();ssh.load_host_keys(str(JOB.parent.parent/'work_known_hosts'))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('A800 SSH password (memory only): ')
    ssh.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=secret,
                timeout=20,allow_agent=False,look_for_keys=False);secret=None
    ssh.get_transport().set_keepalive(20)
    if '--recover-no-gpu' in __import__('sys').argv:
        for c in state['chunks']:
            if c.get('error')!='No authorized idle GPU; attempt stops without launching':continue
            # Exactly a PRELAUNCH scheduling failure, not a model retry.
            with ssh.open_sftp() as s:
                for name in ['LAUNCH.json','GPU_COST.json','ASSET.json']:
                    try:s.stat(remote_dir(c)+'/'+name)
                    except FileNotFoundError:continue
                    raise RuntimeError('Cannot retry an already launched inference')
                for name in ['REQUEST.json','OWNER.json','ONE_ATTEMPT.json','STATUS.json','owner.log']:
                    s.rename(remote_dir(c)+'/'+name,remote_dir(c)+'/PRELAUNCH_NO_GPU_'+name)
            (chunkdir(c)/'ATTEMPT.json').rename(chunkdir(c)/'PRELAUNCH_NO_GPU_ATTEMPT.json')
            (chunkdir(c)/'PRELAUNCH_FAILURE.json').write_text(json.dumps(c,indent=2))
            c.update(status='waiting_camera',stage='waiting_camera',error=None)
            state['budget']['used']-=1
            log('prelaunch_gpu_unavailable_preserved',chunk=c['id'],inferences_launched=0)
    for c in state['chunks']:
        if c['status'] not in ('waiting_camera','ready','failed'):
            request=json.loads((chunkdir(c)/'ATTEMPT.json').read_text())
            threading.Thread(target=generate,args=(c,request,True),daemon=True).start()
    threading.Thread(target=monitor_gpu,daemon=True).start()
    persist();log('server_started',generation_not_started=True)
    print('V25_READY http://127.0.0.1:8770/ actual movement triggers2 bounded jobs',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()

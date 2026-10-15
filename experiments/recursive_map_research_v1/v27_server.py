"""Loopback autonomous cycle. Candidate quarantine is separate from world cache."""
from pathlib import Path
from http.server import ThreadingHTTPServer
import os,time,json,math,threading,getpass,shlex
import paramiko
from v23_server import Handler as OriginalHandler,digest,VENDOR
JOB=Path(__file__).resolve().parent
OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27';PORT=8772
REMOTE='/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1'
R=REMOTE+'/map_object_world_v1/trellis2_autonomous_v27'
lock=threading.RLock();ssh=None
state=dict(status='waiting_camera',worker={},budget={'max_generations':2,'used':0,'max_retries':1},
    message='Map10 骨架可移动。朝未知区域行走后才请求一次生成；未验收资产不会安装。',
    cycle=dict(id='map10-0',status='waiting_camera',attempts=[],accepted=None,feedback=None),
    weight_training=False,map10_semantics='oblique roof/road proxies, not surveyed footprints')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def atomic(p,x):
    tmp=Path(p).with_suffix('.next');tmp.write_text(json.dumps(x,ensure_ascii=False));os.replace(tmp,p)
def persist():atomic(OUT/'STATE.json',state)
def event(kind,**kw):
    with lock:
        with (OUT/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw),ensure_ascii=False)+'\n')
def rjson(s,path):
    with s.open(path,'rb') as f:return json.load(f)
def pull(s,src,dst,expected=None):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if expected:assert digest(dst)==expected
        return
    tmp=dst.with_suffix(dst.suffix+'.part');s.get(src,str(tmp))
    if expected:assert digest(tmp)==expected,'Asset checksum mismatch'
    os.replace(tmp,dst)
def camera(data):
    p=data.get('position');f=data.get('forward')
    assert all(isinstance(v,list) and len(v)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and abs(x)<1e6 for x in v) for v in (p,f))
    with lock:
        old=state.get('camera');now=time.time()
        moved=old is not None and sum((p[i]-old['position'][i])**2 for i in range(3))>.00001
        state['camera']=dict(position=p,forward=f,epoch=now,session=data.get('session'))
        if state['cycle']['status']!='waiting_camera' or data.get('world')!='map' or not moved:return dict(accepted=False,status=state['cycle']['status'])
        # Any in-bounds actual movement toward the single fixed unknown region.
        if not (-50<=p[0]<=50 and -50<=p[2]<=50):return dict(accepted=False)
        request=dict(**state['camera'],chunk_id='map10-0',seed=942271,trigger='actual_map_scaffold_movement',loaded=data.get('loaded',[]))
        with (OUT/'REQUEST.json').open('x') as h:json.dump(request,h)
        with ssh.open_sftp() as s:
            with s.open(R+'/REQUEST.upload','wx') as h:h.write(json.dumps(request))
            s.rename(R+'/REQUEST.upload',R+'/REQUEST.json')
        state['cycle']['status']='queued';state['status']='active';persist();event('camera_trigger',request=request)
        return dict(accepted=True,chunk_id='map10-0')
def monitor():
    seen=set()
    while True:
        try:
            with ssh.open_sftp() as s:
                try:worker=rjson(s,R+'/WORKER_STATUS.json')
                except FileNotFoundError:worker={'stage':'starting'}
                for name in ['MODEL_READY.json','GPU_COST.json','OWNER_LAUNCH.json']:
                    try:pull(s,R+'/'+name,OUT/name)
                    except FileNotFoundError:pass
                for i in range(2):
                    rd=R+f'/attempts/attempt-{i}';d=OUT/'attempts'/f'attempt-{i}'
                    try:complete=rjson(s,rd+'/COMPLETE.json')
                    except FileNotFoundError:continue
                    if i not in seen:
                        with lock:
                            state['cycle']['status']='checking';state['message']='候选生成完成；独立验证中，不替换世界。';persist()
                        pull(s,rd+'/world.glb',d/'world.glb',complete['sha256'])
                        for name in ['COMPLETE.json','ASSET.json','LOAD.json','ONE_ATTEMPT.json','STAGES.jsonl','GENERATED_STRUCTURE.npz']:
                            pull(s,rd+'/'+name,d/name)
                        seen.add(i);event('candidate_downloaded',attempt=i,sha256=complete['sha256'],quarantined_until_verified=True)
                    try:report=rjson(s,rd+'/REPORT.json')
                    except FileNotFoundError:continue
                    for name in ['REPORT.json','TOP_NATIVE.png','RAY_RESULTS.npz','verify.log']:
                        pull(s,rd+'/'+name,d/name)
                    for name in ['FEEDBACK.json','CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','QUARANTINED.json']:
                        try:pull(s,rd+'/'+name,d/name)
                        except FileNotFoundError:pass
                    with lock:
                        record=dict(id=f'attempt-{i}',status='accepted' if report['decision']['accepted'] else 'quarantined',
                            sha256=complete['sha256'],asset_url=f'/attempts/attempt-{i}/world.glb',report=report,
                            seconds=complete['seconds'],model_pipe_id=complete['resident_pipe_id'])
                        rows=state['cycle']['attempts'];existing=next((x for x in rows if x['id']==record['id']),None)
                        if existing:existing.update(record)
                        else:rows.append(record);event('independent_verdict',attempt=i,decision=report['decision'])
                        state['budget']['used']=len(rows);state['cycle']['feedback']=report['targeted_feedback']
                        if report['decision']['accepted']:
                            remote_accept=rjson(s,R+'/ACCEPTED.json')
                            assert remote_accept['sha256']==complete['sha256']
                            accepted=state['cycle']['accepted'];assert accepted is None or accepted['sha256']==record['sha256']
                            if not accepted:
                                state['cycle']['accepted']={k:record[k] for k in ['id','sha256','asset_url']}
                                with (OUT/'ACCEPTED.json').open('x') as h:json.dump(remote_accept,h)
                        persist()
                with lock:
                    state['worker']=worker
                    if worker['stage']=='stopped':
                        state['status']='stopped';state['cycle']['status']='accepted' if state['cycle']['accepted'] else 'stopped_keep_scaffold'
                        state['message']='有界循环结束：未通过的候选保留隔离，原世界未修改。' if not state['cycle']['accepted'] else '候选通过，已永久缓存。'
                    elif worker['stage']=='failed':state['status']='failed';state['cycle']['status']='failed_keep_scaffold'
                    elif state['cycle']['status']!='waiting_camera':state['cycle']['status']=worker['stage']
                    persist()
            if worker['stage'] in ('failed','stopped'):
                if (OUT/'GPU_COST.json').exists():return
        except Exception as e:event('monitor_error',error=str(e))
        time.sleep(1)
class Handler(OriginalHandler):
    def do_GET(self):
        from urllib.parse import urlsplit
        if urlsplit(self.path).path=='/api/state':
            with lock:return self.send_json(state)
        return super().do_GET()
    def do_POST(self):
        if self.path not in ['/api/camera','/api/telemetry']:return self.send_error(404)
        if self.headers.get('Origin')!=f'http://127.0.0.1:{PORT}':return self.send_error(403)
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<262144:return self.send_error(413)
        try:
            data=json.loads(self.rfile.read(n))
            if self.path=='/api/camera':return self.send_json(camera(data))
            with lock:
                with (OUT/'BROWSER_TELEMETRY.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps(dict(epoch=time.time(),payload=data))+'\n')
            return self.send_json(dict(saved=True))
        except Exception as e:return self.send_json(dict(error=str(e)),400)
    def translate_path(self,path):
        from urllib.parse import urlsplit
        p=urlsplit(path).path
        if p.startswith('/vendor/'):
            n=p.removeprefix('/vendor/');return str(VENDOR/n) if n in ['three.module.js','OrbitControls.js'] else str(OUT/'__denied__')
        allowed={'/','/index.html','/app.js','/style.css','/mesh_worker.js','/map10_scaffold.json','/map10_reference.png','/CONTRACT.json'}
        if p in allowed:return str(OUT/('index.html' if p=='/' else p.lstrip('/')))
        if p in {f'/attempts/attempt-{i}/{n}' for i in (0,1) for n in ['world.glb','TOP_NATIVE.png','REPORT.json']}:return str(OUT/p.lstrip('/'))
        return str(OUT/'__denied__')
if __name__=='__main__':
    if (OUT/'STATE.json').exists():state=read(OUT/'STATE.json')
    ssh=paramiko.SSHClient();ssh.load_host_keys(str(JOB.parent.parent/'work_known_hosts'));ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('A800 SSH password (memory only): ')
    ssh.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=secret,timeout=20,allow_agent=False,look_for_keys=False);secret=None
    ssh.get_transport().set_keepalive(20)
    threading.Thread(target=monitor,daemon=True).start();persist()
    print('V27_READY http://127.0.0.1:8772/ map scaffold, independent gate, max one retry; no training',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()

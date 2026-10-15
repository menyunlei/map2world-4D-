"""Actual walking-driven chunk frontier. Resident inference, not one-shot batch."""
from pathlib import Path
_source=Path(__file__).with_name('v25_server.py').read_text().split("\nif __name__=='__main__':")[0]
exec(compile(_source,str(Path(__file__).with_name('v25_server.py')),'exec'),globals())
OUT=JOB/'map_object_world_v1/trellis2_walk_generate_v26';PORT=8771
REMOTE_OUT=REMOTE+'/map_object_world_v1/trellis2_walk_generate_v26'
for i in (2,3):
    d=JOB/f'map_object_world_v1/trellis2_realtime_v25/chunks/stream-0--{i}'
    state['cached'].append(dict(id=f'stream-0--{i}',status='ready',position=[0,0,-SCALE*i],
        asset_url=f'/assets/cache/{i}.glb',sha256=digest(d/'chunk_100m.glb'),historical_immutable=True))
state.update(status='warming',message='常驻模型预热不是生成。实际向前行走才逐块派发；缓存区域不重算。',
             worker={},prefetch_distance=240)
state['budget']=dict(max_generations=3,used=0,max_gpu_seconds=900,max_concurrency=1)
state['chunks']=[dict(id=f'walk-0--{i}',index=i,status='waiting_camera',position=[0,0,-SCALE*i],
                     asset_url=f'/assets/{i}/final.glb',preview_url=f'/assets/{i}/preview.glb') for i in (4,5,6)]
def remote_dir(c):return REMOTE_OUT+'/chunks/'+c['id']

def generate(c,request,resume=False):
    root=remote_dir(c);began=request['epoch']
    try:
        if not resume:
            with ssh.open_sftp() as s:
                try:s.stat(root+'/ASSET.json');raise RuntimeError('Asset already exists before movement')
                except FileNotFoundError:pass
                # Publish only after close: resident reader must never observe
                # the zero-length file briefly created by SFTP open('wx').
                with s.open(root+'/REQUEST.upload','wx') as f:f.write(json.dumps(request))
                s.rename(root+'/REQUEST.upload',root+'/REQUEST.json')
            log('resident_request_sent',chunk=c['id'],request=request)
        while time.time()-began<600:
            with ssh.open_sftp() as s:
                remote=json_remote(s,root+'/STATUS.json')
                worker=json_remote(s,REMOTE_OUT+'/WORKER_STATUS.json')
                if worker['stage'] in ('failed','stopped') and remote.get('status')!='ready':
                    raise RuntimeError('Resident service ended: '+str(worker))
                if remote.get('status')=='failed':raise RuntimeError(remote.get('error'))
                status_name=remote.get('status','queued')
                if status_name=='waiting_camera':status_name='queued'
                mark(c,status_name,stage=remote.get('stage',worker['stage']),elapsed_seconds=time.time()-began)
                if not c.get('preview_sha256'):
                    try:p=json_remote(s,root+'/PREVIEW.json')
                    except FileNotFoundError:p=None
                    if p:
                        transfer(s,c,'geometry_preview.glb',p['sha256']);transfer(s,c,'PREVIEW.json')
                        mark(c,status_name,preview_sha256=p['sha256'],preview_seconds=time.time()-began,preview_ready_epoch=time.time())
                        log('preview_published',chunk=c['id'],seconds=time.time()-began,sha256=p['sha256'])
                if status_name=='ready':
                    complete=json_remote(s,root+'/COMPLETE.json');mark(c,'transferring',stage='download_final')
                    transfer(s,c,'chunk_100m.glb',complete['sha256'])
                    for name in ['COMPLETE.json','PREVIEW.json','STAGES.jsonl','PLAN.json','REQUEST.json','ONE_ATTEMPT.json','LAUNCH.json','RELOAD.json','ASSET.json','LOAD.json']:
                        transfer(s,c,name)
                    import trimesh,numpy as np
                    scene=trimesh.load(chunkdir(c)/'chunk_100m.glb',force='scene',process=False)
                    assert all(len(g.faces)>0 and np.isfinite(g.vertices).all() for g in scene.geometry.values())
                    (chunkdir(c)/'LOCAL_RELOAD.json').write_text(json.dumps(dict(epoch=time.time(),bounds=scene.bounds.tolist(),sha256=complete['sha256'])))
                    mark(c,'ready',stage='cached',sha256=complete['sha256'],elapsed_seconds=time.time()-began,
                         model_generation_seconds=complete['generation_seconds'],resident_pipe_id=complete['resident_pipe_id'],
                         model_reloaded=complete['model_reloaded'],ready_epoch=time.time())
                    log('asset_published',chunk=c['id'],seconds=time.time()-began,**complete)
                    return
            time.sleep(.3)
        raise TimeoutError('600s publication deadline; resident owner separately bounded')
    except Exception as e:
        mark(c,'failed',error=str(e));log('failed',chunk=c['id'],error=str(e))

def camera(data):
    p=data.get('position');d=data.get('forward')
    if not all(isinstance(v,list) and len(v)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and abs(x)<1e6 for x in v) for v in (p,d)):raise ValueError('Invalid camera')
    accepted=[];epoch=time.time()
    with lock:
        old=state.get('camera')
        state['camera']=dict(position=list(p),forward=list(d),epoch=epoch,session=data.get('session'))
        moved_forward=old is not None and p[2]<old['position'][2]-.001
        active=any(c['status'] not in ('waiting_camera','ready','failed') for c in state['chunks'])
        live=state.get('worker',{}).get('stage') not in ('stopped','failed')
        eligible=live and data.get('world')=='native' and moved_forward and p[2]<-160.5 and abs(p[0])<30 and d[2]<-.25
        if eligible and not active and state['budget']['used']<3:
            c=next((c for c in state['chunks'] if c['status']=='waiting_camera'),None)
            loaded={r['id']:r['sha256'] for r in data.get('loaded',[])}
            prev=[x for x in state['chunks'] if c and x['index']==c['index']-1]
            prior_loaded=not prev or prev[0]['status']=='ready' and loaded.get(prev[0]['id'])==prev[0].get('sha256')
            distance=max(0,p[2]-(c['position'][2]+50)) if c else float('inf')
            if c and prior_loaded and distance<=240:
                request=dict(**state['camera'],chunk_id=c['id'],seed=942260+c['index'],
                    trigger='actual_forward_navigation_one_frontier_chunk',distance_to_near_edge=distance,
                    fps=data.get('fps'),loaded=data.get('loaded',[]),previous_generated_chunk_loaded=prior_loaded)
                directory=chunkdir(c);directory.mkdir(parents=True,exist_ok=True)
                with (directory/'ATTEMPT.json').open('x') as f:json.dump(request,f)
                c.update(status='queued',request_epoch=epoch);state['budget']['used']+=1
                log('camera_trigger',chunk=c['id'],request=request)
                threading.Thread(target=generate,args=(c,request),daemon=True).start();accepted=[c['id']]
        persist()
    return dict(accepted=accepted,budget=state['budget'],server_received_epoch=epoch,cached_movement_requires_inference=False)

def worker_monitor():
    while True:
        try:
            with ssh.open_sftp() as s:
                worker=json_remote(s,REMOTE_OUT+'/WORKER_STATUS.json')
                try:owner=json_remote(s,REMOTE_OUT+'/OWNER_LAUNCH_FIX1.json')
                except FileNotFoundError:owner=json_remote(s,REMOTE_OUT+'/OWNER_LAUNCH.json')
                for name in ['MODEL_READY.json','MODEL_READY_FIX1.json','CACHE_HITS.json','GPU_COST.json','GPU_COST_FIX1.json']:
                    try:
                        value=json_remote(s,REMOTE_OUT+'/'+name)
                        tmp=OUT/(name+'.next');tmp.write_text(json.dumps(value));os.replace(tmp,OUT/name)
                    except FileNotFoundError:pass
            with lock:
                state['worker']={**worker,'gpu':owner['gpu'],'hard_deadline_epoch':owner['epoch']+owner['limit_seconds']}
                for c in state['chunks']:c['gpu']=owner['gpu']
                persist()
        except Exception as e:log('worker_monitor_error',error=str(e))
        time.sleep(1)

Base26=Handler
class Handler(Base26):
    def translate_path(self,path):
        p=urlsplit(path).path
        for i in (2,3):
            if p==f'/assets/cache/{i}.glb':return str(JOB/f'map_object_world_v1/trellis2_realtime_v25/chunks/stream-0--{i}/chunk_100m.glb')
        return super().translate_path(path)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'STATE.json').exists():state=json.loads((OUT/'STATE.json').read_text())
    ssh=paramiko.SSHClient();ssh.load_host_keys(str(JOB.parent.parent/'work_known_hosts'));ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('A800 SSH password (memory only): ')
    ssh.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=secret,timeout=20,allow_agent=False,look_for_keys=False);secret=None
    ssh.get_transport().set_keepalive(20)
    for c in state['chunks']:
        if c['status'] not in ('waiting_camera','ready','failed'):
            threading.Thread(target=generate,args=(c,json.loads((chunkdir(c)/'ATTEMPT.json').read_text()),True),daemon=True).start()
    threading.Thread(target=worker_monitor,daemon=True).start();threading.Thread(target=monitor_gpu,daemon=True).start()
    persist();log('server_started',no_generation_from_page_load=True)
    print('V26_READY http://127.0.0.1:8771/ resident models; actual walking gates every chunk',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()

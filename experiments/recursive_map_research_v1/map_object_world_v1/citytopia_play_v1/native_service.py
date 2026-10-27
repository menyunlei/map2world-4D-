"""One on-demand native CityTopia worker; existing model, styles and Map10.

The navigation proxy is not the quality result. Every requested RGB is rendered
by the existing neural background/building generators at the submitted camera.
"""
from pathlib import Path
import gc, json, os, queue, signal, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'citytopia_road_diagnosis_20261027_v1'))
import road_probe as base
PORT=18935
UUIDS=['GPU-e0a56bf9-85ef-c658-67fa-457b76585254','GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa','GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71','GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d']
jobs={}; work=queue.Queue(maxsize=1); state={'phase':'idle','gpu':None}; env=None
ROOT.joinpath('renders').mkdir(exist_ok=True)

def idle_gpu():
    for i,u in enumerate(UUIDS):
        s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
        used,free=map(int,s.strip().split(','))
        if used<256 and free>75000:return i,u
    raise RuntimeError('GPU0–3 busy; no unrelated job interrupted. Try again when idle.')

def worker():
    global env
    gpu,guid=idle_gpu()
    os.environ['CUDA_VISIBLE_DEVICES']=guid
    state['gpu']=gpu
    import torch
    while True:
        try:key=work.get(timeout=120)
        except queue.Empty:
            env=None;gc.collect();torch.cuda.empty_cache();state.update(phase='idle; GPU memory released');continue
        job=jobs[key];start=time.perf_counter()
        try:
            job['status']='loading' if env is None else 'rendering'
            if env is None:
                used,free=map(int,subprocess.check_output(['nvidia-smi','-i',guid,'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
                # A released CUDA context uses a few MiB; never co-schedule a large job.
                if used>1024 or free<75000:raise RuntimeError(f'GPU{gpu} occupied; native view waits for an idle GPU.')
                state.update(phase='loading preserved world',gpu=gpu)
                session=ROOT/'sessions'/uuid.uuid4().hex;session.mkdir(parents=True)
                env=base.setup(session)
            state['phase']='native rendering';job['status']='rendering';job['gpu']=state['gpu']
            native=env['native'];pose=job['pose']
            with torch.inference_mode():
                rays=native.get_voxel_intersection_perspective(env['volume'],pose,env['cfg'],30.,6)
                rgb=native.render_bg((135,240),env['bg'],env['hf'],*rays,None,env['volumes'],env['cfg'],env['inst'])
                visible=native.get_visible_buildings(rays[0][0,:,:,0,0],env['inst'])
                for b in visible.tolist():
                    if b not in env['stats'] or b not in env['styles']:continue
                    fg,mask=native.render_bldg((135,240),env['bldg'],b,env['hf_bldg'],*rays,env['stats'][b],env['styles'][b],env['volumes'],env['cfg'],env['inst'])
                    rgb=rgb*(1-mask)+fg*mask
                pixels=base.rgb(rgb)
                torch.cuda.synchronize()
                base.image(ROOT/'renders'/f'{key}.png',pixels)
                del rays,rgb,pixels
            job.update(status='done',seconds=time.perf_counter()-start,url=f'/renders/{key}.png',
                       method='CityTopia native BG + building generators; saved style codes; no SR/sharpening',resolution=[960,540])
            (ROOT/'renders'/f'{key}.json').write_text(json.dumps(job,indent=2))
            print(json.dumps(job),flush=True);state['phase']='ready; unload after 120s idle'
        except Exception as e:
            job.update(status='error',error=str(e));state['phase']='error'
            print(json.dumps(job),flush=True)
            env=None;gc.collect();torch.cuda.empty_cache()
        finally:work.task_done()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def reply(self,code,value):
        data=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        return self.reply(200,dict(state=state,jobs=jobs))
    def do_POST(self):
        try:
            if any(j['status'] in ['queued','loading','rendering'] for j in jobs.values()):return self.reply(409,{'error':'One native render already running; navigation remains available.'})
            size=int(self.headers.get('Content-Length',0));assert 0<size<8192
            pose=json.loads(self.rfile.read(size))['pose']
            import math
            for kind in ['cam_position','cam_look_at']:
                assert all(math.isfinite(pose[kind][axis]) for axis in ['x','y','z'])
            p=pose['cam_position'];assert 400<p['x']<2600 and 400<p['y']<2600 and 0<p['z']<600
            key=uuid.uuid4().hex[:12];jobs[key]=dict(id=key,pose=pose,status='queued');work.put_nowait(key)
            self.reply(202,jobs[key])
        except Exception as e:self.reply(400,{'error':str(e)})

if __name__=='__main__':
    # No CUDA/model allocations until a user requests a view.
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Thread(target=worker,daemon=True).start()
    print(json.dumps(dict(port=PORT,pid=os.getpid(),model='CityTopia Map10')),flush=True)
    server.serve_forever()

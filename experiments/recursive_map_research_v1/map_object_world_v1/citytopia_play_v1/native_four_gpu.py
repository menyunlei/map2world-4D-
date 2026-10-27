"""Parallelize the existing native 4x4 render tiles, without changing the model.

Each GPU renders one row of four 135x240 tiles with the original 8px context.
Full-scene coordinates, style codes, ray sampler and neural outputs are unchanged.
"""
import inspect, json, multiprocessing as mp, os, queue, subprocess, threading, time, uuid
from http.server import ThreadingHTTPServer
import native_service as one

def tile_worker(rank,inbox,outbox):
    os.environ['CUDA_VISIBLE_DEVICES']=one.UUIDS[rank]
    os.environ['OMP_NUM_THREADS']='4'
    import torch
    session=one.ROOT/'sessions'/uuid.uuid4().hex;session.mkdir(parents=True)
    try:
        env=one.base.setup(session);native=env['native']
        functions={}
        for name in ['render_bg','render_bldg']:
            source=inspect.getsource(getattr(native,name))
            old='for i in range(img_cfg["HEIGHT"] // patch_size[0]):'
            assert source.count(old)==1
            source=source.replace(old,f'for i in [{rank}]:')
            scope=dict(getattr(native,name).__globals__);exec(source,scope)
            functions[name]=scope[name]
        outbox.put(dict(rank=rank,phase='ready'))
        while True:
            item=inbox.get()
            if item is None:break
            key,pose=item;start=time.perf_counter()
            with torch.inference_mode():
                rays=native.get_voxel_intersection_perspective(env['volume'],pose,env['cfg'],30.,6)
                image=functions['render_bg']((135,240),env['bg'],env['hf'],*rays,None,env['volumes'],env['cfg'],env['inst'])
                visible=native.get_visible_buildings(rays[0][0,rank*135:(rank+1)*135,:,0,0],env['inst'])
                for b in visible.tolist():
                    if b not in env['stats'] or b not in env['styles']:continue
                    fg,mask=functions['render_bldg']((135,240),env['bldg'],b,env['hf_bldg'],*rays,env['stats'][b],env['styles'][b],env['volumes'],env['cfg'],env['inst'])
                    image=image*(1-mask)+fg*mask
                torch.cuda.synchronize()
                pixels=one.base.rgb(image)[rank*135:(rank+1)*135].copy()
                outbox.put(dict(rank=rank,phase='done',id=key,pixels=pixels,seconds=time.perf_counter()-start))
                del rays,image,pixels
    except Exception as e:
        import traceback
        outbox.put(dict(rank=rank,phase='error',error=str(e),traceback=traceback.format_exc()))

def four_worker():
    import numpy as np
    ctx=mp.get_context('spawn');pool=[];inboxes=[];outbox=None
    def release():
        nonlocal pool,inboxes,outbox
        for q in inboxes:q.put(None)
        for p in pool:
            p.join(timeout=5)
            if p.is_alive():p.terminate();p.join(timeout=5)
        pool=[];inboxes=[];outbox=None
        one.state.update(phase='idle; GPU0–3 memory released',gpus=[0,1,2,3])
    while True:
        try:key=one.work.get(timeout=120)
        except queue.Empty:
            if pool:release()
            continue
        job=one.jobs[key];start=time.perf_counter()
        try:
            job.update(status='loading' if not pool else 'rendering',gpus=[0,1,2,3],tiles_done=[])
            if not pool:
                for rank,guid in enumerate(one.UUIDS):
                    used,free=map(int,subprocess.check_output(['nvidia-smi','-i',guid,'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    if used>256 or free<75000:raise RuntimeError(f'GPU{rank} busy; no unrelated job interrupted')
                outbox=ctx.Queue();one.state.update(phase='GPU0–3 loading same saved world',gpus=[0,1,2,3])
                for rank in range(4):
                    q=ctx.Queue();p=ctx.Process(target=tile_worker,args=(rank,q,outbox),daemon=True);p.start();pool.append(p);inboxes.append(q)
                for _ in range(4):
                    msg=outbox.get(timeout=120)
                    if msg['phase']=='error':raise RuntimeError(msg['error'])
            one.state['phase']='GPU0–3 native tiles';job['status']='rendering'
            for q in inboxes:q.put((key,job['pose']))
            strips={};times={}
            for _ in range(4):
                msg=outbox.get(timeout=120)
                if msg['phase']=='error':raise RuntimeError(msg['error'])
                strips[msg['rank']]=msg['pixels'];times[msg['rank']]=msg['seconds'];job['tiles_done']=sorted(strips)
            one.base.image(one.ROOT/'renders'/f'{key}.png',np.concatenate([strips[i] for i in range(4)],axis=0))
            job.update(status='done',seconds=time.perf_counter()-start,per_gpu_seconds=times,url=f'/renders/{key}.png',
                method='4 A800 / one pose / original native padded tiles; no SR or model change',resolution=[960,540])
            (one.ROOT/'renders'/f'{key}.json').write_text(json.dumps(job,indent=2))
            print(json.dumps(job),flush=True);one.state['phase']='ready; unload all four after 120s idle'
        except Exception as e:
            job.update(status='error',error=str(e));print(json.dumps(job),flush=True);release()
        finally:one.work.task_done()

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',one.PORT),one.Handler)
    one.state.update(phase='four-GPU ready; no allocations until render',gpus=[0,1,2,3])
    threading.Thread(target=four_worker,daemon=True).start()
    print(json.dumps(dict(pid=os.getpid(),port=one.PORT,mode='native four GPU tiles')),flush=True)
    server.serve_forever()

"""Native padded tiles, cached static encoders, and work stealing across GPU0-3.

No weights, density, ray samples, CNN context, or output resolution are changed.
This accelerates the saved world renderer; it is not new generative refinement.
"""
import json
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from http.server import ThreadingHTTPServer

import native_service as one


class TileEngine:
    def __init__(self, rank):
        import torch
        import torchvision
        self.torch = torch
        session = one.ROOT/'sessions'/uuid.uuid4().hex
        session.mkdir(parents=True)
        self.env = e = one.base.setup(session)
        self.native = e['native']
        self.bg, self.bldg = e['bg'].module, e['bldg'].module
        # Caching is valid only for the fixed, evaluation-mode world conditions.
        assert not self.bg.training and not self.bldg.training
        assert self.bg.cfg.STYLE_DIM is None
        assert not (self.bg.cfg.SKY_ENABLED and self.bg.cfg.SKY_GLOBAL_AVGPOOL)
        margin = e['volumes']['BLDG']
        t = time.perf_counter()
        self.bg_features = self.bg.encoder(e['hf'][:, :, margin:-margin, margin:-margin])
        torch.cuda.synchronize()
        self.initial_encode_seconds = time.perf_counter()-t
        self.features = {}
        self.stats = {}
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2, 2))
        # No source uses these mutable inputs after their encoding has been cached.
        self.rank = rank

    def begin(self, pose):
        e = self.env
        self.rays = self.native.get_voxel_intersection_perspective(e['volume'], pose, e['cfg'], 30., 6)
        self.bg_ids = self.native.get_background_voxel_ids(self.rays[0], e['inst'])

    def forward(self, model, features, ids, depths, dirs, origin, stats=None, style=None):
        # Same native forward after the camera-independent encoder.
        values, opacity = model._forward_perpix(features, ids, depths, dirs, origin, style, stats, True)
        return model._forward_global(values, style), opacity

    def building_features(self, building):
        if building not in self.features:
            import numpy as np
            e = self.env
            stats = e['stats'][building]
            half = e['volumes']['BLDG']//2
            cx = e['volumes']['EXT']//2 + int(stats[1])
            cy = e['volumes']['EXT']//2 + int(stats[0])
            # The original function made an unused full-layout deepcopy here.
            crop = e['hf_bldg'][:, :, cy-half:cy+half, cx-half:cx+half]
            self.features[building] = self.bldg.encoder(crop)
            self.stats[building] = self.torch.from_numpy(np.array(stats)).unsqueeze(0)
        return self.features[building], self.stats[building]

    def tile(self, index):
        torch, e, n = self.torch, self.env, self.native
        sy, sx = (index//4)*135, (index%4)*240
        ey, ex = sy+135, sx+240
        psx, pex, psy, pey = n.get_pad_img_bbox(sx, ex, sy, ey, 540, 960, 8)
        ids = self.rays[0][:, psy:pey, psx:pex]
        depths = self.rays[1][:, psy:pey, psx:pex]
        dirs = self.rays[2][:, psy:pey, psx:pex]
        origin = self.rays[3]
        bgids = self.bg_ids[:, psy:pey, psx:pex]
        bounds = (sx, ex, sy, ey, psx, pex, psy, pey, 8)
        output, _ = self.forward(self.bg, self.bg_features, bgids, depths, dirs, origin)
        road = (bgids[:, None, :, :, 0, 0] == e['inst']['ROAD_CID']).repeat(1, 3, 1, 1).float()
        output = self.blurrer(output)*road + output*(1-road)
        result = n.get_img_without_pad(output, *bounds)
        core = self.rays[0][0, sy:ey, sx:ex, 0, 0]
        buildings = n.get_visible_buildings(core, e['inst']).tolist()
        for building in buildings:
            if building not in e['stats'] or building not in e['styles']:
                continue
            bid = ids.clone()
            facade = ids == building
            roof = ids == building + e['inst']['ROOF_OFFSET']
            bid[~(facade | roof)] = 0
            bid[facade] = e['inst']['FACADE_CID']
            bid[roof] = e['inst']['ROOF_CID']
            bdirs = dirs.clone()
            bdirs[bid[..., 0, 0] == 0] = 0
            features, stats = self.building_features(building)
            output, _ = self.forward(self.bldg, features, bid, depths, bdirs, origin, stats, e['styles'][building])
            mask = ((core == building) | (core == building+e['inst']['ROOF_OFFSET']))[None, None].float()
            result = result*(1-mask) + n.get_img_without_pad(output, *bounds)*mask
        pixels = one.base.rgb(result).copy()
        return pixels, len(buildings)


def worker(rank, inbox, tasks, outbox):
    os.environ['CUDA_VISIBLE_DEVICES'] = one.UUIDS[rank]
    os.environ['OMP_NUM_THREADS'] = '4'
    try:
        import torch
        torch.set_num_threads(4)
        with torch.inference_mode():
            engine = TileEngine(rank)
            outbox.put(dict(rank=rank, phase='ready', encode_seconds=engine.initial_encode_seconds))
            while True:
                item = inbox.get()
                if item is None:
                    break
                key, pose = item
                started = time.perf_counter()
                engine.begin(pose)
                torch.cuda.synchronize()
                ray_seconds = time.perf_counter()-started
                count = 0
                busy = 0.
                while True:
                    index = tasks.get()
                    if index is None:
                        break
                    t = time.perf_counter()
                    pixels, buildings = engine.tile(index)
                    duration = time.perf_counter()-t
                    busy += duration
                    count += 1
                    outbox.put(dict(id=key, rank=rank, phase='tile', index=index, pixels=pixels,
                                    seconds=duration, buildings=buildings))
                outbox.put(dict(id=key, rank=rank, phase='finished', tiles=count, busy_seconds=busy,
                                ray_seconds=ray_seconds, peak_gib=torch.cuda.max_memory_allocated()/2**30))
                engine.rays = None
                engine.bg_ids = None
    except Exception as exc:
        import traceback
        outbox.put(dict(rank=rank, phase='error', error=str(exc), traceback=traceback.format_exc()))


class Pool:
    def __init__(self):
        self.ctx = mp.get_context('spawn')
        self.processes = []
        self.inboxes = []

    def release(self):
        for q in self.inboxes:
            q.put(None)
        for p in self.processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
        self.processes = []
        self.inboxes = []
        one.state.update(phase='idle; four GPU allocations released', gpus=[0, 1, 2, 3])

    def start(self):
        for rank, guid in enumerate(one.UUIDS):
            used, free = map(int, subprocess.check_output(['nvidia-smi', '-i', guid,
                '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            if used > 256 or free < 75000:
                raise RuntimeError(f'GPU{rank} occupied; no unrelated process interrupted')
        self.tasks = self.ctx.Queue()
        self.outbox = self.ctx.Queue()
        for rank in range(4):
            q = self.ctx.Queue()
            p = self.ctx.Process(target=worker, args=(rank, q, self.tasks, self.outbox), daemon=True)
            p.start()
            self.processes.append(p)
            self.inboxes.append(q)
        for _ in range(4):
            msg = self.outbox.get(timeout=120)
            if msg['phase'] != 'ready':
                raise RuntimeError(msg)


pool = Pool()


def coordinator():
    import numpy as np
    while True:
        try:
            key = one.work.get(timeout=120)
        except queue.Empty:
            if pool.processes:
                pool.release()
            continue
        job = one.jobs[key]
        start = time.perf_counter()
        try:
            job.update(status='loading' if not pool.processes else 'rendering', gpus=[0, 1, 2, 3], tiles_done=[], tiles_total=16)
            if not pool.processes:
                one.state['phase'] = 'loading fixed world / caching static encoders'
                pool.start()
            job['load_seconds'] = time.perf_counter()-start
            job['status'] = 'rendering'
            one.state['phase'] = 'four-GPU dynamic native tiles'
            render_start = time.perf_counter()
            for q in pool.inboxes:
                q.put((key, job['pose']))
            # Start expensive central building tiles first; workers pull more as available.
            for index in [5, 6, 9, 10, 4, 7, 8, 11, 0, 1, 2, 3, 12, 13, 14, 15]:
                pool.tasks.put(index)
            for _ in range(4):
                pool.tasks.put(None)
            pixels = np.zeros((540, 960, 3), np.uint8)
            tiles = {}
            ranks = {}
            while len(ranks) < 4:
                msg = pool.outbox.get(timeout=120)
                if msg['phase'] == 'error':
                    raise RuntimeError(msg)
                assert msg['id'] == key
                if msg['phase'] == 'tile':
                    index = msg['index']
                    y, x = (index//4)*135, (index%4)*240
                    pixels[y:y+135, x:x+240] = msg.pop('pixels')
                    tiles[index] = msg
                    job['tiles_done'] = sorted(tiles)
                elif msg['phase'] == 'finished':
                    ranks[msg['rank']] = msg
            assert len(tiles) == 16
            render_seconds = time.perf_counter()-render_start
            one.base.image(one.ROOT/'renders'/f'{key}.png', pixels)
            job.update(status='done', seconds=time.perf_counter()-start, render_seconds=render_seconds,
                       per_gpu=ranks, tile_timings=tiles, url=f'/renders/{key}.png', resolution=[960, 540],
                       method='native fixed world; cached layout encoders; 16 original padded tiles / dynamic four-GPU queue; no SR')
            (one.ROOT/'renders'/f'{key}.json').write_text(json.dumps(job, indent=2))
            print(json.dumps(job), flush=True)
            one.state['phase'] = 'ready; static encoders reused; release after 120s idle'
        except Exception as exc:
            job.update(status='error', error=str(exc))
            print(json.dumps(job), flush=True)
            pool.release()
        finally:
            one.work.task_done()


if __name__ == '__main__':
    def shutdown(*unused):
        pool.release()
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    server = ThreadingHTTPServer(('127.0.0.1', one.PORT), one.Handler)
    one.state.update(phase='balanced renderer ready; load on demand', gpus=[0, 1, 2, 3])
    threading.Thread(target=coordinator, daemon=True).start()
    print(json.dumps(dict(pid=os.getpid(), port=one.PORT, mode='native balanced v2')), flush=True)
    server.serve_forever()

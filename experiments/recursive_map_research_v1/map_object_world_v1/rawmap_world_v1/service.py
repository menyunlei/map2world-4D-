"""Raw-map candidate behind the existing CityTopia player (port 18937).

Same request/response shape as citytopia_play_v1/native_service.py, so the
player and SSH bridge treat it as a second model. The world is generated once
from map.png (cold start, timed); each camera request only renders the saved
Gaussians. Outputs go to citytopia_play_v1/renders like native frames.

  python service.py --checkpoint RUN/checkpoint.pt --scene SCENES/map10_street_b \
      [--map /path/edited_map.png] [--seed -1] [--backbone hf:/path/dinov3]
"""
import argparse
import json
import math
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from data import Scene, load_map
from model import world_code
from render import Camera, pose_to_world
from train import load_checkpoint, render_view, require_idle_gpu

ROOT = Path(__file__).resolve().parent
RENDERS = ROOT.parent / 'citytopia_play_v1' / 'renders'
PORT = 18937
jobs, work = {}, queue.Queue(maxsize=1)
state = {'phase': 'idle', 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '?'), 'model': 'raw-map candidate'}


def worker(a):
    device = torch.device('cuda')
    model, margs, ck = load_checkpoint(a.checkpoint, device, a.backbone)
    scene = Scene(a.scene)
    style = world_code(a.seed, margs.style_dim, device)
    map_path = a.map or scene.folder / 'map.png'
    view = scene.views[0]
    state.update(phase='generating world from map image', checkpoint=str(a.checkpoint), scene=scene.name,
                 split=scene.split, map=str(map_path), seed=a.seed)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode():
        world = model(load_map(map_path, margs.map_size).to(device), scene.region, scene.map_bounds, style)
    torch.cuda.synchronize()
    state.update(phase='ready; world generated once, camera requests render it',
                 cold_start_seconds=time.perf_counter() - t, gaussians=int(world['means'].shape[0]),
                 opaque_fraction=float((world['opacities'] > .1).float().mean()))
    while True:
        key = work.get()
        job = jobs[key]
        try:
            job['status'] = 'rendering'
            origin, look = pose_to_world(job['pose'])
            cam = Camera(origin, look, view['height'], view['width'], view['focal'],
                         view.get('pixel_offset', 0.), view.get('side_sign', 1.))
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.inference_mode():
                rgb, _, _ = render_view(model, world, style, cam)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            pixels = (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()
            Image.fromarray(pixels).save(RENDERS / f'{key}.png')
            job.update(status='done', seconds=seconds, render_seconds=seconds, url=f'/renders/{key}.png',
                       resolution=[cam.width, cam.height], model='raw-map candidate',
                       method='map.png -> generated Gaussians (saved) -> splat at camera; no CityDreamer4D call')
            (RENDERS / f'{key}.json').write_text(json.dumps(job, indent=2))
        except Exception as e:
            job.update(status='error', error=str(e))
        finally:
            print(json.dumps({k: v for k, v in job.items() if k != 'pose'}), flush=True)
            work.task_done()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def reply(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        return self.reply(200, dict(state=state, jobs=jobs))

    def do_POST(self):
        try:
            if any(j['status'] in ['queued', 'rendering'] for j in jobs.values()):
                return self.reply(409, {'error': 'One candidate render already running.'})
            if not state['phase'].startswith('ready'):
                return self.reply(409, {'error': 'Candidate world is still being generated: ' + state['phase']})
            size = int(self.headers.get('Content-Length', 0)); assert 0 < size < 8192
            pose = json.loads(self.rfile.read(size))['pose']
            for kind in ['cam_position', 'cam_look_at']:
                assert all(math.isfinite(pose[kind][axis]) for axis in 'xyz')
            key = 'cand-' + uuid.uuid4().hex[:12]
            jobs[key] = dict(id=key, pose=pose, status='queued', model='raw-map candidate')
            work.put_nowait(key)
            self.reply(202, jobs[key])
        except Exception as e:
            self.reply(400, {'error': str(e)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--map', type=Path, help='alternative map image with the same bounds (edit / restyle tests)')
    p.add_argument('--seed', type=int, default=-1, help='fixed world code; -1 = prior mean')
    p.add_argument('--backbone', help='override the stored backbone path on this machine')
    a = p.parse_args()
    require_idle_gpu()
    RENDERS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, args=(a,), daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(json.dumps(dict(port=PORT, pid=os.getpid(), model='raw-map candidate')), flush=True)
    server.serve_forever()

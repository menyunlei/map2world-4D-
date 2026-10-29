"""Train the raw-map world student on map-image / calibrated-view pairs.

The student sees map.png only. Teacher depth (when present) is a training target,
never an input. Bounded by --steps and --max-seconds; refuses a busy GPU.
"""
import argparse
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import check_split, load_scenes
from model import RawMapWorld, build_backbone, composite, world_code
from render import Camera, render


def ssim(a, b, window=11, sigma=1.5):
    """a, b [H, W, 3] in [0, 1]."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-(torch.arange(window, device=a.device) - window // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None]).expand(3, 1, window, window)
    f = lambda t: F.conv2d(t, k, padding=window // 2, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))


def render_view(model, world, style, camera, backend='auto'):
    rgb, alpha, depth = render(world, camera, backend)
    dirs = camera.ray_directions().to(rgb.device)
    return composite(rgb, alpha, model.sky(dirs, style)), alpha, depth


def view_loss(pred, alpha, depth, target, target_depth, w):
    loss = F.l1_loss(pred, target) + w['ssim'] * (1 - ssim(pred, target))
    parts = {'rgb': loss.item()}
    if target_depth is not None:
        surface = torch.isfinite(target_depth)
        coverage = F.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), surface.float())
        loss = loss + w['alpha'] * coverage
        parts['alpha'] = float(coverage)
        valid = surface & (alpha > .5)
        if valid.any():
            d = (torch.log(depth[valid].clamp_min(1e-3)) - torch.log(target_depth[valid])).abs().mean()
            loss = loss + w['depth'] * d
            parts['depth'] = float(d)
    return loss, parts


def require_idle_gpu():
    """Same policy as the existing services: never share or preempt a GPU."""
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-'):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one authorized physical GPU UUID')
    used, free = map(int, subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=memory.used,memory.free',
                                                   '--format=csv,noheader,nounits'], text=True).strip().split(','))
    if used > 256 or free < 75000:
        raise RuntimeError(f'GPU {uuid} occupied ({used} MiB used); no job interrupted')


def build(args):
    return RawMapWorld(build_backbone(args.backbone), dim=args.dim, style_dim=args.style_dim, cell=args.cell,
                       z_tokens=args.z_tokens, gaussians_per_token=args.k)


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def evaluate(model, scenes, args, device, limit=8):
    model.eval()
    style = world_code(-1, args.style_dim, device)
    rows = []
    with torch.no_grad():
        for scene in scenes:
            world = model(scene.map_image(args.map_size).to(device), scene.region, scene.map_bounds, style)
            for i in range(min(limit, len(scene.views))):
                target, _ = scene.target(i, args.scale)
                pred, _, _ = render_view(model, world, style, Camera.from_view(scene.views[i], args.scale))
                rows.append(psnr(pred.clamp(0, 1), target.to(device)))
    model.train()
    return sum(rows) / max(len(rows), 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--backbone', default='tiny', help='"tiny" (tests) or "hf:<local DINOv2/DINOv3 path>"')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--max-seconds', type=int, default=6 * 3600)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--scale', type=float, default=.5, help='render scale of the 960x540 teacher views')
    p.add_argument('--views-per-step', type=int, default=2)
    p.add_argument('--map-size', type=int, default=512)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--style-dim', type=int, default=64)
    p.add_argument('--cell', type=float, default=4.)
    p.add_argument('--z-tokens', type=int, default=24)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--w-ssim', type=float, default=.2)
    p.add_argument('--w-depth', type=float, default=.1)
    p.add_argument('--w-alpha', type=float, default=.05)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=953101)
    p.add_argument('--cpu', action='store_true', help='CPU tests only')
    args = p.parse_args(argv)

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device('cpu' if args.cpu else 'cuda')
    if not args.cpu:
        require_idle_gpu()
    scenes = load_scenes(args.scenes)
    splits = check_split(scenes)
    train = [s for s in scenes if s.split == 'train']
    val = [s for s in scenes if s.split == 'val']
    if not train:
        raise ValueError('no training scenes')
    args.out.mkdir(parents=True, exist_ok=False)
    model = build(args).to(device)
    codes = torch.nn.Parameter(torch.randn(len(train), args.style_dim, device=device) * .01)
    params = [q for q in model.parameters() if q.requires_grad] + [codes]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    weights = {'ssim': args.w_ssim, 'depth': args.w_depth, 'alpha': args.w_alpha}
    (args.out / 'CONFIG.json').write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, 'splits': splits,
        'student_inputs': ['map.png'], 'device': str(device)}, indent=2))

    def deadline(*unused):
        raise TimeoutError('bounded training deadline')
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, deadline); signal.alarm(args.max_seconds)
    log = (args.out / 'LOG.jsonl').open('a')
    started, history = time.perf_counter(), []
    try:
        for step in range(1, args.steps + 1):
            si = random.randrange(len(train))
            scene = train[si]
            style = codes[si]
            world = model(scene.map_image(args.map_size).to(device), scene.region, scene.map_bounds, style)
            loss, parts = 0., {}
            for i in random.sample(range(len(scene.views)), min(args.views_per_step, len(scene.views))):
                target, target_depth = scene.target(i, args.scale)
                pred, alpha, depth = render_view(model, world, style, Camera.from_view(scene.views[i], args.scale))
                l, parts = view_loss(pred, alpha, depth, target.to(device),
                                     None if target_depth is None else target_depth.to(device), weights)
                loss = loss + l
            loss = loss / args.views_per_step + 1e-4 * (codes[si] ** 2).sum() + 1e-4 * world['opacities'].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step(); sched.step()
            history.append(float(loss))
            if step % 50 == 0 or step == 1:
                row = {'step': step, 'loss': float(loss), **parts, 'seconds': time.perf_counter() - started,
                       'gaussians': int(world['means'].shape[0]),
                       'opaque_fraction': float((world['opacities'] > .1).float().mean())}
                log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
            if val and step % args.eval_every == 0:
                row = {'step': step, 'val_psnr_prior_style': evaluate(model, val, args, device)}
                log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
    finally:
        log.close()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        torch.save({'model': trainable_state(model), 'codes': codes.detach().cpu(), 'args': vars(args),
                    'train_scenes': [s.name for s in train], 'steps_done': len(history),
                    'student_inputs': ['map.png']}, args.out / 'checkpoint.pt')
    return history


def load_checkpoint(path, device, backbone=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ck['args'])
    if backbone:
        args.backbone = backbone
    model = build(args)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    if unexpected or set(missing) - frozen - {n for n, _ in model.named_buffers()}:
        raise RuntimeError(f'checkpoint mismatch: missing={missing} unexpected={unexpected}')
    return model.to(device).eval(), args, ck


if __name__ == '__main__':
    main()

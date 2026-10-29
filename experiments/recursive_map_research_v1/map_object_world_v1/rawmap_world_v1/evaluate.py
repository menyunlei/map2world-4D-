"""Held-out evaluation and map-dependence controls for one checkpoint.

Same cameras and fixed world code for every condition:
  real      the scene's map.png
  blank     map replaced by its mean colour        (no layout information)
  shuffled  32x32-pixel tiles randomly permuted    (local texture, wrong layout)
  restyled  --restyled-map, else an automatic hue/contrast change (weak proxy
            for "same layout, different map styling"; reported as such)
  edited    --edited-map with --edited-scene views of the expected result
If generation depends on the map, real must beat blank and shuffled clearly and
restyled should stay close to real. Timing: cold start (encode + generate) and
per-camera render are measured separately.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import Scene, load_map, load_scenes
from model import world_code
from render import Camera
from train import load_checkpoint, psnr, render_view, ssim


def blank(m):
    return m.mean((2, 3), keepdim=True).expand_as(m).clone()


def shuffled(m, tile=32, seed=0):
    _, c, h, w = m.shape
    t = m.reshape(1, c, h // tile, tile, w // tile, tile).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, tile, tile)
    t = t[torch.randperm(t.shape[0], generator=torch.Generator().manual_seed(seed))]
    return t.reshape(1, h // tile, w // tile, c, tile, tile).permute(0, 3, 1, 4, 2, 5).reshape(1, c, h, w)


def auto_restyle(m):
    rgb = m[0].permute(1, 2, 0)
    rotated = rgb[..., [1, 2, 0]]                        # hue rotation by channel cycling
    return ((rotated - .5) * 1.3 + .5).clamp(0, 1).permute(2, 0, 1)[None]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def run_condition(model, args, scene, image, style, device, views, scale):
    sync(device); t = time.perf_counter()
    with torch.inference_mode():
        world = model(image.to(device), scene.region, scene.map_bounds, style)
    sync(device); cold = time.perf_counter() - t
    frames, times = [], []
    for i in views:
        cam = Camera.from_view(scene.views[i], scale)
        sync(device); t = time.perf_counter()
        with torch.inference_mode():
            rgb, alpha, depth = render_view(model, world, style, cam)
        sync(device); times.append(time.perf_counter() - t)
        frames.append((rgb.clamp(0, 1), alpha, depth))
    return world, frames, cold, times


def score(frames, scene, views, scale, device, lpips_fn=None):
    rows = []
    for (rgb, alpha, depth), i in zip(frames, views):
        target, tdepth = scene.target(i, scale)
        target = target.to(device)
        row = {'view': scene.views[i]['tag'], 'psnr': psnr(rgb, target), 'ssim': float(ssim(rgb, target))}
        if lpips_fn is not None:
            row['lpips'] = float(lpips_fn(rgb.permute(2, 0, 1)[None] * 2 - 1, target.permute(2, 0, 1)[None] * 2 - 1))
        if tdepth is not None:
            tdepth = tdepth.to(device)
            surface = torch.isfinite(tdepth)
            pred = alpha > .5
            row['coverage_iou'] = float((surface & pred).sum() / (surface | pred).sum().clamp_min(1))
            both = surface & pred
            if both.any():
                row['depth_abs_rel'] = float(((depth[both] - tdepth[both]).abs() / tdepth[both]).mean())
        rows.append(row)
    keys = rows[0].keys() - {'view'}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--scenes', required=True, type=Path)
    p.add_argument('--split', default='test')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--views', type=int, default=12)
    p.add_argument('--restyled-map', type=Path)
    p.add_argument('--edited-map', type=Path)
    p.add_argument('--edited-scene', type=Path, help='views of the expected world for --edited-map')
    p.add_argument('--backbone')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    device = torch.device('cpu' if a.cpu else 'cuda')
    model, margs, ck = load_checkpoint(a.checkpoint, device, a.backbone)
    style = world_code(a.seed, margs.style_dim, device)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except Exception:
        lpips_fn = None
    a.out.mkdir(parents=True, exist_ok=False)
    report = {'checkpoint': str(a.checkpoint), 'train_scenes': ck['train_scenes'], 'seed': a.seed, 'scenes': {}}
    for scene in load_scenes(a.scenes, a.split):
        if scene.name in ck['train_scenes']:
            raise ValueError(f'{scene.name} was used in training; not a held-out evaluation')
        views = list(range(0, len(scene.views), max(1, len(scene.views) // a.views)))[:a.views]
        real = scene.map_image(margs.map_size)
        conditions = {'real': real, 'blank': blank(real), 'shuffled': shuffled(real),
                      'restyled': load_map(a.restyled_map, margs.map_size) if a.restyled_map else auto_restyle(real)}
        result, panels = {'restyle_kind': 'supplied map' if a.restyled_map else 'automatic recolour (weak proxy)'}, {}
        for name, image in conditions.items():
            world, frames, cold, times = run_condition(model, margs, scene, image, style, device, views, a.scale)
            summary, rows = score(frames, scene, views, a.scale, device, lpips_fn)
            result[name] = {**summary, 'cold_start_seconds': cold, 'render_seconds_mean': float(np.mean(times)),
                            'gaussians': int(world['means'].shape[0]), 'per_view': rows}
            panels[name] = frames
        result['map_dependence_psnr'] = {k: result['real']['psnr'] - result[k]['psnr'] for k in ['blank', 'shuffled', 'restyled']}
        if a.edited_map and a.edited_scene:
            target_scene = Scene(a.edited_scene)
            ev = list(range(min(a.views, len(target_scene.views))))
            _, frames, _, _ = run_condition(model, margs, target_scene, load_map(a.edited_map, margs.map_size), style, device, ev, a.scale)
            result['edited'], _ = score(frames, target_scene, ev, a.scale, device, lpips_fn)
            _, frames, _, _ = run_condition(model, margs, target_scene, real, style, device, ev, a.scale)
            result['edited_unedited_map_baseline'], _ = score(frames, target_scene, ev, a.scale, device, lpips_fn)
        report['scenes'][scene.name] = result
        folder = a.out / scene.name
        folder.mkdir()
        for k, i in enumerate(views):
            target, _ = scene.target(i, a.scale)
            row = [target.numpy()] + [panels[n][k][0].cpu().numpy() for n in conditions]
            Image.fromarray((np.concatenate(row, 1) * 255).round().astype(np.uint8)).save(folder / f'{scene.views[i]["tag"]}.png')
    report['panel_order'] = ['teacher'] + list(conditions)
    (a.out / 'REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({s: {'psnr_real': r['real']['psnr'], 'dependence': r['map_dependence_psnr']}
                      for s, r in report['scenes'].items()}, indent=2))


if __name__ == '__main__':
    main()

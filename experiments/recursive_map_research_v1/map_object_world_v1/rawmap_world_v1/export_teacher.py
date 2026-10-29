"""Offline teacher pairs from the preserved CityDreamer4D Map10 world.

Runs in the CityDreamer4D environment on one idle GPU. The teacher uses its own
structured inputs (semantic layout, heights, instances, styles); the exported
scene contains only what the student may read (map.png) plus targets (views,
depth). The map image is supplied by the user: a semantic-colour rendering of
the layout would leak the teacher's inputs and is refused by design.

  python export_teacher.py --name map10_street_a --split train \
      --map-image /path/map10_rgb.png --map-bounds r0,r1,c0,c1 --bounds-frame projection \
      --region r0,r1,c0,c1,z0,z1 --path r,c;r,c;r,c --spacing 24 --out ../rawmap_scenes

Coordinates are (row, col) in the frame named by --bounds-frame ('volume' is the
CityDreamer pose frame; 'projection' is pixel indices of Map10's TD_HF/INS_BEV).
Teacher depth is the first-voxel entry of the extruded map volume: a proxy of the
teacher's geometry, not learned density or measured ground truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'citytopia_road_diagnosis_20261027_v1'))
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def floats(text, n):
    v = [float(x) for x in text.split(',')]
    assert len(v) == n, text
    return v


def path_points(spec, spacing):
    pts = np.array([floats(p, 2) for p in spec.split(';')])
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) // spacing))
        for t in range(n):
            out.append((a + (b - a) * t / n, (b - a) / np.linalg.norm(b - a)))
    out.append((pts[-1], (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])))
    return out


def ground_z(volume, r, c):
    col = volume[int(round(r)), int(round(c))]
    solid = (col != 0).nonzero()
    return float(solid.max()) + 1 if len(solid) else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--map-image', required=True, type=Path)
    p.add_argument('--map-bounds', required=True, help='r0,r1,c0,c1 covered by the map image')
    p.add_argument('--bounds-frame', choices=['volume', 'projection'], required=True)
    p.add_argument('--region', required=True, help='r0,r1,c0,c1,z0,z1 to generate (same frame; z in volume units)')
    p.add_argument('--path', required=True, help='street polyline "r,c;r,c;..." (same frame)')
    p.add_argument('--spacing', type=float, default=24., help='units between path cameras (4 units/m)')
    p.add_argument('--eye', type=float, default=1.7, help='camera height above road, metres')
    p.add_argument('--yaws', default='0,45,-45,90,-90,180', help='degrees relative to the street direction')
    p.add_argument('--lateral', default='0,-12,12', help='sideways camera offsets in units (close facade approach)')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()

    import torch
    import road_probe as probe
    from render import Camera, fov_focal, match_native_rays, world_to_pose
    from train import require_idle_gpu
    require_idle_gpu()
    # probe.setup() changes directory into the CityDreamer4D checkout: resolve first.
    out = a.out.resolve() / a.name
    a.map_image = a.map_image.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'views').mkdir()
    started = time.perf_counter()
    session = out / f'teacher_setup_{uuid.uuid4().hex[:8]}'
    session.mkdir()
    env = probe.setup(session)
    native = env['native']
    shift = np.zeros(2)
    if a.bounds_frame == 'projection':
        projections, _ = native.get_city_sample_layout(str(probe.base.SPATIAL), [10, 30000])
        cy, cx = np.array(projections['REST']['TD_HF'].shape) // 2
        shift = np.array([cy - env['volumes']['LAYOUT'] // 2, cx - env['volumes']['LAYOUT'] // 2])
    mb = np.array(floats(a.map_bounds, 4)) - np.repeat(shift, 2)
    rg = floats(a.region, 6)
    rg[:4] = list(np.array(rg[:4]) - np.repeat(shift, 2))
    from PIL import Image
    image = Image.open(a.map_image)
    if image.mode not in ('RGB', 'RGBA'):
        raise ValueError('map image must be RGB; label/palette images are semantic maps, not raw maps')
    image.convert('RGB').save(out / 'map.png')

    H, W = env['cfg']['HEIGHT'], env['cfg']['WIDTH']
    focal = fov_focal(H, env['cfg']['VFOV'])
    yaws = [math.radians(float(y)) for y in a.yaws.split(',')]
    laterals = [float(v) for v in a.lateral.split(',')]
    views, convention = [], None
    with torch.inference_mode():
        for n, (pt, d) in enumerate(path_points(a.path, a.spacing)):
            pt = pt - shift
            normal = np.array([-d[1], d[0]])
            for lat in laterals:
                r, c = pt + normal * lat
                eye_z = ground_z(env['volume'], r, c) + a.eye * 4
                for yaw in yaws:
                    dr = d[0] * math.cos(yaw) - d[1] * math.sin(yaw)
                    dc = d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
                    origin = np.array([r, c, eye_z])
                    look = origin + np.array([dr * 40, dc * 40, -2.])
                    pose = world_to_pose(origin, look)
                    rays = native.get_voxel_intersection_perspective(env['volume'], pose, env['cfg'], env['cfg']['VFOV'], 6)
                    if convention is None:
                        off, sign, cos = match_native_rays(rays[2][0, :, :, 0], torch.tensor(origin, dtype=torch.float32),
                                                           torch.tensor(look, dtype=torch.float32), H, W, focal)
                        if cos < .99999:
                            raise RuntimeError(f'camera convention not reproduced (min cosine {cos})')
                        convention = dict(pixel_offset=off, side_sign=sign, min_cosine=cos)
                    rgb = native.render_bg((135, 240), env['bg'], env['hf'], *rays, None, env['volumes'], env['cfg'], env['inst'])
                    for b in native.get_visible_buildings(rays[0][0, :, :, 0, 0], env['inst']).tolist():
                        if b in env['stats'] and b in env['styles']:
                            fg, mask = native.render_bldg((135, 240), env['bldg'], b, env['hf_bldg'], *rays, env['stats'][b],
                                                          env['styles'][b], env['volumes'], env['cfg'], env['inst'])
                            rgb = rgb * (1 - mask) + fg * mask
                    tag = f'{n:03d}_lat{lat:+.0f}_yaw{math.degrees(yaw):+.0f}'
                    Image.fromarray(probe.rgb(rgb)).save(out / 'views' / f'{tag}.png')
                    cam = Camera(origin, look, H, W, focal, convention['pixel_offset'], convention['side_sign'])
                    forward = cam.R[2].to(rays[2].device)
                    entry = rays[1][0, :, :, 0, 0, 0]
                    z = entry * (rays[2][0, :, :, 0] @ forward)
                    z = torch.where((rays[0][0, :, :, 0, 0] != 0) & torch.isfinite(z), z, torch.full_like(z, float('nan')))
                    np.save(out / 'views' / f'{tag}_depth.npy', z.float().cpu().numpy())
                    views.append(dict(pose=pose, height=H, width=W, focal=focal, tag=tag,
                                      pixel_offset=convention['pixel_offset'], side_sign=convention['side_sign'],
                                      rgb=f'views/{tag}.png', depth=f'views/{tag}_depth.npy',
                                      depth_kind='teacher first-voxel entry of extruded map volume (proxy)'))
    (out / 'views.json').write_text(json.dumps({'views': views}, indent=1))
    (out / 'scene.json').write_text(json.dumps({
        'name': a.name, 'split': a.split, 'units_per_metre': 4,
        'region': {'row': rg[0:2], 'col': rg[2:4], 'z': rg[4:6]},
        'map_bounds': {'row': list(mb[0:2]), 'col': list(mb[2:4])},
        'student_inputs': ['map.png'],
        'teacher': {'model': 'CityDreamer4D CityTopia BG+BLDG, preserved Map10 styles', 'source_hashes': env['hashes'],
                    'map_image_sha256': sha(out / 'map.png'), 'camera_convention': convention,
                    'native_outputs': 'unmodified render_bg + render_bldg composite, road blur kept',
                    'seconds': time.perf_counter() - started}}, indent=2))
    print(json.dumps({'scene': str(out), 'views': len(views), 'convention': convention}), flush=True)


if __name__ == '__main__':
    main()

"""CPU checks for the raw-map branch: no weights, network, CUDA or CityDreamer4D.

Run from this folder:  python -m unittest test_rawmap -v
The synthetic scene checks that the pipeline runs and learns; it is not a result.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import train
from data import Scene, check_split
from evaluate import blank, shuffled
from model import RawMapWorld, TinyBackbone, world_code
from render import Camera, match_native_rays, render_reference, world_to_pose

HERE = Path(__file__).resolve().parent


def gt_world():
    """Grey ground plane with a red box building at rows/cols 20..44, height 24."""
    pts, cols = [], []
    for r in np.arange(2, 64, 4.):
        for c in np.arange(2, 64, 4.):
            pts.append([r, c, 0.]); cols.append([.5, .5, .5])
    for z in np.arange(2, 24, 4.):
        for t in np.arange(20, 45, 4.):
            for r, c in [(20, t), (44, t), (t, 20), (t, 44)]:
                pts.append([r, c, z]); cols.append([.8, .15, .1])
    for r in np.arange(22, 44, 4.):
        for c in np.arange(22, 44, 4.):
            pts.append([r, c, 24.]); cols.append([.8, .15, .1])
    n = len(pts)
    return {'means': torch.tensor(pts, dtype=torch.float32), 'quats': torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
            'scales': torch.full((n, 3), 2.2), 'opacities': torch.full((n,), .99), 'colors': torch.tensor(cols, dtype=torch.float32)}


def write_scene(root, name, split, views=6):
    folder = Path(root) / name
    (folder / 'views').mkdir(parents=True)
    m = np.full((64, 64, 3), 128, np.uint8)
    m[20:44, 20:44] = [200, 40, 25]
    Image.fromarray(m).save(folder / 'map.png')
    world, rows = gt_world(), []
    for k in range(views):
        a = 2 * np.pi * k / views
        origin = [32 + 40 * np.cos(a), 32 + 40 * np.sin(a), 14.]
        view = dict(pose=world_to_pose(origin, [32., 32., 8.]), height=24, width=32, focal=28., pixel_offset=0., side_sign=1.,
                    rgb=f'views/{k}.png', depth=f'views/{k}_depth.npy', tag=f'v{k}')
        with torch.no_grad():
            rgb, alpha, depth = render_reference(world, Camera.from_view(view))
            rgb = rgb + (1 - alpha)[..., None] * torch.tensor([.55, .7, .9])
        Image.fromarray((rgb.clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(folder / view['rgb'])
        np.save(folder / view['depth'], torch.where(alpha > .5, depth, torch.full_like(depth, float('nan'))).numpy())
        rows.append(view)
    (folder / 'views.json').write_text(json.dumps({'views': rows}))
    (folder / 'scene.json').write_text(json.dumps({'name': name, 'split': split, 'units_per_metre': 4,
        'region': {'row': [0, 64], 'col': [0, 64], 'z': [-4, 28]}, 'map_bounds': {'row': [0, 64], 'col': [0, 64]},
        'student_inputs': ['map.png'], 'teacher': {'synthetic': True}}))
    return folder


class CameraTest(unittest.TestCase):
    def test_pixel_ray_projects_to_its_pixel_centre(self):
        for offset, sign in [(0., 1.), (.5, -1.)]:
            cam = Camera(torch.tensor([10., 20., 5.]), torch.tensor([60., 35., 2.]), 30, 40, 33., offset, sign)
            dirs = cam.ray_directions()
            for i, j in [(0, 0), (7, 31), (29, 39)]:
                p = cam.origin + 50 * dirs[i, j]
                q = cam.K @ (cam.R @ p + cam.t)
                self.assertTrue(torch.allclose(q[:2] / q[2], torch.tensor([j + .5, i + .5]), atol=1e-3))

    def test_convention_is_recovered_from_native_rays(self):
        o, t = torch.tensor([1., 2., 3.]), torch.tensor([20., -5., 1.])
        native = Camera(o, t, 18, 24, 20., .5, -1.).ray_directions()
        offset, sign, cos = match_native_rays(native, o, t, 18, 24, 20.)
        self.assertEqual((offset, sign), (.5, -1.))
        self.assertGreater(cos, .99999)

    def test_reference_renderer_places_gaussian_and_depth(self):
        cam = Camera(torch.tensor([0., 0., 0.]), torch.tensor([10., 0., 0.]), 20, 20, 20.)
        g = {'means': torch.tensor([[10., 0., 0.]], requires_grad=True), 'quats': torch.tensor([[1., 0, 0, 0]]),
             'scales': torch.full((1, 3), .5), 'opacities': torch.tensor([.9]), 'colors': torch.tensor([[1., 0., 0.]])}
        rgb, alpha, depth = render_reference(g, cam)
        self.assertGreater(alpha[10, 10], .85)
        self.assertLess(alpha[0, 0], .01)
        self.assertAlmostEqual(float(depth[10, 10]), 10., places=3)
        rgb.sum().backward()
        self.assertTrue(torch.isfinite(g['means'].grad).all())


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = RawMapWorld(TinyBackbone(), dim=32, style_dim=8, cell=8., z_tokens=4, gaussians_per_token=2)
        self.map = torch.rand(1, 3, 64, 64)
        self.region = {'row': [0, 64], 'col': [0, 64], 'z': [0, 32]}
        self.bounds = {'row': [0, 64], 'col': [0, 64]}

    def test_world_is_generated_once_and_reused_across_cameras(self):
        style = world_code(3, 8)
        with torch.no_grad():
            world = self.model(self.map, self.region, self.bounds, style)
            again = self.model(self.map, self.region, self.bounds, style)
            before = {k: v.clone() for k, v in world.items()}
            for o in [[-20., 32., 10.], [32., -20., 10.]]:
                render_reference(world, Camera(torch.tensor(o), torch.tensor([32., 32., 8.]), 12, 16, 14.))
        self.assertEqual(world['means'].shape[0], 8 * 8 * 4 * 2)
        for k in world:
            self.assertTrue(torch.equal(world[k], again[k]), k)
            self.assertTrue(torch.equal(world[k], before[k]), k)
        r, c = world['means'][:, 0], world['means'][:, 1]
        self.assertTrue(((r > -8) & (r < 72) & (c > -8) & (c < 72)).all())

    def test_generation_reads_the_map(self):
        style = world_code(-1, 8)
        with torch.no_grad():
            a = self.model(self.map, self.region, self.bounds, style)
            b = self.model(blank(self.map), self.region, self.bounds, style)
        self.assertFalse(torch.allclose(a['colors'], b['colors']))

    def test_gradients_reach_every_trainable_part(self):
        world = self.model(self.map, self.region, self.bounds, world_code(1, 8))
        rgb, alpha, _ = render_reference(world, Camera(torch.tensor([-20., 32., 14.]), torch.tensor([32., 32., 8.]), 12, 16, 14.))
        (rgb.mean() + alpha.mean() + self.model.sky(torch.tensor([[0., 0., 1.]]), world_code(1, 8)).mean()).backward()
        for name in ['memory.adapt.weight', 'memory.stem4.0.weight', 'memory.relation.layers.0.linear1.weight',
                     'decoder.read.0.offsets.weight', 'decoder.column.layers.0.linear1.weight', 'decoder.head.3.weight']:
            grad = dict(self.model.named_parameters())[name].grad
            self.assertTrue(grad is not None and grad.abs().sum() > 0, name)


class DataTest(unittest.TestCase):
    def test_student_inputs_and_scene_split_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_scene(tmp, 'a', 'train', views=1)
            meta = json.loads((folder / 'scene.json').read_text())
            meta['student_inputs'] = ['map.png', 'semantic.png']
            (folder / 'scene.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                Scene(folder)
        class S:
            def __init__(self, name, split):
                self.name, self.split = name, split
        with self.assertRaises(ValueError):
            check_split([S('x', 'train'), S('x', 'test')])

    def test_controls_keep_size_and_content_budget(self):
        m = torch.rand(1, 3, 64, 64)
        s = shuffled(m)
        self.assertEqual(s.shape, m.shape)
        self.assertTrue(torch.allclose(s.sort(-1).values.sum(), m.sort(-1).values.sum()))
        self.assertFalse(torch.equal(s, m))
        self.assertTrue(torch.allclose(blank(m)[0, :, 5, 7], m.mean((2, 3))[0]))


class PipelineTest(unittest.TestCase):
    def test_train_then_evaluate_on_synthetic_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenes = Path(tmp) / 'scenes'
            write_scene(scenes, 'box_train', 'train')
            write_scene(scenes, 'box_test', 'test', views=3)
            run = Path(tmp) / 'run'
            history = train.main(['--scenes', str(scenes), '--out', str(run), '--cpu', '--steps', '150', '--scale', '1',
                                  '--map-size', '64', '--dim', '32', '--style-dim', '8', '--cell', '8', '--z-tokens', '4',
                                  '--k', '2', '--lr', '3e-3', '--views-per-step', '2', '--eval-every', '1000'])
            first, last = np.mean(history[:10]), np.mean(history[-10:])
            self.assertLess(last, .7 * first, (first, last))
            out = subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(run / 'checkpoint.pt'),
                                  '--scenes', str(scenes), '--out', str(Path(tmp) / 'eval'), '--cpu', '--views', '3'],
                                 cwd=HERE, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            report = json.loads((Path(tmp) / 'eval' / 'REPORT.json').read_text())
            result = report['scenes']['box_test']
            for k in ['real', 'blank', 'shuffled', 'restyled']:
                self.assertIn('psnr', result[k])
            self.assertEqual(len(list((Path(tmp) / 'eval' / 'box_test').glob('*.png'))), 3)


if __name__ == '__main__':
    unittest.main()

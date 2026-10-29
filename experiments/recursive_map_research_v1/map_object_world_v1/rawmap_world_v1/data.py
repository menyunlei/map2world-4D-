"""Scene pairs: one RGB map image -> calibrated views of the corresponding world.

<scene>/scene.json
    name, split ('train' | 'val' | 'test'), units_per_metre,
    region      {'row': [r0, r1], 'col': [c0, c1], 'z': [z0, z1]}   world units
    map_bounds  {'row': [r0, r1], 'col': [c0, c1]}                   extent of map.png
    student_inputs ['map.png']      -- enforced: nothing else may be read by the student
    teacher     free-form provenance (checkpoints, source hashes); never a model input
<scene>/map.png                     raw RGB map image
<scene>/views.json   {'views': [{pose, height, width, focal, pixel_offset, side_sign,
                                 rgb, depth (optional), depth_kind, tag}]}
    depth: .npy float32 camera-z in world units, NaN where no surface (sky).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

STUDENT_INPUTS = ['map.png']


def load_map(path, size=512):
    image = Image.open(path).convert('RGB').resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()[None] / 255


class Scene:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'scene.json').read_text())
        if self.meta.get('student_inputs') != STUDENT_INPUTS:
            raise ValueError(f'{folder}: student_inputs must be exactly {STUDENT_INPUTS}')
        self.views = json.loads((self.folder / 'views.json').read_text())['views']
        self.name, self.split = self.meta['name'], self.meta['split']
        self.region, self.map_bounds = self.meta['region'], self.meta['map_bounds']

    def map_image(self, size=512, path=None):
        return load_map(path or self.folder / 'map.png', size)

    def target(self, index, scale=1.):
        """(rgb [h, w, 3] in [0, 1], depth [h, w] with NaN for sky, or None)."""
        view = self.views[index]
        image = Image.open(self.folder / view['rgb']).convert('RGB')
        h, w = round(view['height'] * scale), round(view['width'] * scale)
        if (w, h) != image.size:
            image = image.resize((w, h), Image.BOX)
        rgb = torch.from_numpy(np.asarray(image).copy()).float() / 255
        depth = None
        if view.get('depth'):
            d = torch.from_numpy(np.load(self.folder / view['depth']).astype(np.float32))
            if d.shape != (h, w):
                # Nearest keeps sky/surface boundaries; never average a NaN into a depth.
                d = F.interpolate(d[None, None], size=(h, w), mode='nearest')[0, 0]
            depth = d
        return rgb, depth


def load_scenes(root, split=None):
    scenes = [Scene(p.parent) for p in sorted(Path(root).glob('*/scene.json'))]
    if split is not None:
        scenes = [s for s in scenes if s.split == split]
    return scenes


def check_split(scenes):
    """Scene-level split: a scene name must not appear in two splits."""
    seen = {}
    for s in scenes:
        if seen.setdefault(s.name, s.split) != s.split:
            raise ValueError(f'scene {s.name} appears in {seen[s.name]} and {s.split}')
    return {k: sorted(n for n, v in seen.items() if v == k) for k in set(seen.values())}

"""Reuse the frozen G2-B parent and legitimate narrow TRAIN schedule."""
from paths import *
import runtime as rt
import torch
from subcell_model_v1 import attach

NEW = JOB / 'subcell_structure_v1'
OLD = JOB / 'narrow_structure_v1'


class Data(rt.TrainData):
    def __init__(self, stage):
        super().__init__(stage)
        for r in read(OLD / 'TARGETS_READY.json')['narrow_records']:
            assert r['split'] == 'SUPPLEMENTARY_TRAIN'
            original = self.lookup[r['family_id'] + '_open4m']
            self.lookup[r['scene_id']] = {**original, 'scene_id': r['scene_id'],
                'map_path': r['map_path'], 'structure': str(Path(r['target_dir']) / 'structure_native.pt')}
        for path, digest in read(OLD / 'TRAIN_FREEZE.json')['files'].items():
            assert sha(path) == digest


def initialize(arm, seed, mode=None):
    model, branch, parent, _ = rt.initialize('g2_b', seed)
    meta = read(JOB / 'runs/g2_b' / f'seed{seed}/CHECKPOINT_000300.json')
    assert sha(meta['path']) == meta['sha256']
    saved = torch.load(meta['path'], map_location='cpu', weights_only=False)
    assert set(branch) == set(saved['branch_state'])
    with torch.no_grad():
        for name, parameter in branch.items():
            parameter.copy_(saved['branch_state'][name])
            assert torch.equal(parameter.cpu(), saved['branch_state'][name])
    attach(model, mode or ('control' if arm == 'baseline' else 'correct'))
    branch = {name: p for name, p in model.named_parameters() if p.requires_grad}
    return model, branch, parent, saved['rng']

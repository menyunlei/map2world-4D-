"""Original G2-B starts and unchanged legitimate TRAIN data; no failed arm reuse."""
from paths import *
from subcell_runtime_v1 import Data, rt
from query_boundary_model_v1 import attach
import torch

NEW = JOB / 'query_boundary_structure_v1'


def initialize(arm, seed, mode=None):
    assert arm in ('baseline', 'local') and seed in (1701, 1702)
    model, branch, parent, _ = rt.initialize('g2_b', seed)
    meta = read(JOB / 'runs/g2_b' / f'seed{seed}/CHECKPOINT_000300.json')
    assert sha(meta['path']) == meta['sha256']
    saved = torch.load(meta['path'], map_location='cpu', weights_only=False)
    assert set(branch) == set(saved['branch_state'])
    with torch.no_grad():
        for name, parameter in branch.items():
            parameter.copy_(saved['branch_state'][name])
            assert torch.equal(parameter.cpu(), saved['branch_state'][name])
    attach(model, mode or ('global' if arm == 'baseline' else 'local'))
    return model, {n: p for n, p in model.named_parameters() if p.requires_grad}, parent, saved['rng']


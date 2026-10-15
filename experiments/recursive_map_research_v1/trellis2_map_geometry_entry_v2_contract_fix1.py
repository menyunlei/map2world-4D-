"""Execute actual selected native sampler classes on CPU, without mesh imports."""
import ast
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
from typing import Any, Optional
from paths import JOB, read, save, sha
from trellis2_prepare_v1 import REPO
from trellis2_map_geometry_entry_v2_prepare import OUT


def load_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Reader:
    def __init__(self):
        self.calls = []
    def __call__(self, x, t, features, mapbatch):
        assert torch.count_nonzero(features) == 0
        self.calls.append(mapbatch is not None)
        return torch.ones_like(x) * (2 if mapbatch is not None else 0)


def main():
    paths = [JOB / 'trellis2_map_geometry_entry_v2_generate.py',
             REPO / 'trellis2/pipelines/samplers/classifier_free_guidance_mixin.py',
             REPO / 'trellis2/pipelines/samplers/flow_euler.py']
    namespace = dict(torch=torch, np=np, tqdm=tqdm, edict=edict,
                     Any=Any, Optional=Optional, Sampler=object)
    flow_class = load_class(paths[0], 'GeometryFlow', namespace)
    load_class(paths[1], 'ClassifierFreeGuidanceSamplerMixin', namespace)
    load_class(paths[2], 'FlowEulerSampler', namespace)
    sampler_class = load_class(paths[2], 'FlowEulerCfgSampler', namespace)
    flow = flow_class.__new__(flow_class)
    torch.nn.Module.__init__(flow)
    flow.reader = Reader()
    flow.mapbatch = {'input_only_map': 'sentinel'}
    flow.call_audit = []
    positive = {'features': torch.zeros(1, 2, 1024), 'use_map': True}
    negative = {'features': torch.zeros(1, 2, 1024), 'use_map': False}
    sampler = sampler_class(sigma_min=1e-5)
    result = sampler.sample(flow, torch.zeros(1, 1), positive, negative,
                            steps=3, guidance_strength=3, verbose=False).samples
    assert flow.reader.calls == [True, False] * 3
    assert torch.allclose(result, torch.full_like(result, -6))
    save(OUT / 'CONDITION_CONTRACT_CPU_FIX1.json', {
         'status': 'PASS', 'GPU_seconds': 0, 'native_source_hashes': {str(p): sha(p) for p in paths},
         'selected_unmodified_native_class_AST_executed': True,
         'actual_native_Euler_CFG_logic_executed': True,
         'mesh_and_Triton_imports_avoided_not_model_repaired': True,
         'zero_image_keeps_positive_map': True, 'negative_map_disabled': True,
         'sampler_calls': flow.call_audit, 'not_a_quality_result': True})
    print('ACTUAL_NATIVE_EULER_CFG_CPU_PASS', flush=True)


if __name__ == '__main__':
    main()

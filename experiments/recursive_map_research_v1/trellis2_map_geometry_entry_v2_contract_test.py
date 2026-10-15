"""CPU check of the ACTUAL native CFG sampler with explicit zero-image map tags."""
import torch
from paths import save
from trellis2_map_geometry_entry_v2_prepare import OUT
from trellis2_map_geometry_entry_v2_generate import GeometryFlow
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler


class RecordingReader:
    def __init__(self):
        self.calls = []

    def __call__(self, x, t, features, mapbatch):
        assert torch.count_nonzero(features) == 0
        self.calls.append(mapbatch is not None)
        return torch.ones_like(x) * (2 if mapbatch is not None else 0)


def main():
    flow = GeometryFlow.__new__(GeometryFlow)
    torch.nn.Module.__init__(flow)
    flow.reader = RecordingReader()
    flow.mapbatch = {'input_map': 'sentinel, not a target'}
    flow.call_audit = []
    features = torch.zeros(1, 2, 1024)
    cond = {'features': features, 'use_map': True}
    negative = {'features': features.clone(), 'use_map': False}
    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    x = torch.zeros(1, 1)
    result = sampler.sample(flow, x, cond, negative, steps=3,
                            guidance_strength=3, verbose=False).samples
    assert flow.reader.calls == [True, False] * 3, flow.reader.calls
    assert torch.allclose(result, torch.full_like(x, -6))
    assert all(c['image_feature_nonzero'] == 0 for c in flow.call_audit)
    save(OUT / 'CONDITION_CONTRACT_CPU.json', {'status': 'PASS',
         'actual_native_CFG_sampler': True, 'GPU_seconds': 0,
         'zero_geometry_features_do_not_disable_positive_map': True,
         'unconditional_negative_has_no_map': True,
         'sampler_calls': flow.call_audit, 'geometry_image_feature_values': 'all zero',
         'not_a_quality_or_learning_result': True})
    print('ACTUAL_NATIVE_CFG_ZERO_IMAGE_MAP_CONTRACT_PASS', flush=True)


if __name__ == '__main__':
    main()

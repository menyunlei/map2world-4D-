"""Same stages, plus matched spatially addressed sparse initial noise for edits."""
import argparse
import sys
import torch
import trellis2_map_geometry_entry_v2_generate as core


class WorldAddressNoiseSampler:
    def __init__(self, sampler, seed, texture=False):
        self.sampler = sampler
        self.seed = seed
        self.texture = texture
        self.calls = 0
        self.grids = {}

    def sample(self, model, noise, *args, **kwargs):
        # These are the native 512/1024 SLat coordinate lattices, not model
        # output bounds. No support cells are created, removed or repositioned.
        coords = noise.coords.detach().cpu().long()
        assert coords.shape[1] == 4 and bool((coords[:, 0] == 0).all())
        assert bool(((coords[:, 1:] >= 0) & (coords[:, 1:] < 64)).all())
        stage = 2 if self.texture else self.calls % 2
        self.calls += 1
        channels = noise.feats.shape[-1]
        key = (stage, channels)
        if key not in self.grids:
            generator = torch.Generator(device='cpu').manual_seed(self.seed + 200000 + stage * 1000)
            self.grids[key] = torch.randn((64, 64, 64, channels), generator=generator)
        features = self.grids[key][coords[:, 1], coords[:, 2], coords[:, 3]]
        addressed = noise.replace(features.to(device=noise.device, dtype=noise.feats.dtype))
        return self.sampler.sample(model, addressed, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['original', 'separated'], required=True)
    parser.add_argument('--seed', type=int, choices=[953101, 953102], required=True)
    args = parser.parse_args()
    native_pipeline = core.Trellis2ImageTo3DPipeline
    generation_seed = 953201 + args.seed - 953101

    class FixedNoisePipeline:
        @classmethod
        def from_pretrained(cls, *arguments, **kwargs):
            pipeline = native_pipeline.from_pretrained(*arguments, **kwargs)
            pipeline.shape_slat_sampler = WorldAddressNoiseSampler(pipeline.shape_slat_sampler, generation_seed)
            pipeline.tex_slat_sampler = WorldAddressNoiseSampler(pipeline.tex_slat_sampler, generation_seed, texture=True)
            return pipeline

    core.Trellis2ImageTo3DPipeline = FixedNoisePipeline
    core.main()


if __name__ == '__main__':
    main()

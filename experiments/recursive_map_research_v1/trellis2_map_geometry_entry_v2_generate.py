"""Explicit native geometry/texture stages; no placement, forced support or repair."""
from trellis2_map_learning_generate_v1 import (base, Trellis2ImageTo3DPipeline, rembg,
    Trellis2MapObjectConditioner, export_native)
from trellis2_map_geometry_entry_v2_prepare import OUT
from paths import JOB, read, save, sha
import argparse
import hashlib
import random
import time
import numpy as np
import torch
from PIL import Image


class GeometryFlow(torch.nn.Module):
    def __init__(self, native, branch):
        super().__init__()
        self.native = native
        self.branch = branch
        self.reader = base.BoundMapReader(native, branch)
        self.mapbatch = None
        self.resolution = native.resolution
        self.in_channels = native.in_channels
        self.call_audit = []

    def forward(self, x, t, cond, **kwargs):
        # Explicit CFG payload separates mapped positive from unmapped negative,
        # even though BOTH native image tensors are zero in the separated arm.
        if isinstance(cond, dict):
            features = cond['features']
            mapped = bool(cond['use_map'])
        else:
            features = cond
            mapped = bool(torch.count_nonzero(cond))
        if len(self.call_audit) < 128:
            self.call_audit.append({'mapped': mapped, 'image_feature_nonzero': int(torch.count_nonzero(features))})
        return self.reader(x, t, features, self.mapbatch if mapped else None)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['original', 'separated'], required=True)
    parser.add_argument('--seed', type=int, choices=[953101, 953102], required=True)
    args = parser.parse_args()
    directory = OUT / args.arm
    plan = read(directory / 'PLAN.json')
    checkpoint_record = read(directory / f'seed{args.seed}/CHECKPOINT_000300.json')
    assert sha(checkpoint_record['path']) == checkpoint_record['sha256']
    checkpoint = torch.load(checkpoint_record['path'], map_location='cpu', weights_only=False)
    assert checkpoint['updates'] == 300 and checkpoint['map10_TRAIN'] is False
    assert sha(plan['appearance_condition']) == plan['appearance_sha256']
    rembg.UnusedBackgroundRemoval = lambda: None
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(JOB / 'map_object_world_v1/trellis2_resume_v3/runtime'))
    pipeline.cuda()
    branch = Trellis2MapObjectConditioner()
    branch.load_state_dict(checkpoint['branch_state'], strict=True)
    branch.eval().requires_grad_(False)
    flow = GeometryFlow(pipeline.models['sparse_structure_flow_model'], branch)
    pipeline.models['sparse_structure_flow_model'] = flow
    image = Image.open(plan['appearance_condition']).convert('RGB')
    # Image features are evaluated deterministically and used only for texture in
    # separated mode. Fixed zero tensors discard ALL image feature values.
    appearance512 = pipeline.get_cond([image], 512)
    appearance1024 = pipeline.get_cond([image], 1024)
    if args.arm == 'original':
        geom512, geom1024 = appearance512, appearance1024
        ss_condition = appearance512
    else:
        geom512 = {key: torch.zeros_like(value) for key, value in appearance512.items()}
        geom1024 = {key: torch.zeros_like(value) for key, value in appearance1024.items()}
        ss_condition = {'cond': {'features': geom512['cond'], 'use_map': True},
                        'neg_cond': {'features': geom512['neg_cond'], 'use_map': False}}
        assert all(torch.count_nonzero(value) == 0 for value in [*geom512.values(), *geom1024.values()])
    modes = [('correct', 'map.npz'), ('edit', 'removed_last_building_map.npz')]
    if args.arm == 'separated':
        modes.append(('shuffled', 'shuffled_map.npz'))
    generation_seed = 953201 + args.seed - 953101
    source_weights = {key: sorted({str(p.dtype) for p in model.parameters()}) for key, model in pipeline.models.items()}
    for mode, filename in modes:
        folder = directory / f'seed{args.seed}/assets/{mode}'
        folder.mkdir(parents=True, exist_ok=False)
        with np.load(OUT / 'real_input' / filename) as archive:
            flow.mapbatch = {key: torch.from_numpy(archive[key].copy()).unsqueeze(0).cuda() for key in archive.files}
        flow.call_audit = []
        random.seed(generation_seed)
        np.random.seed(generation_seed)
        torch.manual_seed(generation_seed)
        start = time.time()
        trace = {}
        def record_decode(_module, arguments, output):
            trace['latent'] = arguments[0].detach().float().cpu().numpy()
            trace['native_support64'] = (output > 0).detach().cpu().numpy()
        handle = pipeline.models['sparse_structure_decoder'].register_forward_hook(record_decode)
        with torch.no_grad():
            coords = pipeline.sample_sparse_structure(ss_condition, 32)
            handle.remove()
            assert len(coords), 'EMPTY_MODEL_GENERATED_SUPPORT: do not manufacture geometry'
            with (folder / 'SS_GENERATED.npz').open('xb') as output:
                np.savez_compressed(output, **trace, native_coords32=coords.cpu().numpy())
            shape, resolution = pipeline.sample_shape_slat_cascade(geom512, geom1024,
                pipeline.models['shape_slat_flow_model_512'], pipeline.models['shape_slat_flow_model_1024'],
                512, 1024, coords, max_num_tokens=49152)
            geometry_digest = {'shape_feats_sha256': digest_tensor(shape.feats.float()),
                               'shape_coords_sha256': digest_tensor(shape.coords),
                               'coarse_support_sha256': digest_tensor(coords)}
            texture = pipeline.sample_tex_slat(appearance1024,
                pipeline.models['tex_slat_flow_model_1024'], shape)
            meshes = pipeline.decode_latent(shape, texture, resolution)
            export_native(meshes[0], (shape, texture, resolution), folder)
        if args.arm == 'separated':
            assert flow.call_audit and all(row['image_feature_nonzero'] == 0 for row in flow.call_audit)
            assert any(row['mapped'] for row in flow.call_audit), 'ZERO_IMAGE_MUST_NOT_DISABLE_MAP'
        save(folder / 'GENERATION.json', {'arm': args.arm, 'mode': mode, 'training_seed': args.seed,
             'additional_updates': 300, 'starting_parent_updates': 100,
             'generation_seed': generation_seed, 'seconds': time.time()-start,
             'condition_entry': 'image geometry + map' if args.arm == 'original' else 'map SS; zero-image native fine ShapeFlow; reference image texture only',
             'geometry_digest': geometry_digest, 'SS_CFG_actual_calls': flow.call_audit,
             'samplers': {'SS': pipeline.sparse_structure_sampler_params,
                          'shape': pipeline.shape_slat_sampler_params, 'texture': pipeline.tex_slat_sampler_params},
             'precision': source_weights, 'native_resolution': resolution,
             'actual_model_support_cells': len(coords), 'appearance_reference': 'same fixed stylized building; not recovered real facades',
             'map10_not_training': True, 'no_postplacement_or_output_pruning': True,
             'manual_roof_proxy_input_not_surveyed_footprints': True})
        print('NATIVE_ENTRY_ASSET_SAVED', args.arm, args.seed, mode, len(coords), flush=True)
        del meshes, shape, texture, coords
        torch.cuda.empty_cache()
    save(directory / f'seed{args.seed}/GENERATION_COMPLETE.json', {'assets': len(modes),
         'additional_training_updates': 300, 'asset_acceptance_pending': True})


if __name__ == '__main__':
    main()

"""One legal TRAIN mesh -> actual frozen TRELLIS.2 shape codec, no model training."""
from paths import JOB, read, save, sha
from trellis2_prepare_v1 import DEST as INSTALL, REPO
import os
import sys
import time
os.environ['CUDA_HOME'] = str(INSTALL / 'environment')
os.environ['PATH'] = str(INSTALL / 'environment/bin') + ':' + os.environ['PATH']
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import o_voxel
import trimesh
from PIL import Image
from trellis2 import models
from trellis2.modules.sparse import SparseTensor

OUT = JOB / 'map_object_world_v1/trellis2_map_geometry_tokens_v3'
OLD = JOB / 'map_object_world_v1/trellis2_map_learning_v1/trial_fix1'
WEIGHTS = INSTALL / 'weights/TRELLIS.2-4B/ckpts'


def encode_record(record, encoder):
    assert record['split'] in ('adapter_TRAIN', 'adapter_HELDOUT')
    assert sha(record['target_mesh']) == record['mesh_sha256']
    with np.load(record['target_mesh']) as mesh:
        vertices = torch.from_numpy(mesh['vertices_native'].copy()).float()
        faces = torch.from_numpy(mesh['faces'].copy()).int()
    assert torch.isfinite(vertices).all() and bool((vertices.abs() <= .50001).all())
    # Keep the same common world cube. No per-scene/per-object bbox normalization.
    coords, dual, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
        face_weight=1., boundary_weight=.2, regularization_weight=1e-2)
    assert coords.ndim == 2 and coords.shape[1] == 3 and len(coords)
    assert bool(((coords >= 0) & (coords < 512)).all()), 'NATIVE_COORDINATE_OUT_OF_RANGE'
    # Exact official .vxz coordinate/uint8 unpacking convention, geometry only.
    local = ((dual * 512 - coords).clamp(0, 1) * 255).to(torch.uint8).float() / 255
    batch_coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords.int()], dim=1).cuda()
    v = SparseTensor(feats=local.cuda(), coords=batch_coords)
    i = v.replace(intersected.float().cuda())
    with torch.no_grad():
        latent = encoder(v, i, sample_posterior=False)
    assert latent.feats.shape[1] == 32 and torch.isfinite(latent.feats).all()
    assert bool(((latent.coords[:, 1:] >= 0) & (latent.coords[:, 1:] < 32)).all())
    return latent, {'source': record['scene_id'], 'target_mesh_sha256': record['mesh_sha256'],
                    'input_vertices': len(vertices), 'input_faces': len(faces),
                    'voxel_resolution': 512, 'surface_voxels': len(coords),
                    'latent_channels': 32, 'latent_cells': len(latent.coords),
                    'common_world_coordinates_preserved': True,
                    'no_inspected_DEV_or_Map10_training': True}


def main():
    torch.set_num_threads(8)
    start = time.time()
    plan = read(OLD / 'PLAN.json')
    record = next(r for r in plan['records'] if r['split'] == 'adapter_TRAIN')
    OUT.mkdir(exist_ok=True)
    encoder = models.from_pretrained(str(WEIGHTS / 'shape_enc_next_dc_f16c32_fp16')).cuda().eval().requires_grad_(False)
    latent, metadata = encode_record(record, encoder)
    with (OUT / 'PROBE_NATIVE_SHAPE.npz').open('xb') as handle:
        np.savez_compressed(handle, coords=latent.coords.cpu().numpy(), feats=latent.feats.float().cpu().numpy())
    del encoder
    torch.cuda.empty_cache()
    decoder = models.from_pretrained(str(WEIGHTS / 'shape_dec_next_dc_f16c32_fp16')).cuda().eval().requires_grad_(False)
    decoder.set_resolution(512)
    decoder.low_vram = True
    with torch.no_grad():
        meshes = decoder(latent)
    assert len(meshes) == 1 and len(meshes[0].faces) and torch.isfinite(meshes[0].vertices).all()
    mesh = meshes[0]
    with (OUT / 'CODEC_PROBE_GEOMETRY.pt').open('xb') as handle:
        torch.save({'vertices': mesh.vertices.cpu(), 'faces': mesh.faces.cpu(), 'source_TRAIN': record['scene_id']}, handle)
    matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    native = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy() @ matrix.T, faces=mesh.faces.cpu().numpy(), process=False)
    path = OUT / 'CODEC_PROBE_GEOMETRY.glb'
    assert not path.exists()
    trimesh.Scene(native).export(path)
    from trellis2.renderers import MeshRenderer
    import utils3d
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = .001
    renderer.rendering_options.far = 10
    renderer.rendering_options.chunk_size = 8000000
    for name, eye, up in [('oblique', [1., -1., 1.], [0., 0., 1.]),
                          ('top', [0., 0., 1.5], [0., 1., 0.])]:
        extrinsic = utils3d.torch.extrinsics_look_at(torch.tensor(eye, device='cuda'),
            torch.tensor([0., 0., 0.], device='cuda'), torch.tensor(up, device='cuda'))
        fov = torch.tensor(np.deg2rad(65), device='cuda')
        intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        with torch.no_grad():
            rendered = renderer.render(mesh, extrinsic, intrinsic, return_types=['normal', 'mask'])
        shaded = (rendered['normal']*.4+.5)*rendered['mask'][None] + .15*(1-rendered['mask'])[None]
        Image.fromarray((shaded.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(OUT/f'CODEC_PROBE_{name}.png')
    save(OUT / 'NATIVE_SHAPE_PROBE.json', {**metadata, 'status': 'ACTUAL_FROZEN_ENCODER_DECODER_INTERFACE_PASS',
         'model_updates': 0, 'is_reference_codec_reconstruction_not_generation': True,
         'decoded_vertices': len(mesh.vertices), 'decoded_faces': len(mesh.faces),
         'geometry_visual_acceptance_pending': True,
         'geometry_file': str(path), 'geometry_sha256': sha(path),
         'encoder_checkpoint': str(WEIGHTS / 'shape_enc_next_dc_f16c32_fp16'),
         'decoder_checkpoint': str(WEIGHTS / 'shape_dec_next_dc_f16c32_fp16'),
         'seconds': time.time()-start, 'max_allocated_GiB': torch.cuda.max_memory_allocated()/2**30})
    print('LEGITIMATE_TRAIN_NATIVE_SHAPE_CODEC_PASS', metadata, flush=True)


if __name__ == '__main__':
    main()

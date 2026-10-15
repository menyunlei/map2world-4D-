"""Render saved codec output after float64-FOV display error; no re-encoding."""
from trellis2_map_geometry_tokens_v3_probe import OUT, OLD, WEIGHTS
from paths import read, save, sha
import numpy as np
import torch
from PIL import Image
from trellis2.representations import Mesh
from trellis2.renderers import MeshRenderer
import utils3d

if __name__ == '__main__':
    record = next(r for r in read(OLD / 'PLAN.json')['records'] if r['split'] == 'adapter_TRAIN')
    data = torch.load(OUT / 'CODEC_PROBE_GEOMETRY.pt', map_location='cpu', weights_only=False)
    assert data['source_TRAIN'] == record['scene_id']
    with np.load(OUT / 'PROBE_NATIVE_SHAPE.npz') as encoded:
        assert encoded['feats'].shape[1] == 32 and np.isfinite(encoded['feats']).all()
        latent_cells = len(encoded['coords'])
    mesh = Mesh(data['vertices'].float().cuda(), data['faces'].int().cuda())
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = .001
    renderer.rendering_options.far = 10
    renderer.rendering_options.chunk_size = 8000000
    for name, eye, up in [('oblique', [1., -1., 1.], [0., 0., 1.]),
                          ('top', [0., 0., 1.5], [0., 1., 0.])]:
        extrinsic = utils3d.torch.extrinsics_look_at(torch.tensor(eye, dtype=torch.float32, device='cuda'),
            torch.zeros(3, dtype=torch.float32, device='cuda'), torch.tensor(up, dtype=torch.float32, device='cuda'))
        fov = torch.tensor(np.deg2rad(65), dtype=torch.float32, device='cuda')
        intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        with torch.no_grad():
            rendered = renderer.render(mesh, extrinsic, intrinsic, return_types=['normal', 'mask'])
        shaded = (rendered['normal']*.4+.5)*rendered['mask'][None] + .15*(1-rendered['mask'])[None]
        Image.fromarray((shaded.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(OUT/f'CODEC_PROBE_{name}.png')
    save(OUT / 'NATIVE_SHAPE_PROBE.json', {'status': 'ACTUAL_NATIVE_ENCODER_DECODER_INTERFACE_PASS',
         'source_TRAIN': record['scene_id'], 'source_mesh_sha256': record['mesh_sha256'],
         'native_encoded_latent_sha256': sha(OUT / 'PROBE_NATIVE_SHAPE.npz'),
         'native_latent_channels': 32, 'native_latent_cells': latent_cells,
         'decoded_vertices': len(data['vertices']), 'decoded_faces': len(data['faces']),
         'common_coordinate_frame_preserved': True, 'native_geometry_visual_acceptance_pending': True,
         'original_probe_failure': 'Render-only FOV tensor float64 versus extrinsic float32',
         'reused_same_saved_latent_and_decoded_geometry': True,
         'encoder_decoder_rerun': False, 'model_updates': 0,
         'codec_reconstruction_not_generated_world': True,
         'geometry_file': str(OUT / 'CODEC_PROBE_GEOMETRY.glb'),
         'geometry_sha256': sha(OUT / 'CODEC_PROBE_GEOMETRY.glb'),
         'encoder_checkpoint': str(WEIGHTS / 'shape_enc_next_dc_f16c32_fp16'),
         'decoder_checkpoint': str(WEIGHTS / 'shape_dec_next_dc_f16c32_fp16')})
    print('SAVED_NATIVE_SHAPE_CODEC_RELOADED_RENDERED', latent_cells, flush=True)

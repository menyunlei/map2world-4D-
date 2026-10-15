"""Encode unchanged legal paired meshes into actual native shape targets."""
from trellis2_map_geometry_tokens_v3_probe import WEIGHTS
from trellis2_map_geometry_tokens_v3_prepare import OUT, RESOLUTION
from paths import read, save, sha
import argparse
import time
import numpy as np
import torch
import o_voxel
from trellis2 import models
from trellis2.modules.sparse import SparseTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = read(OUT / 'PLAN.json')
    encoder = models.from_pretrained(str(WEIGHTS / 'shape_enc_next_dc_f16c32_fp16')).cuda().eval().requires_grad_(False)
    records = []
    for index, record in enumerate(plan['records']):
        if index % 4 != args.worker:
            continue
        start = time.time()
        assert record['original_split'] == 'train'
        assert sha(record['target_mesh']) == record['mesh_sha256']
        with np.load(record['target_mesh']) as mesh:
            vertices = torch.from_numpy(mesh['vertices_native'].copy()).float()
            faces = torch.from_numpy(mesh['faces'].copy()).int()
        assert torch.isfinite(vertices).all() and bool((vertices.abs() <= .50001).all())
        coords, dual, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices, faces, grid_size=RESOLUTION, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
            face_weight=1., boundary_weight=.2, regularization_weight=1e-2)
        assert len(coords) and bool(((coords >= 0) & (coords < RESOLUTION)).all())
        local = ((dual*RESOLUTION-coords).clamp(0,1)*255).to(torch.uint8).float()/255
        batch_coords = torch.cat([torch.zeros(len(coords),1,dtype=torch.int32),coords.int()],1).cuda()
        sparse = SparseTensor(feats=local.cuda(), coords=batch_coords)
        with torch.no_grad():
            latent = encoder(sparse, sparse.replace(intersected.float().cuda()), sample_posterior=False)
        assert latent.feats.shape[1] == 32 and torch.isfinite(latent.feats).all()
        assert bool(((latent.coords[:,1:] >= 0)&(latent.coords[:,1:] < 64)).all())
        path = OUT / 'targets' / (record['scene_id']+'.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle, coords=latent.coords.cpu().numpy(), feats=latent.feats.float().cpu().numpy())
        encoded = {**record, 'native_shape_target': str(path), 'native_shape_sha256': sha(path),
             'native_shape_channels':32, 'native_shape_cells':len(latent.coords),
             'surface_voxels':len(coords), 'voxel_resolution':RESOLUTION,
             'encoding_seconds':time.time()-start, 'frozen_encoder':True,
             'common_native_frame_unchanged':True, 'not_old_8D_or_20D_world_latent':True}
        save(OUT / 'target_records' / (record['scene_id']+'.json'), encoded)
        records.append(record['scene_id'])
        print('ENCODED_LEGITIMATE_PAIR', record['scene_id'], len(latent.coords), flush=True)
        del sparse, latent, coords, local, dual, intersected
    save(OUT / f'ENCODE_COMPLETE_WORKER{args.worker}.json', {'scenes':records,
         'count':len(records), 'frozen_native_encoder':True, 'model_training_updates':0})

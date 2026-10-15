"""Independent RAW_NATIVE reload/inspection; never imports training or generation.

Remote invocation: --seed 953101 --arm original --mode correct --out <experiment root>.
Canonical input: <out>/<arm>/seed<seed>/assets/<mode>/RAW_NATIVE.pt.
Outputs: the selected asset's own views/ directory; existing views are preserved.
Only the caller launches this renderer; importing it and --help are CPU-only.
"""
import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from paths import JOB, read, save, sha


EXPERIMENT = JOB / 'map_object_world_v1/trellis2_map_geometry_entry_v2'
FROZEN = JOB / 'map_object_world_v1/trellis2_map_learning_v1/trial_fix1'
RESOLUTION = 640
NEAR, FAR = .001, 10.
CUBE = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                 for z in (-.5, .5)], dtype=np.float64)
PROXY_PLANE_Z = 0.  # Display convention only: input has NO measured Z.
COLORS = ((49, 190, 255), (235, 99, 245), (89, 225, 151), (255, 179, 63))
MODE_MAPS = {'correct': 'map.npz', 'edit': 'removed_last_building_map.npz',
             'shuffled': 'shuffled_map.npz'}


def cameras(protocol):
    """Keep every frozen pose, including blank street/lateral negatives."""
    poses = protocol['cameras']
    expected = ['top', 'oblique', 'front', 'back', 'left', 'right',
                'street_front', 'street_lateral_left', 'street_lateral_right']
    if [p[0] for p in poses] != expected:
        raise ValueError('Unexpected frozen CAMERA_PROTOCOL camera order')
    # These additions are fixed constants, never functions of generated bounds.
    diagonal = math.sqrt(.75)  # norm([d,-d,d]) == 1.5
    return poses + [
        ['close', [0, -.85, .1], [0, 0, -.3], [0, 0, 1], 45],
        ['whole_unit_cube', [diagonal, -diagonal, diagonal], [0, 0, 0], [0, 0, 1], 80],
        ['top_whole_unit_cube', [0, 0, 1.5], [0, 0, 0], [0, 1, 0], 65],
    ]


def camera_matrices(eye, look, up, fov):
    """CPU oracle for native OpenCV (+Z forward, +Y down), normalized K."""
    eye, look, up = (np.asarray(v, dtype=np.float64) for v in (eye, look, up))
    forward = look - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ eye
    focal = .5 / math.tan(math.radians(fov) / 2)
    K = np.array([[focal, 0, .5], [0, focal, .5], [0, 0, 1.]])
    return K, w2c


def project(points, K, w2c):
    points = np.asarray(points, dtype=np.float64)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    if not np.isfinite(camera).all() or (camera[:, 2] <= NEAR).any():
        raise ValueError('Projection contains nonfinite/behind-camera points')
    image = camera @ K.T
    return image[:, :2] / image[:, 2:3], camera[:, 2]


def framing_diagnostic(K, w2c):
    # Street/close cameras can be inside the cube: behind-camera corners are
    # diagnostics, not errors and never reasons to discard a negative view.
    camera = CUBE @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[:, 2]
    uv = []
    for point in camera:
        image = K @ point
        uv.append((image[:2] / image[2]).tolist() if point[2] > NEAR else None)
    in_front = depth > NEAR
    inside = all(p is not None and all(0 < v < 1 for v in p) for p in uv)
    return {'unit_cube_uv_corners': uv,
            'unit_cube_corners_in_front': in_front.tolist(),
            'unit_cube_depth_range': [float(depth.min()), float(depth.max())],
            'entire_unit_cube_inside_frustum': bool(inside and (depth < FAR).all()),
            'asset_bounds_used_to_choose_camera': False}


def raw_path(out, seed, arm, mode='correct'):
    """Resolve only the final exact layout; no cross-mode or legacy fallback."""
    path = out / arm / f'seed{seed}/assets' / mode / 'RAW_NATIVE.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing selected native asset: {path}')
    return path


def overlay_input(out, mode):
    """Bind proxies to the copied map bytes, never reuse correct polygons on controls."""
    path = out / 'real_input' / MODE_MAPS[mode]
    if not path.is_file():
        raise FileNotFoundError(f'Missing selected input map: {path}')
    map_hash = sha(path)
    manifest = out / 'real_input/MAP_INPUT.json'
    manifest_hash = sha(manifest) if manifest.is_file() else None
    status = {'created': False, 'selected_map': str(path), 'selected_map_sha256': map_hash,
              'metric_registration': False, 'geographic_registration': False,
              'assumed_display_plane_z': PROXY_PLANE_Z, 'roof_height_known': False}
    metadata = None
    if mode != 'correct':
        # The copied manifest records raster edits/permutations, but provides
        # no exact variant polygon set. Do not derive contours or pretend the
        # unedited roof polygons represent the modified raster condition.
        status['reason'] = ('Exact polygon representation unavailable for selected '
                            f'{mode} map; original roof polygons not substituted')
    elif manifest_hash is None:
        status['reason'] = 'Copied MAP_INPUT.json unavailable; exact polygons cannot be verified'
    else:
        candidate = read(manifest)
        expected_hash = candidate.get('artifacts', {}).get(path.name, {}).get('sha256')
        if expected_hash != map_hash:
            status['reason'] = 'Selected map bytes do not match the polygon manifest SHA256'
        elif not candidate.get('objects'):
            status['reason'] = 'Exact roof polygon metadata unavailable'
        else:
            with np.load(path, allow_pickle=False) as arrays:
                valid = arrays['object_valid']
                footprints = arrays['footprints']
                matches = (valid.shape == (len(candidate['objects']),) and bool(valid.all())
                           and footprints.shape[0] == len(candidate['objects']))
            if matches:
                metadata = candidate
            else:
                status['reason'] = 'Map object slots/validity differ from the exact polygon metadata'
    return path, manifest, metadata, status, map_hash, manifest_hash


def numpy_value(value):
    return value.detach().float().cpu().numpy()


def write_image(value, path):
    array = numpy_value(value)
    if array.ndim == 2:
        array = array[None]
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    if array.shape != (3, RESOLUTION, RESOLUTION) or not np.isfinite(array).all():
        raise ValueError(f'Invalid image array for {path.name}: {array.shape}')
    Image.fromarray(np.clip(array.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)).save(path)


def map_overlay(meta, K, w2c, folder):
    """Project only frozen roof XY proxies on a declared plane over native RGB.

    Perspective projection needs Z; the chosen plane is a visualization
    assumption, never recovered roof elevation or image-to-world registration.
    """
    transform = np.asarray(meta['coordinate_transform'][
        'source_pixel_to_normalized_xy_homogeneous'], dtype=np.float64)
    homography = K @ np.column_stack((w2c[:3, 0], w2c[:3, 1],
                                     w2c[:3, 3] + PROXY_PLANE_Z * w2c[:3, 2]))
    inverse = np.linalg.inv(homography)
    polygons = []
    for item in meta['objects']:
        xy = np.asarray(item['outer_normalized_xy'], dtype=np.float64)
        source = np.asarray(item['outer_source_px'], dtype=np.float64)
        source_h = np.column_stack((source, np.ones(len(source)))) @ transform.T
        if not np.allclose(xy, source_h[:, :2] / source_h[:, 2:3], atol=1e-10):
            raise ValueError('Roof proxy XY disagrees with frozen screenshot transform')
        xyz = np.column_stack((xy, np.full(len(xy), PROXY_PLANE_Z)))
        uv, _ = project(xyz, K, w2c)
        back = np.column_stack((uv, np.ones(len(uv)))) @ inverse.T
        if not np.allclose(back[:, :2] / back[:, 2:3], xy, atol=1e-9):
            raise ValueError('Roof proxy projection failed plane-homography round trip')
        if (uv < 0).any() or (uv > 1).any():
            raise ValueError('Roof proxy projection extends outside fixed top frame')
        # Native normalized screen coordinates address image edges [0,W].
        polygons.append((item['id'], uv * RESOLUTION))
    with Image.open(folder / 'top_whole_unit_cube.png') as native:
        result = native.convert('RGBA')
    overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, (label, pixels) in enumerate(polygons):
        points = [tuple(p) for p in pixels]
        color = COLORS[index % len(COLORS)]
        draw.polygon(points, fill=(*color, 32))
        draw.line(points + points[:1], fill=(*color, 255), width=2)
        draw.text(tuple(pixels.mean(0)), label, fill=(*color, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0, 255))
    draw.rectangle((0, 0, RESOLUTION, 45), fill=(0, 0, 0, 220))
    draw.text((6, 5), 'ROOF XY PROXIES: assumed display plane Z=0; unknown roof heights', fill='white')
    draw.text((6, 23), 'Screenshot affine XY only; NO metric/geographic registration', fill='white')
    Image.alpha_composite(result, overlay).convert('RGB').save(folder / 'MAP_OVERLAY.png')
    return {'created': True, 'base_native_view': 'top_whole_unit_cube.png',
            'source': 'MAP_INPUT.json objects.outer_normalized_xy only',
            'roof_proxies_not_surveyed_footprints': True,
            'assumed_display_plane_z': PROXY_PLANE_Z, 'roof_height_known': False,
            'metric_registration': False, 'geographic_registration': False,
            'projection': 'uv=K*(w2c*[X,Y,assumed_Z,1]); pixels=uv*640',
            'plane_homography_normalized_uv': homography.tolist(),
            'source_affine_and_projection_round_trip_validated': True,
            'image_reference_pixels_used_as_asset_render': False}


def contact_sheet(folder, records):
    width, height = 320, 350
    sheet = Image.new('RGB', (3 * width, math.ceil(len(records) / 3) * height), '#202020')
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        x, y = (index % 3) * width, (index // 3) * height
        with Image.open(folder / f"{record['name']}.png") as frame:
            frame.thumbnail((width, width))
            sheet.paste(frame, (x, y))
        blank = ' | BLANK (retained)' if record['blank'] else ''
        draw.text((x + 5, y + 322), record['name'] + blank, fill='white')
        draw.text((x + 5, y + 335), f"visible pixels: {record['visible_pixels']}", fill='white')
    sheet.save(folder / 'contact.jpg', quality=92)


def native_runtime():
    # Copy ONLY the install bootstrap and exact native renderer imports from v1.
    # Do not import train_fix1/base: that transitively imports model code.
    from trellis2_prepare_v1 import DEST as INSTALL, REPO
    os.environ['CUDA_HOME'] = str(INSTALL / 'environment')
    os.environ['PATH'] = str(INSTALL / 'environment/bin') + os.pathsep + os.environ['PATH']
    sys.path.insert(0, str(REPO))
    import torch
    import utils3d
    from trellis2.representations import MeshWithVoxel
    from trellis2.renderers import MeshRenderer
    return torch, utils3d, MeshWithVoxel, MeshRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=['original', 'separated'], required=True)
    parser.add_argument('--mode', choices=list(MODE_MAPS), default='correct')
    parser.add_argument('--out', type=Path, required=True, help='trellis2_map_geometry_entry_v2 root')
    args = parser.parse_args()
    out = args.out.resolve()
    if out != EXPERIMENT.resolve():
        raise ValueError(f'--out must point to the new experiment root: {EXPERIMENT}')
    raw = raw_path(out, args.seed, args.arm, args.mode)
    protocol_path = FROZEN / 'CAMERA_PROTOCOL.json'
    input_path, manifest_path, meta, overlay_status, map_hash, manifest_hash = overlay_input(out, args.mode)
    protocol = read(protocol_path)
    poses = cameras(protocol)  # Freeze cameras BEFORE opening the asset.
    hashes = {'raw': sha(raw), 'protocol': sha(protocol_path), 'map_input': map_hash,
              'map_manifest': manifest_hash}
    folder = raw.parent / 'views'
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing inspection: {folder}')
    torch, utils3d, MeshWithVoxel, MeshRenderer = native_runtime()
    data = torch.load(raw, map_location='cpu', weights_only=False)
    vertices, faces = data['vertices'], data['faces']
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError('RAW_NATIVE vertices must be finite Nx3')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('RAW_NATIVE must contain nonempty triangle faces')
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError('RAW_NATIVE face indices outside vertex array')
    spatial = (data['coords'].max(0).values + 1).tolist()
    mesh = MeshWithVoxel(vertices.cuda(), faces.cuda(), [-.5, -.5, -.5], data['voxel_size'],
                         data['coords'].cuda(), data['attrs'].cuda(),
                         torch.Size([1, data['attrs'].shape[1], *spatial]), data['layout'])
    if not torch.equal(mesh.vertices.cpu(), vertices) or not torch.equal(mesh.faces.cpu(), faces):
        raise ValueError('Native reconstruction changed vertices or triangles')
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = RESOLUTION
    renderer.rendering_options.near = NEAR
    renderer.rendering_options.far = FAR
    renderer.rendering_options.ssaa = 1
    renderer.rendering_options.chunk_size = 8000000
    renderer.rendering_options.antialias = False
    folder.mkdir(parents=True, exist_ok=False)
    records, overlay = [], overlay_status.copy()
    for name, eye, look, up, fov in poses:
        native_vectors = [torch.tensor(v, dtype=torch.float32, device='cuda') for v in (eye, look, up)]
        extr = utils3d.torch.extrinsics_look_at(*native_vectors)
        angle = torch.tensor(math.radians(fov), device='cuda')
        intr = utils3d.torch.intrinsics_from_fov_xy(angle, angle)
        K, w2c = numpy_value(intr), numpy_value(extr)
        expected_K, expected_w2c = camera_matrices(eye, look, up, fov)
        if not np.allclose(K, expected_K, atol=2e-6) or not np.allclose(w2c, expected_w2c, atol=2e-6):
            raise ValueError('Native camera convention disagrees with projection oracle')
        framing = framing_diagnostic(K, w2c)
        if name in ('whole_unit_cube', 'top_whole_unit_cube') and not framing['entire_unit_cube_inside_frustum']:
            raise ValueError('Fixed whole-unit-cube camera failed frustum validation')
        with torch.no_grad():
            gb = renderer.render(mesh, extr, intr, return_types=['attr', 'mask', 'normal', 'depth'])
        mask, depth = numpy_value(gb['mask']).squeeze(), numpy_value(gb['depth']).squeeze()
        if mask.shape != (RESOLUTION, RESOLUTION) or depth.shape != mask.shape:
            raise ValueError('Unexpected native depth/mask shape')
        visible = mask > 0
        if not np.isfinite(mask).all() or not np.isfinite(depth[visible]).all():
            raise ValueError('Invalid visible depth or mask')
        write_image(gb['base_color'] + .18 * (1 - gb['mask'])[None], folder / f'{name}.png')
        write_image((gb['normal'] * .6 + .25) * gb['mask'][None] + .18 * (1 - gb['mask'])[None],
                    folder / f'{name}_geometry.png')
        Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8)).save(folder / f'{name}_mask.png')
        # Fixed 0..3 display range; raw depth is always retained, even for blank views.
        depth_display = np.zeros_like(depth)
        depth_display[visible] = np.clip(depth[visible] / 3., 0, 1)
        Image.fromarray((depth_display * 255).astype(np.uint8)).save(folder / f'{name}_depth.png')
        np.savez_compressed(folder / f'{name}_diagnostic.npz', depth=depth, visible_mask=mask,
                            normal=numpy_value(gb['normal']), K=K, w2c=w2c,
                            K_pixels=np.diag([RESOLUTION, RESOLUTION, 1.]) @ K)
        record = {'name': name, 'eye': eye, 'look': look, 'up': up, 'fov_xy_degrees': fov,
                  'visible_pixels': int(visible.sum()), 'blank': not bool(visible.any()),
                  'depth_display_range': [0, 3], **framing}
        records.append(record)
        if name == 'top_whole_unit_cube' and meta is not None:
            try:
                overlay.update(map_overlay(meta, K.astype(np.float64), w2c.astype(np.float64), folder))
            except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                overlay.update({'created': False, 'reason': str(error)})
        print('SAVED_NATIVE_ENTRY_VIEW', args.seed, args.arm, args.mode, name,
              'blank=', record['blank'], flush=True)
    contact_sheet(folder, records)
    if sha(raw) != hashes['raw'] or sha(protocol_path) != hashes['protocol'] or sha(input_path) != hashes['map_input']:
        raise ValueError('Asset or frozen inspection inputs changed during rendering')
    current_manifest_hash = sha(manifest_path) if manifest_path.is_file() else None
    if current_manifest_hash != hashes['map_manifest']:
        raise ValueError('Copied polygon manifest changed during rendering')
    save(folder / 'RELOAD.json', {
        'schema': 'trellis2_map_entry_views_v2', 'seed': args.seed, 'arm': args.arm, 'mode': args.mode,
        'raw_asset': str(raw), 'raw_asset_sha256': hashes['raw'],
        'script_sha256': sha(Path(__file__)), 'fresh_process_raw_reload': True,
        'model_sampling': False, 'optimizer': False, 'viewer': False,
        'geometry_or_material_modified': False, 'all_triangles_retained': True,
        'native_faces': len(faces), 'native_vertices': len(vertices),
        'exact_vertex_face_reload_equality_checked': True,
        'native_bounds_diagnostic_only': [vertices.min(0).values.tolist(), vertices.max(0).values.tolist()],
        'shader': 'unchanged native MeshRenderer; all triangles in 8M chunks; native unlit base_color',
        'PBR_lighting_not_claimed': True, 'opaque_geometry_mask_not_material_alpha': True,
        'resolution': RESOLUTION, 'near': NEAR, 'far': FAR, 'ssaa': 1, 'antialias': False,
        'camera_protocol': str(protocol_path), 'camera_protocol_sha256': hashes['protocol'],
        'map_input': str(input_path), 'map_input_sha256': hashes['map_input'],
        'map_manifest': str(manifest_path), 'map_manifest_sha256': hashes['map_manifest'],
        'fixed_before_asset_inspection': True, 'no_output_based_camera_selection': True,
        'camera_frame': 'native unit cube +Z up; street floor -.4375 synthetic, not measured elevation',
        'K_convention': 'normalized image edges [0,1]; OpenCV +Z forward/+Y down; K_pixels also saved',
        'normal_convention': 'unchanged native renderer output',
        'depth_convention': 'unchanged native renderer output; background governed by visible_mask',
        'negative_blank_views_retained': True, 'views': records, 'map_overlay': overlay,
        'reference_image_used_as_fake_asset_render': False,
        'no_map_control_or_geographic_success_claim_from_inspection': True,
    })


if __name__ == '__main__':
    main()

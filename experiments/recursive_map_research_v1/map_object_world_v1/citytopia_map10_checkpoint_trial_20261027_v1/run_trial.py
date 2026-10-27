"""Bounded static Map10 inference with new CityTopia weights; no training.

Compare current-code building semantic routing with checkpoint-declared routing.
Use the same saved world, poses, styles and common pre-existing inference fixes.
Never modify the original checkout, checkpoints, or source scene.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
JOB = ROOT.parents[1]
PROJECT = JOB.parents[2]
REPO = PROJECT / 'CityDreamer4D'
REPAIR = ROOT.parent / 'citydreamer4d_inference_repair_20261026_v1'
SPATIAL = ROOT.parent / 'citydreamer4d_building_capability_20261026_v1/city_sample_map10'
SOURCE = ROOT.parent / 'citydreamer4d_map10_experiment_b_20261026_v1/recovery_combined_v1'
WEIGHTS = PROJECT / 'DreamX-World-master/inputs/citypth'
UUIDS = {4: 'GPU-a0e3c471-51ce-dd7f-74a6-c0fb291fa88a', 5: 'GPU-ef639cfb-4614-013e-1333-f2d1b43345f1'}
EXPECTED = {'CityDreamer4D-BG.pth': 'b5375b725ba32397f9e8e9106eb5addd4a577cf5d3f022639526aff895b10fe0',
            'CityDreamer4D-BLDG.pth': 'eeff67cbda5d402e1ec46946babc1e2f4acd1c5a2ea47f742dd97335c0c856bb'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def progress(**row):
    print(json.dumps(row), flush=True)


def execute(gpu, out, budget):
    import cv2
    import numpy as np
    import torch

    torch.manual_seed(953101)
    np.random.seed(953101)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    os.chdir(REPO)
    sys.path[:0] = [str(REPO), str(REPAIR)]
    from overlay_loader import load
    native = load(REPO)
    import utils.helpers as helpers
    hashes = {str(p): sha(p) for p in [SOURCE / 'WORLD_LAYOUT.npz', SOURCE / 'WORLD_STYLES.pt', SOURCE / 'CAMERAS.json', SPATIAL / 'MANIFEST.json']}
    for name, expected in EXPECTED.items():
        actual = sha(WEIGHTS / name)
        assert actual == expected, ('checkpoint changed', name)
        hashes[str(WEIGHTS / name)] = actual
    manifest = json.loads((SPATIAL / 'MANIFEST.json').read_text())
    saved = torch.load(SOURCE / 'WORLD_STYLES.pt', map_location='cpu', weights_only=False)
    original_cameras = json.loads((SOURCE / 'CAMERAS.json').read_text())['cameras']
    cameras = json.loads((SPATIAL / 'CAMERAS.json').read_text())['cameras']
    for c, original in zip(cameras, original_cameras):
        assert c['index'] == original['index']
        for kind in ['cam_position', 'cam_look_at']:
            a, b = c['citydreamer_pose'][kind], original['citydreamer_pose'][kind]
            for axis in ['x', 'y']:
                assert abs(a[axis] - ((b[axis] - 768) * 4 + 1536)) < 1e-5
            assert abs(a['z'] - ((b['z'] - 4) * 4 + 14)) < 1e-5
    progress(phase='loading_new_CITY_SAMPLE_models', gpu=gpu)
    bg, bldg, car = native.get_models('CITY_SAMPLE', str(WEIGHTS / 'CityDreamer4D-BG.pth'), str(WEIGHTS / 'CityDreamer4D-BLDG.pth'), None)
    assert car is None
    contracts = {}
    for name, model in [('CityDreamer4D-BG.pth', bg), ('CityDreamer4D-BLDG.pth', bldg)]:
        ck = torch.load(WEIGHTS / name, map_location='cpu', weights_only=False)
        result = model.load_state_dict(ck['gancraft_g'], strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.isfinite(v).all() for v in ck['gancraft_g'].values() if v.is_floating_point())
        model.eval().requires_grad_(False)
        contracts[name] = {'dataset': ck['cfg'].CONST.DATASET, 'network': dict(ck['cfg'].NETWORK.GANCRAFT),
                           'classes': dict(ck['cfg'].DATASETS.CITY_SAMPLE.CLASSES),
                           'delimeter': list(model.module.delimeter), 'center_offset': model.module.center_offset,
                           'n_classes': model.module.n_classes, 'strict_load': True,
                           'semantic_weight_column_norm': model.module.render_net.fc_m_a.weight.detach().norm(dim=0).cpu().tolist()}
        del ck
    assert contracts['CityDreamer4D-BLDG.pth']['classes']['BLDG_FACADE'] == 7
    assert contracts['CityDreamer4D-BLDG.pth']['classes']['BLDG_ROOF'] == 8
    assert bldg.module.n_classes == {'SMT': 9, 'LYT': 9}
    assert bg.module.n_classes == {'SMT': 9, 'LYT': 9}
    projections, footprints = native.get_city_sample_layout(str(SPATIAL), [10, 30000])
    volumes = {k: native.get_cfg_value(v, 'CITY_SAMPLE') for k, v in [('LAYOUT', 'VOL_SIZE/LAYOUT'), ('BLDG', 'VOL_SIZE/BLDG'), ('CAR', 'VOL_SIZE/CAR'), ('EXT', 'VOL_SIZE/EXTEND')]}
    cy, cx = np.array(projections['REST']['TD_HF'].shape) // 2
    common = {'INST_RANGE': [10, 30000], 'MAX_HEIGHT': 2560, 'VOL_SIZE': volumes['EXT'], 'CLASS_MAPPER': native._bldg_class_callback}
    hf_bg, ids = native.get_hf_seg_tensor(projections['REST'], int(cx), int(cy), 9, dict(common, FACADE_CID=8), bg.output_device)
    hf_bldg, ids_b = native.get_hf_seg_tensor(projections['REST'], int(cx), int(cy), 9, dict(common, FACADE_CID=7), bldg.output_device)
    assert np.array_equal(ids, ids_b)
    # Only building-class channels change; height and non-building channels stay exact.
    assert torch.equal(hf_bg[:, :8], hf_bldg[:, :8])
    bmask = torch.from_numpy((projections['REST']['INS_BEV'] >= 100)).to(hf_bg.device)
    assert torch.all(hf_bldg[0, 8][bmask] == 1) and torch.all(hf_bldg[0, 9][bmask] == 0)
    assert torch.all(hf_bg[0, 9][bmask] == 1)
    del bmask
    stats = native.get_part_bldg_stats(ids, footprints, int(cx), int(cy), [10, 30000])
    mapping = {int(k): int(v) for k, v in manifest['old_to_new_instances'].items()}
    styles = {new: saved['building_zs'][old].to(bg.output_device) for old, new in mapping.items()}
    assert all(v.shape[-1] == 256 for v in styles.values())
    assert set(stats).issubset(styles)
    torch.save({'building_zs': {k: v.cpu() for k, v in styles.items()}, 'seed': 953101, 'old_to_new_instances': mapping}, out / 'WORLD_STYLES.pt')
    save(out / 'CAMERAS.json', {'cameras': cameras, 'transform': {'xy': '(old-768)*4+1536', 'z': '(old-4)*4+14', 'units_per_metre': 4}})
    save(out / 'CONTRACT.json', {'models': contracts, 'source_hashes': hashes, 'map10_spatial_manifest': str(SPATIAL / 'MANIFEST.json'),
                               'static': True, 'target_rgb_used': False, 'new_geometry_generated': False,
                               'routing_ablation': {'current_code': {'building_bev': 8, 'facade': 1, 'roof': 2},
                                                    'checkpoint_declared': {'building_bev': 7, 'facade': 7, 'roof': 8}},
                               'background': {'building_bev': 8, 'facade': 8},
                               'limitation': 'Checkpoint class labels are tested as a compatibility hypothesis; stored config does not alone prove historical training transformations.'})
    progress(phase='strict_weights_and_map_contract_passed', stats_count=len(stats), allocated_gib=torch.cuda.memory_allocated()/2**30)
    # Keep native full voxel volume and network normalization unchanged.
    volume = native.get_seg_volume(projections, None, int(cx), int(cy), volumes,
                                  {'ROOF_HEIGHT': 1, 'ROOF_OFFSET': 1, 'INST_RANGE': [10, 30000], 'MAX_HEIGHT': 2560})
    progress(phase='native_volume_ready', shape=list(volume.shape), allocated_gib=torch.cuda.memory_allocated()/2**30)
    cfg = {'HEIGHT': 540, 'WIDTH': 960, 'PADDING': 8, 'VFOV': 30.}
    common_cfg = {'DATASET': 'CITY_SAMPLE', 'BLDG_INST_RANGE': [10, 30000], 'BLDG_INST_MULTIPLIER': 4,
                  'ROAD_CID': 1, 'BG_FACADE_CID': 8, 'BG_CAR_CID': 0, 'FACADE_CID': 1, 'ROOF_CID': 2, 'ROOF_OFFSET': 1,
                  'CAR_INST_RANGE': [30000, 32767], 'N_CAR_CLASSES': 6, 'CAR_BEV_FILE': None, 'N_VOXEL_SAMPLES': 6}
    rows = []
    for name in ['current_code', 'checkpoint_declared', 'geometry']:
        (out / name).mkdir()
    native_render_bldg = native.render_bldg
    patch = (135, 240)
    with torch.inference_mode():
        # Same final cameras as the completed GoogleEarth comparison. Last view is
        # recomputed, not copied, to check saved identity/style and revisit stability.
        for index in [14, 31, 45, 47]:
            cam = next(c for c in cameras if c['index'] == index)
            t = time.perf_counter()
            rays = native.get_voxel_intersection_perspective(volume, cam['citydreamer_pose'], cfg, 30., 6)
            first = rays[0][0, :, :, 0, 0]
            visible = native.get_visible_buildings(first, common_cfg)
            assert all(int(b) in stats and int(b) in styles for b in visible)
            bg_image = native.render_bg(patch, bg, hf_bg, *rays, None, volumes, cfg, common_cfg)
            torch.cuda.synchronize()
            bg_seconds = time.perf_counter() - t
            fp = first.cpu().numpy()
            depth = rays[1][0, :, :, 0, 0, 0].cpu().numpy() / 4
            np.savez_compressed(out / 'geometry' / f'{index:04d}.npz', first_instance=fp, first_depth_metres=depth)
            bg_pixels = None
            conditions = ['current_code', 'checkpoint_declared'] if index != 47 else ['checkpoint_declared']
            for condition in conditions:
                start = time.perf_counter()
                inst = dict(common_cfg)
                hf = hf_bg
                if condition == 'checkpoint_declared':
                    inst.update(FACADE_CID=7, ROOF_CID=8)
                    hf = hf_bldg
                img = bg_image.clone()
                for b in visible.tolist():
                    fg, mask = native_render_bldg(patch, bldg, b, hf, *rays, stats[b], styles[b], volumes, cfg, inst)
                    img = img * (1 - mask) + fg * mask
                assert torch.isfinite(img).all(), 'Nonfinite neural output'
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                rgb = (helpers.tensor_to_image(img, 'RGB') * 255).astype(np.uint8)
                dest = out / condition / f'{index:04d}.png'
                assert cv2.imwrite(str(dest), rgb[..., ::-1])
                building = (fp >= 10) & (fp < 30000)
                if bg_pixels is None:
                    bg_pixels = rgb[~building].copy()
                else:
                    assert np.array_equal(bg_pixels, rgb[~building]), 'Semantic ablation changed background'
                row = {'index': index, 'tag': cam['tag'], 'condition': condition, 'sha256': sha(dest),
                       'background_raycast_seconds_shared': bg_seconds, 'building_render_seconds': seconds,
                       'visible_buildings': len(visible), 'building_pixels': int(building.sum()),
                       'allocated_gib': torch.cuda.memory_allocated()/2**30}
                if index == 47:
                    row['revisit_frame'] = 14
                    row['revisit_hash_equal'] = sha(dest) == sha(out / condition / '0014.png')
                    assert row['revisit_hash_equal']
                rows.append(row)
                progress(phase='render_complete', **row)
            del img, rays, first, bg_image
    after = {p: sha(p) for p in hashes}
    assert hashes == after
    report = {'status': 'CITYTOPIA_MAP10_NATIVE_COMPARISON_RENDERED', 'gpu': gpu,
              'wall_seconds': time.perf_counter()-started, 'max_seconds': budget,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'training': False, 'source_hashes_unchanged': True, 'source_hashes': hashes,
              'spatial_alignment': 'Same registered Map10 source; 4 units/metre conversion verified against all saved cameras.',
              'screen_space_enhancement': False, 'native_postprocess': 'Unmodified native RenderCNN and native road blur, common to both new-weight conditions.',
              'source_geometry': 'Map footprint/height extrusion, not learned new surfaces; imputed heights retained.',
              'quality_decision': 'PENDING_VISUAL_INSPECTION', 'frames': rows}
    save(out / 'RESULT.json', report)
    progress(phase='finished', wall_seconds=report['wall_seconds'], peak_allocated_gib=report['peak_allocated_gib'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[4, 5], required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args()
    assert 0 < args.max_seconds <= 900
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUIDS[args.gpu]
    row = subprocess.check_output(['nvidia-smi', '-i', UUIDS[args.gpu], '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    used, free = map(int, row.strip().split(','))
    assert used < 256 and free > 75000, ('GPU occupied; do not preempt', row)
    out = ROOT / 'run_v1'
    out.mkdir(exist_ok=False)
    def timeout(*unused):
        raise TimeoutError('Bounded comparison deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        execute(args.gpu, out, args.max_seconds)
    except BaseException as e:
        save(out / 'FAILURE.json', {'error': repr(e), 'traceback': traceback.format_exc(), 'gpu': args.gpu})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()

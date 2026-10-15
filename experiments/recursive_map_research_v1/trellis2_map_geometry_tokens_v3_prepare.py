"""One final bounded shared geometry-token experiment; no new target worlds."""
from paths import JOB, read, save, sha
import shutil
import numpy as np

OUT = JOB / 'map_object_world_v1/trellis2_map_geometry_tokens_v3'
PARENT = JOB / 'map_object_world_v1/trellis2_map_learning_v1/trial_fix1'
SEEDS = [953101, 953102]
ARMS = ['original', 'separated']
RESOLUTION = 1024


if __name__ == '__main__':
    assert read(OUT / 'NATIVE_SHAPE_PROBE.json')['status'] == 'ACTUAL_NATIVE_ENCODER_DECODER_INTERFACE_PASS'
    assert read(OUT / 'CODEC_VISUAL_GATE.json')['actual_images_inspected']
    assert not (OUT / 'PLAN.json').exists()
    previous = read(PARENT / 'PLAN.json')
    shutil.copytree(PARENT / 'real_input', OUT / 'real_input')
    with np.load(OUT / 'real_input/map.npz') as source:
        edited = {key: source[key].copy() for key in source.files}
    vacated = edited['footprints'][-1] > 0
    edited['footprints'][-1] = 0
    edited['footprint_known'][-1] = 0
    edited['object_summary'][-1] = 0
    edited['object_valid'][-1] = False
    # Removing an object must specify known open space, not merely forget it.
    edited['open_space'][0][vacated] = 1
    edited['open_space_known'][0][vacated] = 1
    edit_path = OUT / 'real_input/explicit_remove_last_building_map.npz'
    with edit_path.open('xb') as handle:
        np.savez_compressed(handle, **edited)
    save(OUT / 'EDIT_INPUT_REGISTRATION.json', {'before_new_optimizer_or_asset_outputs': True,
         'source_map_sha256': sha(OUT / 'real_input/map.npz'), 'edit_sha256': sha(edit_path),
         'edit': 'Remove fixed last roof proxy and mark its former region known-open',
         'remaining_object_slots': int(edited['object_valid'].sum()),
         'no_generated_geometry_editing': True,
         'old_v2_forget_only_edit_preserved_not_reinterpreted': True})
    for name in ['targets', 'target_records']:
        (OUT / name).mkdir()
    parents = {str(seed): read(PARENT / f'seed{seed}/CHECKPOINT_000100.json') for seed in SEEDS}
    for record in parents.values():
        assert sha(record['path']) == record['sha256']
    plan = {**previous, 'status': 'REGISTERED_SHARED_NATIVE_GEOMETRY_TOKENS',
        'seeds': SEEDS, 'arms': ARMS, 'max_updates': 300, 'checkpoints': [20, 100, 300],
        'starting_parent_updates': 100, 'parent_checkpoints': parents,
        'single_new_mechanism': 'Shared learned map geometry condition tokens in native SS and ShapeFlow cross-attention',
        'geometry_head': 'Mean native XYZ context over Z -> 256 map tokens, 128->256->1024 shared head',
        'trainable': 'existing map branch + one shared map condition head; no native backbone weights',
        'training_targets': 'Same legitimate 96 TRAIN meshes; frozen native SS codec plus frozen native TRELLIS.2 32D shape encoder',
        'target_voxel_resolution': RESOLUTION,
        'fine_flow': 'slat_flow_img2shape_dit_1_3B_1024_bf16',
        'pipeline': 'official direct 1024 for BOTH arms: native SS support64 -> native ShapeFlow1024 -> texture1024',
        'original_arm': 'Reference image tokens concatenated with map tokens in geometry; same reference also texture',
        'separated_arm': 'Map tokens ONLY for ALL geometry; reference image enters texture ONLY',
        'objective': 'Only original native flow MSE at SS and shape stages, equal 1/2 stage weighting, effective batch2',
        'no_added_geometry_loss_or_sweep': True,
        'teacher_shape_coords_TRAIN_only': True,
        'normal_inference': 'Predicted SS support and sampled ShapeFlow; no target mesh/encoder/coordinates',
        'sampling': 'Official samplers/CFG unchanged; fixed coordinate-addressed shape/texture noise matched for edits',
        'noise_lattice': 'Fixed64^3 Gaussian grids; shape stage1, texture stage2',
        'data_families_and_order_unchanged': True,
        'real_paired_TRAIN_verified': 0,
        'real_map_eval': '10.jpg held out; approximate normalized roof proxies, not measured reconstruction',
        'appearance_quality_limit': 'Same stylized fixed reference, not recovered real facades',
        'hard_train_seconds_per_job': 1800,
        'hard_generation_seconds_per_job': 900,
        'no_third_mechanism_or_automatic_extension': True}
    save(OUT / 'PLAN.json', plan)
    for arm in ARMS:
        (OUT / arm).mkdir()
    save(OUT / 'REGISTRATION_FREEZE.json', {
         str(OUT / 'PLAN.json'): sha(OUT / 'PLAN.json'),
         str(PARENT / 'PLAN.json'): sha(PARENT / 'PLAN.json'),
         **{record['path']: record['sha256'] for record in parents.values()}})
    print('FINAL_GEOMETRY_TOKEN_MECHANISM_REGISTERED', RESOLUTION, len(plan['records']), flush=True)

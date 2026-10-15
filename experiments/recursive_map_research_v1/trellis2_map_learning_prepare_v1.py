"""Register one bounded, frozen-prior map-adapter trial from legitimate old TRAIN.

Does not create target geometry or read the historical DEV/sealed datasets.
The independent real-map demonstration is not a measured reconstruction target.
"""
from paths import JOB, NATIVE, read, save, sha
from trellis2_prepare_v1 import DEST as INSTALL, REPO
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

OUT = JOB / 'map_object_world_v1/trellis2_map_learning_v1'
SEEDS = [953101, 953102]


def masks(record):
    """Use registered input polygons, never target heights or generated bboxes."""
    objects = record['objects']
    layers = []
    summaries = []
    for obj in objects:
        im = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(im)
        components = obj.get('footprint_components_m')
        if components is None:
            components = [{'exterior': obj['footprint_polygon_m'], 'holes': []}]
        def pixels(p):
            return [((float(x)+16)/32*64-.5, (float(y)+16)/32*64-.5) for x, y in p]
        for component in components:
            draw.polygon(pixels(component['exterior']), fill=255)
            for hole in component.get('holes', []):
                draw.polygon(pixels(hole), fill=0)
        support = np.asarray(im).astype(np.float32)/255
        assert support.any(), ('EMPTY_REGISTERED_OBJECT', record['scene_id'], obj['object_id'])
        layers.append(support)
        y, x = np.where(support > 0)
        summaries.append([(x.mean()+.5)/64-.5, (y.mean()+.5)/64-.5,
                          (x.max()-x.min()+1)/64, (y.max()-y.min()+1)/64,
                          float(support.mean()), 1., 0., 1.])
    assert 3 <= len(objects) <= 5
    with np.load(record['map_path']) as z:
        m = z['map_tensor'].astype(np.float32)
    # Registered map row convention is verified separately before optimizer launch.
    # Original raster row0 is +Y; branch raster row0 is -Y. Flip ONLY the input.
    road = m[1].reshape(64,4,64,4).mean((1,3))[::-1].copy()[None]
    opened = m[2].reshape(64,4,64,4).mean((1,3))[::-1].copy()[None]
    return {'footprints':np.stack(layers), 'roads':road, 'open_space':opened,
            'confidence':np.ones((1,64,64),np.float32),
            'object_summary':np.array(summaries,np.float32),
            'object_valid':np.ones(len(objects),bool)}


def prepare():
    assert not (OUT/'PLAN.json').exists(), 'No rewriting registration'
    catalogue = read(JOB/'TRAIN_DATA.json')
    families = sorted({catalogue['records'][sid]['family_id'] for sid in catalogue['original_order']})
    # Reserve independent families for this NEW adapter, before any learned output.
    train_families = families[:32]
    heldout_families = families[32:40]
    records = []
    for sid in catalogue['original_order']:
        r = catalogue['records'][sid]
        if r['family_id'] not in train_families + heldout_families:
            continue
        assert r['split'] == 'train' and 'map10' not in json.dumps(r).lower()
        manifest = Path(r['map_path']).parent/'MANIFEST.json'
        meta = read(manifest)
        assert meta['scene_id'] == sid and meta['split'] == 'train'
        mesh = Path(meta['target_mesh_path'])
        assert sha(mesh) == meta['mesh_sha256']
        assert sha(Path(r['map_path'])) == meta['map_sha256']
        with np.load(mesh) as z:
            assert len(z['faces']) > 0 and np.isfinite(z['vertices_world_m']).all()
            assert np.allclose(z['vertices_native'][:,:2],z['vertices_world_m'][:,:2]/32, atol=1e-5)
            assert np.allclose(z['vertices_native'][:,2],(z['vertices_world_m'][:,2]-14)/32,atol=1e-5)
        package = OUT/'data'/f'{sid}_map.npz'
        package.parent.mkdir(parents=True,exist_ok=True)
        with package.open('xb') as f:
            np.savez_compressed(f, **masks(meta))
        records.append({'scene_id':sid,'family_id':r['family_id'],
            'split':'adapter_TRAIN' if r['family_id'] in train_families else 'adapter_HELDOUT',
            'original_split':'train','map_input':str(package),'map_sha256':sha(package),
            'registered_map':r['map_path'],'registered_map_sha256':meta['map_sha256'],
            'structure_latent':r['structure'],'structure_sha256':sha(r['structure']),
            'source_manifest':str(manifest),'manifest_sha256':sha(manifest),
            'target_mesh':str(mesh),'mesh_sha256':meta['mesh_sha256'],
            'object_count':len(meta['objects']),
            'provenance':'licensed authored mixed-style multi-building assets; NOT real-city paired reconstruction',
            'quality_limit':meta['target_quality']})
    assert len(records)==120 and len({r['family_id'] for r in records if r['split']=='adapter_TRAIN'})==32
    # Do not use Map10 pixels, features, or geometry during optimization.
    appearance = JOB/'map_object_world_v1/trellis2_resume_v3/inputs/official_building/input.png'
    assert sha(appearance)=='f25e24956eec3a186d3da8cb79ecabcf76e4cee4b083f472cd64037b2fbd57b3'
    plan = {'status':'REGISTERED_PENDING_NATIVE_TARGET_AND_GRADIENT_PREFLIGHT',
        'model':'official frozen TRELLIS.2 SparseStructureFlowModel + trainable object-map residual branch',
        'seeds':SEEDS,'max_updates':100,'checkpoints':[20,100],
        'effective_batch':2,'learning_rate':1e-4,'weight_decay':.01,
        'optimizer':'AdamW betas(.9,.95) eps1e-8 grad_clip1',
        'objective':'only original native SS flow MSE sigma_min1e-5, logitNormal(mean1,std1)',
        'no_new_loss_or_prior_or_decoder_training':True,
        'native_target_contract':'original immutable SS codec S0 [1,8,16,16,16], not old 8D Detail SLAT; native decoder replay required',
        'appearance_condition':str(appearance),'appearance_sha256':sha(appearance),
        'appearance_provenance':'official stylized building reference, NOT verified real urban photos',
        'TRAIN_counts':{'scenes':96,'families':32},'HELDOUT_counts':{'scenes':24,'families':8},
        'real_map_mesh_training_pairs_verified':0,'real_map_evaluation':'10.jpg separate fixed block, normalized screenshot layout only',
        'map10_used_in_training':False,'historical_DEV_or_sealed_test_used':False,
        'geometry_injection':'object-centric map residuals inside SS denoiser blocks4,14,24 before frozen shape/texture generation; no post-placement',
        'height_known':False,'training_geometry_world_unit':'native=(world_m-[0,0,14])/32; canonical ground=-.4375',
        'input_raster_convention':'row0=-Y, col0=-X; original registered raster flipped in Y, no target modified',
        'trial_scope':'one fixed small-data mechanism trial, not convergence or real-world reconstruction proof',
        'hard_GPU_seconds_per_training_seed':1800,'automatic_extension':False,
        'baseline':'same frozen prior/no map branch, same reference and generation noise; plus historical Map10 output separately',
        'replay_required':['correct map','shuffled correspondence','remove one predeclared building'],
        'records':records}
    save(OUT/'PLAN.json',plan)
    save(OUT/'SESSION_STATE.json',{'stage':'PREREGISTERED','updates':0,'training_launched':False,
        'blocker_if_native_replay_fails':'existing SS target cannot be substituted for TRELLIS.2 target',
        'quality_gate':'actual reloaded real-map assets, NOT loss or export success'})
    print('REGISTERED',len(records),'Map10_TRAIN=False','real_paired_TRAIN=0',flush=True)


if __name__=='__main__':
    prepare()

"""Bounded streaming inference; same v24 generation, earlier native mesh publication.

No training, resolution reduction, decimation, scene repair or repeat samples.
The preview is the actual decoded shape, not procedural placeholder geometry.
"""
from pathlib import Path
import os, sys, json, time, shutil

INDEX = int(os.environ.get('V25_CHUNK_INDEX', '2'))
assert INDEX in (2, 3)
_source = Path(__file__).with_name('v24_native_job.py').read_text()
_source = _source.split("\nif __name__=='__main__':")[0]
exec(compile(_source, str(Path(__file__).with_name('v24_native_job.py')), 'exec'), globals())
OUT = JOB / f'map_object_world_v1/trellis2_realtime_v25/chunks/stream-0--{INDEX}'
SEED = 942225 + INDEX - 2
_base_generate = generate
_base_export = fixed_world_export


def prepare():
    assert not OUT.exists(), 'Preserve existing attempt'
    OUT.mkdir(parents=True)
    for name in ['logs', 'work', 'cache', 'tmp', 'extensions']:
        (OUT / name).mkdir()
    prior = JOB / 'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1'
    p = read(prior / 'PLAN.json')
    for name in ['input.png', 'ROAD_OBSERVATION.npz']:
        shutil.copyfile(prior / name, OUT / name)
    scale = p['boundary']['fixed_world_frame']['global_native_to_meter_scale']
    p.update(seed=SEED, input=str(OUT / 'input.png'), input_sha256=sha(OUT / 'input.png'),
             chunk_id=f'stream-0--{INDEX}', fixed_chunk_translation_GLTF=[0, 0, -INDEX * scale],
             parent_chunk_id=f'road-boundary-0--{INDEX-1}',
             generation_trigger='Actual camera movement, predictive forward requests; no inference before request',
             streaming_only_change='Publish native decoded geometry before appearance/export; sampler unchanged',
             no_automatic_extension=True, GPU_limit_seconds=600)
    p['boundary']['scope'] = 'Inherited straight-road hypothesis in fixed parent frame, NOT map control or seamless continuation'
    save(OUT / 'PLAN.json', p)
    files = [Path(__file__), JOB/'v24_native_job.py', JOB/'v23_native_job.py',
             JOB/'v24_boundary_sampler.py', JOB/'trellis2_native_street_v17.py',
             OUT/'PLAN.json', OUT/'input.png', OUT/'ROAD_OBSERVATION.npz', Path(p['runtime'])/'pipeline.json']
    save(OUT/'SOURCE_FREEZE.json', {str(f):sha(f) for f in files})
    status('waiting_camera', no_inference_yet=True)
    print('V25_PREPARED', INDEX, SEED, flush=True)


def stage(name, **data):
    import torch
    torch.cuda.synchronize()
    p = OUT/'PROGRESS.next'
    value = dict(stage=name, epoch=time.time(), chunk=INDEX, **data)
    p.write_text(json.dumps(value)); os.replace(p, OUT/'PROGRESS.json')
    with (OUT/'STAGES.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n')


def generate():
    # Hook only publication/timing, leaving the native random stream and tensors intact.
    import trellis2_native_street_v17 as native
    sys.path.insert(0, str(native.REPO))
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    original_load = Trellis2ImageTo3DPipeline.from_pretrained
    @classmethod
    def load(cls, *args, **kwargs):
        stage('loading_model')
        pipe = original_load(*args, **kwargs)
        for method, label in [('get_cond','image_condition'), ('sample_sparse_structure','structure'),
                              ('sample_shape_slat_cascade','shape'), ('sample_tex_slat','appearance')]:
            original = getattr(pipe, method)
            def wrapped(*a, _method=original, _label=label, **kw):
                stage(_label)
                result = _method(*a, **kw)
                stage(_label+'_done')
                return result
            setattr(pipe, method, wrapped)
        original_shape = pipe.decode_shape_slat
        def decode_shape(*a, **kw):
            stage('decode_geometry')
            result = original_shape(*a, **kw)
            # Export exact generated geometry in the SAME predetermined world frame.
            import numpy as np, trimesh
            mesh = result[0][0]
            vertices = mesh.vertices.detach().float().cpu().numpy()
            faces = mesh.faces.detach().cpu().numpy()
            axis = np.array([[1,0,0],[0,0,1],[0,-1,0]], np.float32)
            scene = trimesh.Scene(trimesh.Trimesh(vertices=vertices@axis.T, faces=faces, process=False))
            matrix = np.array(read(OUT/'PLAN.json')['boundary']['fixed_world_frame']['glTF_root_matrix']).reshape(4,4).T
            scene.apply_transform(matrix)
            tmp = OUT/'geometry_preview.next.glb'; path=OUT/'geometry_preview.glb'
            scene.export(tmp, file_type='glb'); os.replace(tmp, path)
            save(OUT/'PREVIEW.json', dict(epoch=time.time(), sha256=sha(path), bytes=path.stat().st_size,
                 vertices=len(vertices), faces=len(faces), appearance='neutral preview, actual AI shape',
                 generated_geometry=True, fixed_frame=True, geometry_repair=False))
            stage('geometry_preview_ready', vertices=len(vertices), faces=len(faces))
            return result
        pipe.decode_shape_slat = decode_shape
        return pipe
    Trellis2ImageTo3DPipeline.from_pretrained = load
    # v24 also installs its road sampler around the above loader.
    _base_generate()
    stage('native_asset_exported')


if __name__ == '__main__':
    try:
        if '--prepare' in sys.argv: prepare()
        elif '--generate' in sys.argv: generate()
        elif '--detach' in sys.argv:
            with (OUT/'owner.log').open('x') as f:
                child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--run'],cwd=JOB,
                    stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(OUT/'OWNER.json',dict(pid=child.pid,epoch=time.time(),bounded=True))
            print(json.dumps(dict(pid=child.pid)),flush=True)
        else: run()
    except Exception as e:
        if OUT.exists(): status('failed',error=str(e),no_retry=True)
        raise

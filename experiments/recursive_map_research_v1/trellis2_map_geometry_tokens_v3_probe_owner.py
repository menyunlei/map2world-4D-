"""One bounded native shape interface probe; never an optimizer launch."""
from paths import JOB, save, sha
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded

OUT = JOB / 'map_object_world_v1/trellis2_map_geometry_tokens_v3'

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for name in ['audit', 'logs', 'gpu_cost']:
        (OUT / name).mkdir(exist_ok=True)
    files = ['trellis2_map_geometry_tokens_v3_probe.py', 'trellis2_map_geometry_tokens_v3_probe_owner.py']
    freeze = {str(JOB / name): sha(JOB / name) for name in files}
    save(OUT / 'PROBE_SOURCE_FREEZE.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in freeze.items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    fn(GPUS[0], 'geometry_tokens', 953101, 'TRAIN_shape_codec_probe',
       'trellis2_map_geometry_tokens_v3_probe.py', [], seconds=300)

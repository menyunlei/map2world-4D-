"""One frozen-asset display repair after probe encoding already succeeded."""
from paths import JOB, save, sha
from map_object_composition_owner_v1 import bounded
from trellis2_map_geometry_entry_v2_owner import GPUS

if __name__ == '__main__':
    OUT = JOB / 'map_object_world_v1/trellis2_map_geometry_tokens_v3'
    freeze = {str(JOB / name): sha(JOB / name) for name in [
         'trellis2_map_geometry_tokens_v3_probe_readout_fix1.py',
         'trellis2_map_geometry_tokens_v3_probe_readout_owner.py']}
    for name in ['PROBE_NATIVE_SHAPE.npz', 'CODEC_PROBE_GEOMETRY.pt']:
        freeze[str(OUT / name)] = sha(OUT / name)
    save(OUT / 'CODEC_READOUT_FREEZE.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in freeze.items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    fn(GPUS[0], 'geometry_tokens', 953101, 'codec_readout_fix1',
       'trellis2_map_geometry_tokens_v3_probe_readout_fix1.py', [], seconds=120)

"""One bounded render-only retry after missing FFmpeg PATH; no model training."""
from trellis2_map_learning_owner_v1 import OUT, JOB, save, sha, bounded

if __name__ == '__main__':
    freeze = {str(JOB / n): sha(JOB / n) for n in [
        'trellis2_map_learning_motion_fix1.py', 'trellis2_map_learning_motion_owner_fix1.py']}
    save(OUT / 'MOTION_SOURCE_FREEZE_FIX1.json', freeze)
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in freeze.items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    fn({'index': 2, 'uuid': 'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},
       'trellis2_map', 953101, 'lateral_motion_fix1', 'trellis2_map_learning_motion_fix1.py', [], seconds=240)

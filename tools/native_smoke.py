"""Portable wrapper around the EXISTING v17 native generator. No automatic GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from reproduce import ROOT, JOB, LOCK, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root', type=Path, required=True,
                   help='Directory containing environment/ and vendor/TRELLIS.2/')
    p.add_argument('--runtime', type=Path, required=True, help='Directory with configured pipeline.json')
    p.add_argument('--input', type=Path, required=True, help='Already padded RGB input; no background removal')
    p.add_argument('--output-name', default='reproduction_native_smoke')
    p.add_argument('--seed', type=int, default=942217)
    p.add_argument('--gpu', help='Authorized physical GPU UUID, required with --run')
    p.add_argument('--run', action='store_true', help='Explicitly run ONE native inference; default is CPU checks only')
    a = p.parse_args()
    a.native_root = a.native_root.resolve()
    repo = a.native_root / 'vendor/TRELLIS.2'
    rev = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != json.loads(LOCK.read_text())['backbone']['commit']:
        raise ValueError('Backbone commit mismatch: ' + rev)
    if not a.input.is_file() or not (a.runtime / 'pipeline.json').is_file():
        raise FileNotFoundError('Missing input or configured pipeline.json')
    output = (JOB / 'map_object_world_v1' / a.output_name).resolve()
    if not output.is_relative_to((JOB / 'map_object_world_v1').resolve()) or output.exists():
        raise ValueError('Output must be a fresh directory under map_object_world_v1')
    print(json.dumps({'source_commit': rev, 'input_sha256': digest(a.input),
                      'pipeline_sha256': digest(a.runtime / 'pipeline.json'),
                      'output': str(output), 'will_run': a.run}), flush=True)
    if not a.run:
        return
    if not a.gpu or not a.gpu.startswith('GPU-'):
        raise ValueError('--run requires an authorized physical GPU UUID')
    import fcntl  # Native inference is Linux-only; CPU argument checks remain portable.
    locks = JOB / 'gpu_leases'
    locks.mkdir(exist_ok=True)
    with (locks / (a.gpu + '.lock')).open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == a.gpu for line in owners.splitlines()):
            raise RuntimeError('Selected GPU already has a compute owner; refusing to share or stop it')
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        sys.path.insert(0, str(JOB))
        import paths
        # In-process overrides only. Historical files and server paths are untouched.
        paths.ROOT, paths.JOB = ROOT, JOB
        import trellis2_native_street_v17 as native
        native.DEST, native.REPO = a.native_root, repo
        native.OUT, native.SEED = output, a.seed
        paths.save(output / 'PLAN.json', {
            'input': str(a.input.resolve()), 'input_sha256': digest(a.input),
            'runtime': str(a.runtime.resolve()), 'seed': a.seed,
            'pipeline_type': '1024_cascade', 'zero_training': True, 'map_control_claim': False
        })
        native.generate()


if __name__ == '__main__':
    main()

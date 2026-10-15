"""Finish the registered v19 assets once GPUs 0/1 have been released."""
from pathlib import Path
import os
import subprocess
import sys
import time

from paths import JOB, save, sha
from trellis2_map_volume_v19 import OUT
from trellis2_map_volume_v19_owner import run, verify


def main():
    verify()
    assert (OUT / 'COARSE_COMPLETE.json').exists()
    assert not (OUT / 'GENERATE_COMPLETE.json').exists(), 'Generation already complete'
    assert not any((OUT / m / 'ASSET.json').exists() for m in ['correct', 'edit']), 'Preserve partial assets; inspect before resuming'
    save(OUT / 'RESUME_SOURCE.json', {str(Path(__file__).resolve()): sha(__file__)})
    try:
        run('generate')
        run('views')
        subprocess.run([sys.executable, '-B', str(JOB / 'trellis2_map_volume_v19_finish.py')],
                       cwd=JOB, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
    except Exception as exc:
        save(OUT / 'RESUME_FAILURE.json', {'type': type(exc).__name__, 'error': str(exc),
             'epoch': time.time(), 'no_automatic_retry': True, 'preserve_partials': True})
        raise


if __name__ == '__main__':
    if '--detach' in sys.argv:
        with (OUT / 'resume_owner_retry1.log').open('x') as handle:
            child = subprocess.Popen([sys.executable, '-B', '-u', __file__], cwd=JOB,
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(OUT / 'RESUME_OWNER.json', {'pid': child.pid, 'finite': True, 'no_automation': True,
             'stages': ['generate', 'views', 'CPU finish'], 'same_registered_experiment': True})
        print('RESUME_OWNER', child.pid, flush=True)
    else:
        main()

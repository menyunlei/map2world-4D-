"""Four UUID-checked bounded jobs, no automation or unrelated job termination."""
from paths import JOB, read, save, sha
from trellis2_map_geometry_entry_v2_prepare import OUT, ARMS, SEEDS
from map_object_composition_owner_v1 import bounded
import concurrent.futures
import fcntl
import subprocess
import sys

GPUS = [
    {'index': 0, 'uuid': 'GPU-e0a56bf9-85ef-c658-67fa-457b76585254'},
    {'index': 1, 'uuid': 'GPU-eddd0d38-b3e3-66b1-f60b-71605e5e68aa'},
    {'index': 2, 'uuid': 'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},
    {'index': 3, 'uuid': 'GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d'}]


def run():
    fn = bounded()
    fn.__globals__['NEW'] = OUT
    def verify():
        for path, digest in read(OUT / 'SOURCE_FREEZE.json').items():
            assert sha(path) == digest, path
    fn.__globals__['frozen'] = verify
    with (OUT / 'owner.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(fn, GPUS[i*2], 'entry_'+arm, SEEDS[0], 'preflight',
                    'trellis2_map_geometry_entry_v2_train.py',
                    ['--arm', arm, '--seed', str(SEEDS[0]), '--preflight'], seconds=240)
                    for i, arm in enumerate(ARMS)]
            for job in jobs:
                job.result()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            jobs = [pool.submit(fn, GPUS[i*2+j], 'entry_'+arm, seed, 'train300',
                    'trellis2_map_geometry_entry_v2_train.py', ['--arm', arm, '--seed', str(seed)], seconds=1200)
                    for i, arm in enumerate(ARMS) for j, seed in enumerate(SEEDS)]
            for job in jobs:
                job.result()
    save(OUT / 'TRAINING_COMPLETE.json', {'arms': ARMS, 'seeds': SEEDS,
         'additional_updates_each': 300, 'total_additional_updates': 1200, 'automatic_extension': False})


if __name__ == '__main__':
    if '--detach' in sys.argv:
        with (OUT / 'owner.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', '-u', __file__], stdout=log,
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(OUT / 'OWNER.json', {'pid': process.pid, 'GPUs': GPUS, 'finite': True,
             'hard_steps': 300, 'no_automation': True})
        print('ENTRY_OWNER', process.pid, flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(OUT / 'OWNER_FAILURE.json', {'type': type(error).__name__, 'error': str(error),
                                            'negatives_preserved': True})
            raise

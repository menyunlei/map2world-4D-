"""One finite train/evaluate run with physical GPU occupancy accounting."""
from paths import JOB, read, save, sha
from trellis2_rehearsal_v14_train import OUT, ARM, SEEDS
import subcell_owner_v1 as lease
import types, threading, concurrent.futures, subprocess, sys, os, fcntl

def verify():
    for path, h in read(OUT/'SOURCE_FREEZE.json').items():
        assert sha(path) == h, ('SOURCE_CHANGED', path)

def bounded():
    scope = {**lease.run.__globals__, 'NEW': OUT, 'frozen': verify,
       'costs': lambda: sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
       'guard': threading.Lock(), 'active': {}, 'halt': threading.Event()}
    fn = types.FunctionType(lease.run.__code__, scope, lease.run.__name__, lease.run.__defaults__)
    assert 114900 in fn.__code__.co_consts
    # Reserve 40 seconds for termination within the preregistered occupancy ceiling.
    fn.__code__ = fn.__code__.replace(co_consts=tuple(5360 if c == 114900 else c for c in fn.__code__.co_consts))
    return fn

def run():
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(); fn = bounded(); gpus = read(OUT/'PLAN.json')['GPUs']
        assert [g['index'] for g in gpus] == [0, 1]
        def task(i, seed):
            args = ['--arm', ARM, '--seed', str(seed)]
            fn(gpus[i], ARM, seed, 'preflight', 'trellis2_rehearsal_v14_train.py', args+['--preflight'], seconds=180)
            probe = read(OUT/'small'/ARM/f'seed{seed}/preflight/PREFLIGHT.json')
            assert probe['lora_parameters'] == 245760
            assert all(r['head_grad_norm'] > 0 and r['lora_grad_norm'] > 0 for r in probe['rows'])
            fn(gpus[i], ARM, seed, 'train1000', 'trellis2_rehearsal_v14_train.py', args, seconds=1500)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for f in [pool.submit(task, i, seed) for i, seed in enumerate(SEEDS)]:
                f.result()
        save(OUT/'TRAINING_COMPLETE.json', {'runs': 2, 'updates_each': 1000,
             'no_more_training_queued': True, 'historical_control_reused': True})
        def evaluate(i, seed):
            for mode, seconds in [('structure', 2400), ('mesh', 1000), ('views', 300)]:
                fn(gpus[i], 'evaluation', seed, mode, 'trellis2_rehearsal_v14_eval.py',
                   ['--mode', mode, '--seed', str(seed)], seconds=seconds)
        errors = []
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for f in [pool.submit(evaluate, i, seed) for i, seed in enumerate(SEEDS)]:
                try: f.result()
                except Exception as e: errors.append(str(e))
        subprocess.run([sys.executable, '-B', str(JOB/'trellis2_rehearsal_v14_score.py')], cwd=JOB,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        save(OUT/'COMPLETE.json', {'errors': errors, 'no_more_jobs': True,
             'status': 'COMPLETE' if not errors else 'PARTIAL_STOP',
             'GPU_seconds': sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json'))})

if __name__ == '__main__':
    if '--detach' in sys.argv:
        verify()
        with (OUT/'owner.log').open('x') as h:
            p = subprocess.Popen([sys.executable, '-B', '-u', __file__], cwd=JOB, stdout=h,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(OUT/'OWNER.json', {'pid': p.pid, 'finite': True, 'only_GPUs': [0, 1],
                              'no_automation': True, 'GPU_cap_seconds': 5400})
        print('REPLAY_OWNER', p.pid, flush=True)
    else:
        try: run()
        except Exception as e:
            save(OUT/'FAILURE.json', {'type': type(e).__name__, 'error': str(e), 'preserve_partials': True})
            raise

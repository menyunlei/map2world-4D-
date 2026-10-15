"""Finite two-A800 owner for the fixed 100-update schedule experiment."""
from paths import *
import concurrent.futures
import fcntl
import inspect
import os
import subprocess
import sys
import threading
import time
import launch_training as infrastructure
import subcell_owner_v1 as previous

DEST = JOB / 'dice_schedule_probe_v1'
GPUS = previous.GPUS[:2]
START = time.time()
guard = threading.Lock()
active = {}
halt = threading.Event()


def costs():
    return sum(read(path)['gpu_seconds'] for path in (DEST / 'gpu_cost').glob('*.json'))


def frozen():
    infrastructure.verify()
    for path, digest in read(DEST / 'PLAN.json')['frozen_source_and_inputs'].items():
        assert sha(path) == digest, ('FROZEN_CHANGED', path)


# Reuse the established physical-UUID lease, live ownership check, and cost
# watchdog. Its historical shared GPU-hour cap was revoked by the human; this
# owner remains bounded by exactly 100 updates and per-job wall time.
source = inspect.getsource(previous.run)
old_check = "assert costs()+sum(time.time()-v[0] for v in active.values())<114900,'SHARED_GPU_CAP'"
old_stop = "if costs()+sum(time.time()-v[0] for v in active.values())>=114900:halt.set()"
assert source.count(old_check) == source.count(old_stop) == 1
source = source.replace(old_check, "assert not halt.is_set(), 'OWN_OPERATIONAL_STOP'")
source = source.replace(old_stop, "if time.time()-START>=43200:halt.set()")
scope = {**vars(previous), 'NEW': DEST, 'costs': costs, 'frozen': frozen,
         'START': START, 'guard': guard, 'active': active, 'halt': halt}
exec(compile(source, str(JOB / 'subcell_owner_v1.py'), 'exec'), scope)
run = scope['run']


def policy_for(arm, seed, tag):
    work = DEST / 'work' / f'subcell_{arm}_{seed}_{tag}'
    policy = read(infrastructure.policy(work, 'g2_b', tag))
    policy['write_root'] = str(DEST)
    policy['probe_output'] = f'ISOLATION_{arm}_{seed}_{tag}.json'
    needed = ['dice_schedule_train_v1.py', 'subcell_runtime_v1.py',
              'query_boundary_runtime_v1.py']
    policy['read_paths'] += [str(JOB / file) for file in needed]
    old = JOB / 'narrow_structure_v1'
    policy['read_paths'] += [str(old / name) for name in ('TARGETS_READY.json', 'TRAIN_FREEZE.json')]
    policy['read_paths'] += list(read(old / 'TRAIN_FREEZE.json')['files'])
    start = JOB / 'runs/g2_b' / f'seed{seed}/CHECKPOINT_000300.json'
    policy['read_paths'] += [str(start), read(start)['path']]
    path = DEST / 'policies' / f'{arm}_{seed}_{tag}.json'
    save(path, policy)
    return path


def train(gpu, arm, seed, until, resume=None, probe=False):
    tag = 'preflight' if probe else f'train{resume or 0}_{until}'
    path = policy_for(arm, seed, tag)
    args = [str(path), str(JOB / 'dice_schedule_train_v1.py'), '--arm', arm,
            '--seed', str(seed), '--until', str(until)]
    if resume:
        args += ['--resume', str(resume)]
    if probe:
        args += ['--probe-only']
    run(gpu, arm, seed, tag, 'sandbox.py', args, seconds=1200 if probe else 2400 if until == 20 else 9000)


def evaluate(gpu, arm, seed, update):
    run(gpu, arm, seed, f'eval{update}', 'dice_schedule_eval_v1.py',
        ['--arm', arm, '--seed', str(seed), '--update', str(update)], seconds=2400)


def parallel(left, right):
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(left[0], *left[1], **left[2]),
                pool.submit(right[0], *right[1], **right[2])]
        for future in jobs:
            future.result()


def main():
    with (DEST / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (DEST / 'COMPLETE.json').exists()
        frozen()
        for arm in ('old', 'corrected'):
            for seed in (1701, 1702):
                assert not (DEST / arm / f'seed{seed}/TRAIN_CURVE.jsonl').exists()
        parallel((train, (GPUS[0], 'old', 1701, 20), {'probe': True}),
                 (train, (GPUS[1], 'corrected', 1701, 20), {'probe': True}))
        parallel((evaluate, (GPUS[0], 'old', 1701, 0), {}),
                 (evaluate, (GPUS[1], 'old', 1702, 0), {}))
        for seed in (1701, 1702):
            for until, resume in ((20, None), (100, 20)):
                parallel((train, (GPUS[0], 'old', seed, until, resume), {}),
                         (train, (GPUS[1], 'corrected', seed, until, resume), {}))
                parallel((evaluate, (GPUS[0], 'old', seed, until), {}),
                         (evaluate, (GPUS[1], 'corrected', seed, until), {}))
        subprocess.run([sys.executable, '-B', str(JOB / 'dice_schedule_score_v1.py')],
                       cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        print('MATCHED_100_COMPLETE', flush=True)


if __name__ == '__main__':
    if '--detach' in sys.argv:
        assert (DEST / 'PLAN.json').exists()
        with (DEST / 'owner.log').open('x') as handle:
            child = subprocess.Popen([sys.executable, '-B', '-u', __file__], stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(DEST / 'OWNER.json', {'pid': child.pid, 'finite_not_automation': True,
            'max_updates_per_arm_seed': 100, 'physical_gpus': [0, 1]})
        print('OWNER', child.pid, flush=True)
    else:
        try:
            main()
        except Exception as error:
            save(DEST / 'FAILURE.json', {'type': type(error).__name__, 'error': str(error),
                'epoch': time.time(), 'preserve_partials': True, 'no_automatic_optimizer_retry': True})
            raise

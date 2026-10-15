"""Two leased GPUs, matched 20/100-update jobs; finite, no automation."""
from paths import *
import concurrent.futures
import fcntl
import os
import subprocess
import sys
import threading
import time
import types
from collections import Counter
import launch_training as infra
import dice_schedule_owner_v1 as previous

DEST = JOB / 'dice_generated_state_v1'
SOURCE = JOB / 'dice_schedule_probe_v1'
ARMS = ('baseline', 'generated')
SEEDS = (1701, 1702)


def prepare():
    assert not DEST.exists(), 'Preserve historical and current runs'
    infra.verify()
    old = read(SOURCE / 'PLAN.json')
    freeze = dict(old['frozen_source_and_inputs'])
    starts = {}
    for seed in SEEDS:
        p = SOURCE / 'corrected' / f'seed{seed}/CHECKPOINT_000100.json'
        meta = read(p)
        assert meta['update'] == 100 and sha(meta['path']) == meta['sha256']
        starts[str(seed)] = str(p)
        freeze[str(p)] = sha(p)
        freeze[meta['path']] = meta['sha256']
    for name in ('dice_generated_state_v1.py', 'dice_generated_state_owner_v1.py',
        'dice_generated_state_eval_v1.py', 'dice_generated_state_score_v1.py',
        'subcell_model_v1.py', 'dice_schedule_owner_v1.py', 'subcell_owner_v1.py'):
        freeze[str(JOB / name)] = sha(JOB / name)
    for name in ('flow_euler.py', 'guidance_interval_mixin.py', 'classifier_free_guidance_mixin.py'):
        p = NATIVE / 'TRELLIS/trellis/pipelines/samplers' / name
        freeze[str(p)] = sha(p)
    freeze[str(NATIVE / 'sources/train_map_native_prior.py')] = sha(NATIVE / 'sources/train_map_native_prior.py')
    for p, digest in freeze.items():
        assert sha(p) == digest
    # Reuse the exact legitimate TRAIN queue; all eval noises remain disjoint.
    queue = old['dice_queues']['corrected']
    selected = [q for q in queue if not q['drop'] and q['t'] < .5]
    assert selected and {q['scene_id'] for q in selected} == set(old['units'])
    plan = {**old, 'arms': list(ARMS), 'starts': starts,
        'start': 'corrected Dice +100 per fixed seed; same branch weights, optimizer reset',
        'dice_queue': queue, 'frozen_source_and_inputs': freeze,
        'sole_intervention': 'Dice input state for conditioned t<0.5 slots; full flow data/targets unchanged',
        'generated_state': {'detach_rollout': True, 'from_current_weights': True,
            'noise': 'same registered TRAIN noise seed as each Dice slot',
            'sampler': 'native 25 Euler intervals, rescale_t=3; last interval shortened to exact existing Dice t',
            'CFG': {'strength':7.5,'interval':[.5,.95]},
            'negative_text':'existing TRAIN empty text', 'no_clean_target_in_rollout':True,
            'criterion':'not drop and t<0.5', 'eligible_count':len(selected),
            'eligible_per_scene':dict(Counter(q['scene_id'] for q in selected))},
        'matching': {'both_arms_roll_out_all_eligible_slots':True,
            'same_NFE_budget':True,'same_train_noise_t_dropout_flow_and_O1':True,
            'same_Dice_frequency_coefficient_and400_forwards':True,
            'inference_uses_original_sampler_no_target':True},
        'max_updates_per_arm_seed':100, 'no_300_extension':True,
        'preflight':'one actual generated-state backward, nonzero finite branch gradient, no optimizer',
        'new_family_or_width_generalization_claim':False}
    save(DEST / 'PLAN.json', plan)
    save(DEST / 'PROGRESS.json', {'stage':'REGISTERED','max_updates':100,'human_review':'PENDING'})
    print('GENERATED_STATE_REGISTERED', len(selected), plan['generated_state']['eligible_per_scene'], flush=True)


def frozen():
    infra.verify()
    for path, digest in read(DEST / 'PLAN.json')['frozen_source_and_inputs'].items():
        assert sha(path) == digest, ('FROZEN_CHANGED', path)


def runner():
    source = previous.run
    scope = {**source.__globals__, 'NEW':DEST,'frozen':frozen,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (DEST/'gpu_cost').glob('*.json')),
        'START':time.time(),'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    return types.FunctionType(source.__code__, scope, source.__name__, source.__defaults__)


def parallel(left, right):
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        fs = [pool.submit(f, *a, **kw) for f,a,kw in (left,right)]
        for f in fs:
            f.result()


def run():
    bounded = runner()
    def train(gpu, arm, seed, until, resume=None, probe=False):
        tag = 'preflight' if probe else f'train{resume or 0}_{until}'
        work = DEST/'work'/f'subcell_{arm}_{seed}_{tag}'
        policy = read(infra.policy(work,'g2_b',f'generatedstate_{arm}_{seed}_{tag}'))
        policy['write_root'] = str(DEST)
        policy['probe_output'] = f'ISOLATION_{arm}_{seed}_{tag}.json'
        policy['read_paths'] += list(read(DEST/'PLAN.json')['frozen_source_and_inputs'])
        policy['read_paths'] += [str(DEST/'PLAN.json'),str(JOB/'narrow_structure_v1/TARGETS_READY.json'),
            str(JOB/'narrow_structure_v1/TRAIN_FREEZE.json')]
        policy['read_paths'] += list(read(JOB/'narrow_structure_v1/TRAIN_FREEZE.json')['files'])
        path = DEST/'policies'/f'{arm}_{seed}_{tag}.json'
        save(path,policy)
        args=[str(path),str(JOB/'dice_generated_state_v1.py'),'--arm',arm,'--seed',str(seed),'--until',str(until)]
        if resume:
            args += ['--resume',str(resume)]
        if probe:
            args += ['--probe-only']
        bounded(gpu,arm,seed,tag,'sandbox.py',args,seconds=1200 if probe else 3600 if until==20 else 9000)
    def evaluate(gpu, arm, seed, update):
        bounded(gpu,arm,seed,f'eval{update}','dice_generated_state_eval_v1.py',
            ['--arm',arm,'--seed',str(seed),'--update',str(update)],seconds=2400)
    def stage(seed, update, name):
        p=DEST/'milestones'/f'{seed}_{update}_{name}.json'
        save(p,{'seed':seed,'update':update,'stage':name,'epoch':time.time(),'max_update':100})
        print('MILESTONE',seed,update,name,flush=True)
    with (DEST/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (DEST/'COMPLETE.json').exists()
        for arm in ARMS:
            for seed in SEEDS:
                assert not (DEST/arm/f'seed{seed}/TRAIN_CURVE.jsonl').exists()
        frozen()
        g0,g1=previous.GPUS
        parallel((train,(g0,'baseline',1701,20),{'probe':True}),
                 (train,(g1,'generated',1701,20),{'probe':True}))
        stage(1701,0,'PREFLIGHT_PASSED')
        for seed in SEEDS:
            for until,resume in ((20,None),(100,20)):
                parallel((train,(g0,'baseline',seed,until,resume),{}),
                         (train,(g1,'generated',seed,until,resume),{}))
                stage(seed,until,'TRAINING_COMPLETE')
                parallel((evaluate,(g0,'baseline',seed,until),{}),
                         (evaluate,(g1,'generated',seed,until),{}))
                stage(seed,until,'EVALUATION_COMPLETE')
        frozen()
        subprocess.run([sys.executable,'-B',str(JOB/'dice_generated_state_score_v1.py')],
            cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        print('GENERATED_STATE_MATCHED_100_COMPLETE',flush=True)


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--detach' in sys.argv:
        assert (DEST/'PLAN.json').exists() and not (DEST/'OWNER.json').exists()
        with (DEST/'owner.log').open('x') as handle:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=handle,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(DEST/'OWNER.json',{'pid':child.pid,'physical_gpus':[0,1],
            'finite_not_automation':True,'hard_stop100':True})
        print('GENERATED_STATE_OWNER',child.pid,flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(DEST/'FAILURE.json',{'type':type(error).__name__,'error':str(error),
                'preserve_partials':True,'no_automatic_optimizer_retry':True})
            raise

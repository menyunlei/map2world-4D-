"""Matched fixed TRAIN readout, all seeds and families, original thresholds."""
from paths import *
from collections import Counter
from statistics import mean
from verifier_core import THRESHOLDS

DEST = JOB/'dice_generated_state_v1'
SOURCE = JOB/'dice_schedule_probe_v1'


def opening(m):
    return m['free_air_wrong_fraction']==0 and m['overhead_column_fraction']==0 and m['required_surface_recall']>=THRESHOLDS['required_surface_recall']


def summarize(packet):
    rows={r['scene']:r for r in packet['records']}
    records=[]
    for family in (7,9,13,15):
        prefix=f'courtyard_new_{family:03d}_'
        a,b=rows[prefix+'narrow1p5m'],rows[prefix+'closed']
        kinds={}
        for kind,part in [('free',None),('mid',('13','conditional')),('low',('25','conditional'))]:
            am,bm=(a['free'],b['free']) if part is None else (a['teacher'][part[0]][part[1]],b['teacher'][part[0]][part[1]])
            kinds[kind]={'open':am,'closed':bm,'joint_pass':opening(am) and bm['closed_closure_recall']>=THRESHOLDS['closed_closure_recall']}
        records.append({'family':family,'metrics':kinds})
    return {'per_family':records,'joint_counts':{kind:sum(r['metrics'][kind]['joint_pass'] for r in records) for kind in ('free','mid','low')}}


def main():
    plan=read(DEST/'PLAN.json')
    results=[]
    counts={}
    audits=[]
    for seed in (1701,1702):
        base=summarize(read(SOURCE/'eval/corrected'/f'seed{seed}/plus100/RESULTS.json'))
        results.append({'arm':'shared_start','seed':seed,'update':0,**base})
        counts[str(seed)]={'start':base['joint_counts']}
        for arm in ('baseline','generated'):
            for update in (20,100):
                item=summarize(read(DEST/'eval'/arm/f'seed{seed}'/f'plus{update:03d}/RESULTS.json'))
                results.append({'arm':arm,'seed':seed,'update':update,**item})
                counts[str(seed)][f'{arm}_{update}']=item['joint_counts']
            rows=[json.loads(line) for line in (DEST/arm/f'seed{seed}/TRAIN_CURVE.jsonl').read_text().splitlines()]
            assert [r['update'] for r in rows]==list(range(1,101))
            assert sum(len(r['dice_audit']) for r in rows)==400
            assert sum(len(r['flow_audit']) for r in rows)==1600
            rollout=[q for r in rows for q in r['rollout_audit']]
            assert len(rollout)==400
            eligible=[q for q in rollout if q['eligible']]
            assert len(eligible)==plan['generated_state']['eligible_count']
            audits.append({'arm':arm,'seed':seed,'rollout_NFE':sum(q['rollout_NFE'] for q in rollout),
                'generated_Dice_count':sum(q['input_kind']=='generated' for q in rollout),
                'eligible_count':len(eligible),'eligible_per_scene':dict(Counter(q['scene_id'] for q in eligible)),
                'flow_grad_mean':mean(r['flow_grad_norm'] for r in rows),
                'Dice_grad_active_mean':mean(r['dice_grad_norm'] for r in rows if r['update']%4==0)})
    matched=[]
    for seed in (1701,1702):
        a=[r for r in audits if r['seed']==seed]
        assert a[0]['rollout_NFE']==a[1]['rollout_NFE'] and a[0]['eligible_per_scene']==a[1]['eligible_per_scene']
        curves={arm:[json.loads(line) for line in (DEST/arm/f'seed{seed}/TRAIN_CURVE.jsonl').read_text().splitlines()] for arm in ('baseline','generated')}
        assert all(a['flow_audit']==b['flow_audit'] and a['dice_audit']==b['dice_audit'] for a,b in zip(curves['baseline'],curves['generated']))
        left=next(r for r in results if r['seed']==seed and r['arm']=='baseline' and r['update']==100)
        right=next(r for r in results if r['seed']==seed and r['arm']=='generated' and r['update']==100)
        matched.append({'seed':seed,'free_joint_gain':right['joint_counts']['free']-left['joint_counts']['free'],
            'mid_joint_gain':right['joint_counts']['mid']-left['joint_counts']['mid'],
            'low_joint_gain':right['joint_counts']['low']-left['joint_counts']['low']})
    costs=[read(p) for p in (DEST/'gpu_cost').glob('*.json')]
    assert costs and all(c['returncode']==0 for c in costs)
    replicated=all(q['free_joint_gain']>0 and q['low_joint_gain']>=0 for q in matched)
    result={'status':'GENERATED_STATE_MATCHED_100_COMPLETE','counts':counts,'results':results,
        'matched_100':matched,'training_audits':audits,
        'decision':'GENERATED_STATE_DICE_REPLICATED_TRAIN_GAIN' if replicated else 'GENERATED_STATE_DICE_NOT_REPLICATED_FREE_GENERATION_FIX',
        'gpu_seconds_including_load_idle':sum(c['gpu_seconds'] for c in costs),'gpu_cost_receipts':costs,
        'optimizer_updates_per_arm_seed':100,'no_300_extension':True,'HUMAN_REVIEW_PENDING':True,
        'scope':'seen-TRAIN learning capability only; teacher recovery and full generation reported separately',
        'structural_preservation_scope':'closed and enclosing walls in this subset; 3m/legacy/Gaussian preservation requires separate unchanged-chain evaluation'}
    save(DEST/'RESULTS.json',result)
    save(DEST/'COMPLETE.json',{k:result[k] for k in ('status','decision','counts','gpu_seconds_including_load_idle')})
    print(json.dumps(read(DEST/'COMPLETE.json')),flush=True)


if __name__=='__main__':
    main()

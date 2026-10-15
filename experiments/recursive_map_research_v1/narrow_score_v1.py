"""Reuse frozen native-Q metrics; compare both fixed seeds and actual visibility."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from paths import *
import argparse,collections,numpy as np,torch
import score_outputs as old
from score_outputs import verifier
NEW=JOB/'narrow_structure_v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--update',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);inputs=old.Inputs()
    manifest=inputs.json(JOB/'verifier_private/LABEL_MANIFEST.json',frozen=True)
    contract=inputs.json(JOB/'CONTRACT.json',frozen=True)
    legacy,_=old.load_legacy(inputs,contract)
    packet=inputs.json(NEW/'baseline/packets'/f'g2_b_{a.update}.json')
    labels=old.load_labels(inputs,manifest,{r['scene_id'] for r in packet['rows'] if r['kind']!='legacy'})
    records=[];matched={};all_costs=[]
    for seed in [1701,1702]:
        for arm in ['baseline','narrow']:
            other=inputs.json(NEW/arm/'packets'/f'g2_b_{a.update}.json');assert other==packet
            rows=[];preds={};noises={}
            checkpoint=inputs.json(NEW/arm/'runs/g2_b'/f'seed{seed}/CHECKPOINT_{a.update:06d}.json')
            inputs.file(checkpoint['path'],checkpoint['sha256'])
            for r in packet['rows']:
                d=NEW/arm/'evaluation/g2_b'/f'seed{seed}/plus{a.update:03d}'/r['scene_id']/f"seed{r['seed_index']}"
                receipt=inputs.json(d/'COMPLETE.json');assert receipt['packet']==r
                pred,raw=old.load_raw(inputs,d/'RAW_STRUCTURE.pt',receipt['files']['RAW_STRUCTURE.pt'],generation=True)
                assert raw['checkpoint']==checkpoint and raw['no_target_input']
                key=(seed,r['scene_id'],r['seed_index'])
                if arm=='baseline':matched[key]=raw['structure_noise']
                else:assert torch.equal(matched[key],raw['structure_noise'])
                value=old.legacy_score(pred,legacy[r['scene_id']]) if r['kind']=='legacy' else verifier.score(pred,labels[r['scene_id']]['target'],labels[r['scene_id']],r['variant'])
                rows.append({'scene':r['scene_id'],'family':r['family'],'variant':r['variant'],'kind':r['kind'],'generation_seed':r['seed_index'],**value})
                preds[r['scene_id'],r['seed_index']]=pred;noises[r['scene_id'],r['seed_index']]=raw['structure_noise']
            edits=[]
            for r in packet['rows']:
                if r['kind']=='legacy' or r['variant']=='closed':continue
                sid=r['scene_id'];cid=r['family']+'_closed';si=r['seed_index']
                if (cid,si) not in preds:continue
                cl,op=labels[cid],labels[sid]
                edits.append({'scene':sid,'generation_seed':si,**verifier.paired(preds[cid,si],preds[sid,si],cl['target'],op['target'],op['shared_closed_walls'],cl['protected']&op['protected'],op['free_air']|op['overhead'],torch.equal(noises[cid,si],noises[sid,si]))})
            summary={}
            for label,selected in [('narrow',[r for r in rows if r['kind']=='challenge']),('open3m',[r for r in rows if r['variant']=='open3m']),('open4m',[r for r in rows if r['variant']=='open4m']),('closed',[r for r in rows if r['variant']=='closed']),('legacy',[r for r in rows if r['kind']=='legacy'])]:
                keys=['non_ground_native_IoU'] if label=='legacy' else ['closed_closure_recall','required_surface_recall'] if label=='closed' else ['free_air_wrong_fraction','overhead_column_fraction','required_surface_recall','non_ground_native_IoU']
                summary[label]={k:float(np.mean([r[k] for r in selected])) if selected else None for k in keys}
            trans=inputs.json(NEW/arm/'transmittance'/f'seed{seed}/plus{a.update:03d}.json')
            records.append({'seed':seed,'arm':arm,'summary':summary,'per_family_rows':rows,'paired_edits':edits,'actual_Gaussian_transmittance':trans})
    comparison=[]
    for seed in [1701,1702]:
        b=next(r for r in records if r['seed']==seed and r['arm']=='baseline');n=next(r for r in records if r['seed']==seed and r['arm']=='narrow')
        bs,ns=b['summary'],n['summary']
        gain=all(ns['narrow'][k]<bs['narrow'][k] for k in ['free_air_wrong_fraction','overhead_column_fraction'])
        preserve=all(ns['open3m'][k]<=bs['open3m'][k]+.02 for k in ['free_air_wrong_fraction','overhead_column_fraction'])
        checks=[('closed','closed_closure_recall',verifier.THRESHOLDS['closed_closure_recall']),('closed','required_surface_recall',verifier.THRESHOLDS['required_surface_recall']),('open3m','required_surface_recall',verifier.THRESHOLDS['required_surface_recall']),('legacy','non_ground_native_IoU',verifier.THRESHOLDS['legacy_non_ground_IoU'])]
        margins={f'{s}.{k}':{'matched_delta':ns[s][k]-bs[s][k],'frozen_margin':ns[s][k]-threshold} for s,k,threshold in checks}
        preserve=preserve and all(v['matched_delta']>=-.02 and v['frozen_margin']>=0 for v in margins.values())
        bt={r['scene']:r for r in b['actual_Gaussian_transmittance']['rows']};nt={r['scene']:r for r in n['actual_Gaussian_transmittance']['rows']}
        visibility=[]
        for sid in bt:
            assert len(bt[sid]['rays'])==len(nt[sid]['rays']) and [r['xy_m'] for r in bt[sid]['rays']]==[r['xy_m'] for r in nt[sid]['rays']]
            visibility.append({'scene':sid,'variant':bt[sid]['variant'],'baseline_opacity':bt[sid]['mean_opacity'],'candidate_opacity':nt[sid]['mean_opacity']})
        nr=[v for v in visibility if v['variant']=='challenge1p5m']
        native_gain=bool(nr) and np.mean([v['candidate_opacity'] for v in nr])<np.mean([v['baseline_opacity'] for v in nr])
        comparison.append({'seed':seed,'narrow_Q_gain':bool(gain),'structural_preservation':bool(preserve),'preservation_margins':margins,'native_narrow_opacity_improved':bool(native_gain),'native_visibility':visibility})
    for pth in (NEW/'gpu_cost').glob('*.json'):all_costs.append(inputs.json(pth))
    result={'update':a.update,'records':records,'two_seed_comparison':comparison,
       'decision':'NARROW_Q_AND_NATIVE_SIGNAL_WITH_PRESERVATION' if all(r['narrow_Q_gain'] and r['structural_preservation'] and r['native_narrow_opacity_improved'] for r in comparison) else 'NO_JOINT_TWO_SEED_IMPROVEMENT_YET',
       'not_full_world_success':True,'width_extrapolation':False,'HUMAN_REVIEW_PENDING':True,'human_not_blocking':True,
       'gpu_cost':all_costs,'gpu_seconds':sum(r['gpu_seconds'] for r in all_costs),'input_sha256':inputs.hashes}
    inputs.recheck();old.finite_tree(result);save(NEW/f'RESULTS_{a.update:03d}.json',result)
    print(json.dumps({'update':a.update,'decision':result['decision'],'comparison':comparison}),flush=True)


if __name__=='__main__':main()

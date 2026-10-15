"""Read saved v12 logs/checkpoints/supports only. CPU, no model forward."""
from paths import JOB,read,save,sha
from trellis2_coverage_v12_prepare import OUT as V12,ARMS,SEEDS
from trellis2_coverage_v12_eval import folder,case,GROUPS
import torch,numpy as np,json,collections
OUT=V12/'diagnosis_v1'

def avg(x):return float(np.mean(x)) if len(x) else None
def squared(x):return float(x.double().square().sum())
def drift(now,old):
    return {'relative_L2':float((sum(squared(now[k]-old[k]) for k in now)/max(1e-20,sum(squared(old[k]) for k in old)))**.5),
      'norm':float(sum(squared(v) for v in now.values())**.5)}
def effective(state):return {k[:-2]:state[k[:-2]+'.B'].float()@v.float() for k,v in state.items() if k.endswith('.A')}
def support_stats(q):
    coords=np.argwhere(q)
    if len(coords)==0:return {'cells':0}
    z=coords[:,2];hist=q.sum((0,1));x=np.sort(hist[hist>0])
    return {'cells':int(q.sum()),'z_min_max':[int(z.min()),int(z.max())],
      'z_quantiles':np.quantile(z,[.05,.25,.5,.75,.95]).tolist(),
      'central90_z_span':float(np.quantile(z,.95)-np.quantile(z,.05)),
      'per_layer_support_median':float(np.median(x)),'histogram_z':hist.tolist()}

def main():
    assert not (OUT/'CPU_EVIDENCE.json').exists();torch.set_num_threads(4)
    plan=read(V12/'PLAN.json');rows=[];logs={};weights=[];native=[];sources={}
    for seed in SEEDS:
        p=plan['parent_checkpoints'][str(seed)];assert sha(p['path'])==p['sha256'];sources[p['path']]=p['sha256']
        parent=torch.load(p['path'],map_location='cpu',weights_only=False)
        pe=effective(parent['lora_state'])
        for arm in ARMS:
            d=V12/'small'/arm/f'seed{seed}/train'
            log=[json.loads(x) for x in (d/'train_log.jsonl').read_text().splitlines()];logs[seed,arm]=log
            sources[str(d/'train_log.jsonl')]=sha(d/'train_log.jsonl')
            for u in [100,300,1000]:
                counts=collections.Counter();cond=collections.Counter();mid=collections.Counter();noise=collections.defaultdict(list)
                for r in log[:u]:
                    for s in r['slots']:
                        counts[s['object']]+=1;cond[s['object']]+=not s['drop']
                        mid[s['object']]+=not s['drop'] and .3<=s['t']<=.8;noise[s['object']].append(s['t'])
                window=log[max(0,u-100):u]
                rows.append({'seed':seed,'arm':arm,'update':u,'counts':dict(counts),'conditioned':dict(cond),
                  'midnoise_conditioned':dict(mid),'per_object_mean_t':{k:avg(v) for k,v in noise.items()},
                  'last100_flow_MSE':avg([x['flow_MSE'] for x in window]),
                  'last100_head_grad':avg([x['head_grad_norm'] for x in window]),
                  'last100_lora_grad':avg([x['lora_grad_norm'] for x in window]),
                  'clipped_updates':sum(r['preclip_grad_norm']>1 for r in log[:u])})
                ck=read(d/f'CHECKPOINT_{u:06d}.json');assert sha(ck['path'])==ck['sha256'];sources[ck['path']]=ck['sha256']
                state=torch.load(ck['path'],map_location='cpu',weights_only=False)
                weights.append({'seed':seed,'arm':arm,'update':u,'adapter':drift(state['branch_state'],parent['branch_state']),
                  'LoRA_AB':drift(state['lora_state'],parent['lora_state']),
                  'LoRA_effective_BA':drift(effective(state['lora_state']),pe),
                  'optimizer_step_range':[min(float(v['step']) for v in state['optimizer']['state'].values()),
                    max(float(v['step']) for v in state['optimizer']['state'].values())]})
        a=logs[seed,ARMS[0]];b=logs[seed,ARMS[1]]
        assert len(a)==len(b)==1000
        assert all(all(x[k]==y[k] for k in ['noise_key','t','drop']) for x,y in zip(
          [s for r in a for s in r['slots']],[s for r in b for s in r['slots']]))
        for arm,u in GROUPS:
            for index in [0,1,6,7]:
                for gen in plan['generation_seeds']:
                    p=case(folder(seed,arm,u),gen,'correct',index)/'SS_GENERATED.npz'
                    with np.load(p) as z:q=z['native_support64'].copy()
                    native.append({'seed':seed,'arm':arm,'update':u,'index':index,'noise':gen,**support_stats(q),'source':str(p),'sha256':sha(p)})
    targets=[]
    for i,r in enumerate(plan['selected_records']):
        with np.load(r['native_target']) as z:
            targets.append({'index':i,'object':r['scene_id']+':'+str(r['object_id']),
              'source_asset':r['source_asset_id'],'S0_RMS':float(np.sqrt(np.mean(z['S0']**2))),**support_stats(z['support64'][0,0].astype(bool))})
    save(OUT/'CPU_EVIDENCE.json',{'exposures_and_gradients':rows,'weight_drift':weights,'saved_native_Q':native,
      'TRAIN_target_diagnostics':targets,'hashes':sources,'matched_slot_noise_time_dropout_exact':True,
      'exact_exposure_matched_checkpoint_unavailable':True,'near_comparison':'24-object +300 (25 slots/object) vs6-object +100 (33/34); 24-object +1000 (83/84) vs6-object +300 (100). Not exact causal matching.',
      'weight_norms_not_causal_proof':True,'no_training_or_model_forward':True})
    for x in weights:print('DRIFT',x['seed'],x['arm'],x['update'],x['adapter']['relative_L2'],x['LoRA_effective_BA']['relative_L2'],flush=True)
    for x in rows:print('LOG',x['seed'],x['arm'],x['update'],'exposure',min(x['counts'].values()),max(x['counts'].values()),'flow',x['last100_flow_MSE'],'clipped',x['clipped_updates'],flush=True)
    print('CPU_DIAGNOSIS_COMPLETE',flush=True)

if __name__=='__main__':main()

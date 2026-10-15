"""One evidence-gated correction of Structure map guidance; frozen model weights."""
from paths import JOB,read,save,sha
from trellis2_mapguided_v18 import OUT as OLD,SEED,REPO
from pathlib import Path
import argparse,numpy as np,sys
sys.path.insert(0,str(REPO))
OUT=JOB/'map_object_world_v1/trellis2_map_volume_v19'

def prepare():
    assert not OUT.exists()
    p=read(OLD/'PLAN.json')
    save(OUT/'DIAGNOSIS_PLAN.json',{'source':str(OLD),'source_plan_sha256':sha(OLD/'PLAN.json'),
        'scope':'Saved latent replay and gradient support, no sampling or training',
        'seed':SEED,'GPU_cap_seconds':7200,'candidate_limit':1,
        'controls':'If justified, reuse v18 matched original/edit; same weights/image/noise/sampling and inner-step budget',
        'hypothesis':'Heightwise amax directly penalizes only the maximum-logit voxel, leaving other occupied heights unsupervised',
        'alternatives_checked':'Final Euler coefficient and saved support roundtrip; not a broad audit',
        'no_automatic_training_or_search':True})
    save(OUT/'DIAGNOSIS_SOURCE.json',{str(JOB/'trellis2_map_volume_v19.py'):sha(JOB/'trellis2_map_volume_v19.py'),
        str(JOB/'trellis2_map_volume_v19_owner.py'):sha(JOB/'trellis2_map_volume_v19_owner.py')})

def diagnose():
    import torch
    from trellis2 import models
    torch.set_num_threads(4);p=read(OLD/'PLAN.json')
    cfg=read(Path(p['runtime'])/'pipeline.json')['args']
    decoder=models.from_pretrained(cfg['models']['sparse_structure_decoder']).eval().requires_grad_(False).cuda()
    rows=[]
    for mode in ['baseline','correct','edit']:
        mapmode='correct' if mode=='baseline' else mode
        with np.load(p['maps'][mapmode]['path']) as z:mask=torch.from_numpy(z['footprints'].any(0).T.copy()).bool().cuda()
        with np.load(OLD/mode/'SS.npz') as z:
            y=torch.from_numpy(z['latent'].copy()).cuda();saved=z['support64'].copy()
        with torch.no_grad():full=decoder(y).float()[0,0]
        mismatch=float(((full>0).cpu().numpy()!=saved).mean())
        assert mismatch<1e-5,('ROUNDTRIP_MISMATCH',mode,mismatch)
        logits=full[:,:,p['guidance']['building_z_start']:].detach().requires_grad_(True)
        outside=~mask;maximum=logits.amax(-1)
        oldloss=torch.nn.functional.softplus(maximum[outside]).mean()
        oldgrad=torch.autograd.grad(oldloss,logits)[0]
        occupied=(logits>0)&outside[...,None]
        directly_hit=occupied&(oldgrad!=0)
        counts=occupied.sum(-1)[outside];nonempty=counts[counts>0]
        # Analytic distributed derivative, matched to the old per-column L1 budget.
        prob=logits.detach().sigmoid();scale=maximum.detach().sigmoid()/prob.sum(-1).clamp_min(1e-12)
        newgrad=prob*scale[...,None]*outside[...,None]/outside.sum()
        rows.append({'mode':mode,'saved_support_reproduced':mismatch==0,
            'outside_occupied_voxels':int(occupied.sum()),'directly_penalized_occupied_voxels':int(directly_hit.sum()),
            'old_direct_gradient_coverage':float(directly_hit.sum()/occupied.sum()),
            'occupied_heights_per_nonempty_column_p10_p50_p90':torch.quantile(nonempty.float(),torch.tensor([.1,.5,.9],device='cuda')).tolist(),
            'outside_positive_logits_p10_p50_p90':torch.quantile(logits.detach()[occupied],torch.tensor([.1,.5,.9],device='cuda')).tolist(),
            'matched_distribution_direct_coverage':float(((newgrad>0)&occupied).sum()/occupied.sum()),
            'per_column_gradient_L1_match_max_error':float((oldgrad.abs().sum(-1)-newgrad.abs().sum(-1)).abs().max()),
            'old_negative_loss':float(oldloss.detach())})
    pars=cfg['sparse_structure_sampler']['params'];sigma=cfg['sparse_structure_sampler']['args']['sigma_min']
    ts=np.linspace(1,0,pars['steps']+1);ts=pars['rescale_t']*ts/(1+(pars['rescale_t']-1)*ts)
    gain=[{'t':float(t),'t_next':float(n),'clean_correction_gain':float((t-n)/(sigma+(1-sigma)*t))} for t,n in zip(ts[:-1],ts[1:])]
    result={'rows':rows,'Euler_clean_correction_coefficients':gain,
        'final_clean_correction_gain':gain[-1]['clean_correction_gain'],
        'sample_loop':'ret.samples is propagated pred_x_prev; correction is not dropped',
        'downstream':'v18 generation replaces native sample_sparse_structure with saved coords32; no Structure resampling',
        'supported_issue':'Sparse heightwise gradient allocation' if all(r['old_direct_gradient_coverage']<.25 for r in rows) else 'Not supported',
        'candidate_condition':'Only change free-space gradient allocation; match per-column L1 to avoid strength increase confound',
        'not_yet_proven':'Whether this one correction can change the global layout while preserving native appearance',
        'no_model_updates':True}
    save(OUT/'DIAGNOSIS.json',result);print('SAVED_WEIGHT_DIAGNOSIS',result,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--diagnose',action='store_true');v=a.parse_args()
    prepare() if v.prepare else diagnose()

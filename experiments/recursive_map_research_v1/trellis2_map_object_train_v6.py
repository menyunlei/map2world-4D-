"""Existing adapter only; matched object-native SS/Shape flow objectives."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_data_v6_fix1 import OUT, ARMS, SEEDS
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_map_object_module_v1 import native_world_grid
from paths import read, save, sha, JOB
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
import torch,numpy as np,argparse,time,random,json,hashlib

def setup(arm,seed):
    plan=read(OUT/'PLAN.json');records=read(OUT/'PAIRED_NATIVE_MANIFEST.json')['records']
    native_ss=base.native_model()
    shape=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16')).cuda().eval().requires_grad_(False)
    torch.manual_seed(seed)
    branch=ObjectContractConditioner(reading_mode=arm).cuda()
    parent=plan['parent_checkpoints'][str(seed)]
    assert sha(parent['path'])==parent['sha256']
    branch.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'],strict=True)
    reader=base.BoundMapReader(native_ss,branch)
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    mean=torch.tensor(cfg['shape_slat_normalization']['mean'],device='cuda')[None]
    std=torch.tensor(cfg['shape_slat_normalization']['std'],device='cuda')[None]
    return plan,records,branch,reader,native_ss,shape,mean,std

def stage(record,phase,noise_key,components,training=True):
    plan,records,branch,reader,ss,shape,mean,std=components
    m=base.load_map(record)
    context=branch.prepare(m,native_world_grid(device='cuda'))
    generator=torch.Generator(device='cuda').manual_seed(noise_key)
    t=torch.sigmoid(torch.randn((1,),device='cuda',generator=generator)+1) if training else torch.tensor([.6],device='cuda')
    dropped=training and float(torch.rand((),device='cuda',generator=generator))<.1
    cond=branch(context,arm='separated',negative_cfg=dropped)
    assert sha(record['native_target'])==record['native_target_sha256']
    with np.load(record['native_target']) as z:
        if phase=='SS':x0=torch.from_numpy(z['S0'].copy()).float().cuda()
        else:x0=SparseTensor(feats=(torch.from_numpy(z['feats'].copy()).float().cuda()-mean)/std,
                             coords=torch.from_numpy(z['coords'].copy()).int().cuda())
    if phase=='SS':
        noise=torch.randn(x0.shape,device='cuda',generator=generator)
        tt=t[:,None,None,None,None];x=(1-tt)*x0+(1e-5+(1-1e-5)*tt)*noise
        reader.context=cond.context
        output=ss(x,t*1000,cond.features).float();target=(1-1e-5)*noise-x0
    else:
        noise=torch.randn(x0.feats.shape,device='cuda',generator=generator)
        x=x0.replace((1-t)*x0.feats+(1e-5+(1-1e-5)*t)*noise)
        output=shape(x,t*1000,cond.features).feats.float();target=(1-1e-5)*noise-x0.feats
    loss=(output-target).square().mean();assert torch.isfinite(loss)
    return loss,dropped

def heldout(components):
    seen=set();units=[]
    for r in components[1]:
        if r['split']=='adapter_HELDOUT' and r['family_id'] not in seen:
            units.append(r);seen.add(r['family_id'])
    assert len(units)==8
    results=[];components[2].eval()
    with torch.no_grad():
        for i,r in enumerate(units):
            row={'family_id':r['family_id'],'scene_id':r['scene_id'],'object_id':r['object_id']}
            for j,p in enumerate(['SS','shape']):row[p+'_flow_MSE']=float(stage(r,p,998600+i*2+j,components,False)[0])
            results.append(row)
    components[2].train();components[3].context=None
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=setup(a.arm,a.seed);plan,records,branch,reader,ss,shape,mean,std=c
    train=[r for r in records if r['split']=='adapter_TRAIN']
    order=list(range(len(train)));random.Random(993600).shuffle(order)
    opt=torch.optim.AdamW(branch.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.95),eps=1e-8)
    d=OUT/a.arm/f'seed{a.seed}';d.mkdir(parents=True,exist_ok=True)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in branch.state_dict().values())).hexdigest()
    source={'arm':a.arm,'seed':a.seed,'initial_branch_sha256':initial,'parent':plan['parent_checkpoints'][str(a.seed)],
        'trainable_parameters':sum(p.numel() for p in branch.parameters()),
        'native_frozen_parameters':sum(p.numel() for m in [ss,shape] for p in m.parameters()),
        'condition_geometry':['map_only','map_only'],'geometry_images_read':False,
        'per_object_native_targets':True,'inference_contract_identical':True,'map10_TRAIN':False,
        'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest()}
    assert source['trainable_parameters']==611584
    if not a.preflight:
        assert (OUT/'TRAIN_BUDGET_FREEZE.json').exists()
        save(d/'TRAIN_START.json',source)
        save(d/'HELDOUT_000000.json',{'units':heldout(c),'not_quality_success':True})
    rows=[];exposures={};start=time.time()
    updates=2 if a.preflight else 300
    with (d/('preflight_log.jsonl' if a.preflight else 'train_log.jsonl')).open('x') as f:
        for update in range(1,updates+1):
            tick=time.time();opt.zero_grad(set_to_none=True);losses={'SS':[],'shape':[]};grads={'SS':[],'shape':[]}
            for micro in range(2):
                r=train[order[((update-1)*2+micro)%len(order)]]
                for j,phase in enumerate(['SS','shape']):
                    before=branch.map_condition_head[2].weight.grad
                    before=None if before is None else before.detach().clone()
                    loss,drop=stage(r,phase,986600+(update-1)*4+micro*2+j,c,not a.preflight)
                    losses[phase].append(float(loss.detach()))
                    if not drop:
                        (loss/4).backward()
                        after=branch.map_condition_head[2].weight.grad
                        grads[phase].append(float((after if before is None else after-before).norm()))
                key=r['scene_id']+':'+str(r['object_id']);exposures[key]=exposures.get(key,0)+1
            norm=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1.))
            assert np.isfinite(norm) and all(p.grad is None for m in [ss,shape] for p in m.parameters())
            opt.step()
            row={'update':update,'sec_update':time.time()-tick,'preclip_grad_norm':norm,
                'SS_flow_MSE':float(np.mean(losses['SS'])),'shape_flow_MSE':float(np.mean(losses['shape'])),
                'SS_head_gradient':float(np.mean(grads['SS'])) if grads['SS'] else 0.,
                'shape_head_gradient':float(np.mean(grads['shape'])) if grads['shape'] else 0.,
                'allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
            f.write(json.dumps(row)+'\n');f.flush()
            if a.preflight:
                assert row['SS_head_gradient']>0 and row['shape_head_gradient']>0
                rows.append(row)
            if a.preflight or update%10==0:print('OBJECT_NATIVE_UPDATE',a.arm,a.seed,row,flush=True)
            if not a.preflight and update in [20,100,300]:
                path=d/f'branch_{update:06d}.pt'
                torch.save({'branch_state':branch.state_dict(),'optimizer':opt.state_dict(),'updates':update,
                    'arm':a.arm,'seed':a.seed,'plan_sha256':sha(OUT/'PLAN.json'),
                    'parent':source['parent'],'native_frozen':True,'geometry_images_read':False,
                    'per_object_contract':True,'map10_TRAIN':False,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},path)
                save(d/f'CHECKPOINT_{update:06d}.json',{'path':str(path),'sha256':sha(path),
                    'additional_updates':update,'exposures':exposures})
                save(d/f'HELDOUT_{update:06d}.json',{'units':heldout(c),'not_quality_success':True})
    reader.context=None
    save(d/('PREFLIGHT.json' if a.preflight else 'TRAIN_COMPLETE.json'),{**source,
         'updates':updates,'discarded_reset_before_primary':a.preflight,'rows':rows,
         'seconds':time.time()-start,'exposures':exposures,'no_automatic_extension':True})

if __name__=='__main__':main()

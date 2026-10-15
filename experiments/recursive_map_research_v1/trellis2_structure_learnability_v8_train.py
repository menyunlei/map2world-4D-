"""Matched SS-only native-flow updates; targets never enter normal sampling."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_contract_v6 import ObjectContractConditioner
from trellis2_object_confidence_fix_v7 import correct_object_confidence
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_structure_learnability_v8_prepare import OUT,OLD,ARMS,SEEDS
import trellis2_structure_lora_v8 as lora
from paths import read,save,sha
import torch,numpy as np,argparse,time,random,json,hashlib

def is_lora_name(n):return '.cross_attn.' in n and n.endswith(('.A','.B'))
def lora_parameters(native):return [p for n,p in native.named_parameters() if is_lora_name(n)]
def lora_state_dict(native):return lora.lora_state_dict(native) if lora_parameters(native) else {}
def load_lora_state_dict(native,state):
    if lora_parameters(native):lora.load_lora_state_dict(native,state)
    else:assert not state

def setup(arm,seed):
    plan=read(OUT/'PLAN.json'); native=base.native_model()
    torch.manual_seed(seed)
    branch=ObjectContractConditioner(reading_mode='object_grounded').cuda()
    parent=plan['parent_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    branch.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'],strict=True)
    reader=base.BoundMapReader(native,branch)
    if arm=='structure_lora':lora.attach_lora(native)
    params=list(branch.parameters())+list(lora_parameters(native))
    assert sum(p.numel() for p in branch.parameters())==611584
    assert all(not p.requires_grad for n,p in native.named_parameters() if not is_lora_name(n))
    return plan,native,branch,reader,params

def load_record(r):
    assert sha(r['map_input'])==r['map_sha256']
    with np.load(r['map_input']) as z:m={k:z[k].copy() for k in z.files}
    m=correct_object_confidence(m)
    return {k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in m.items()}

def step(r,noise_key,c,probe=False):
    plan,native,branch,reader,params=c
    m=load_record(r);context=branch.prepare(m,native_world_grid(device='cuda'))
    gen=torch.Generator(device='cuda').manual_seed(noise_key)
    t=torch.sigmoid(torch.randn((1,),device='cuda',generator=gen)+1)
    dropped=float(torch.rand((),device='cuda',generator=gen))<.1
    if probe:t=torch.tensor([.6],device='cuda');dropped=False
    cond=branch(context,arm='separated',negative_cfg=dropped)
    assert sha(r['native_target'])==r['native_target_sha256']
    with np.load(r['native_target']) as z:x0=torch.from_numpy(z['S0'].copy()).float().cuda()
    noise=torch.randn(x0.shape,device='cuda',generator=gen)
    tt=t[:,None,None,None,None]; x=(1-tt)*x0+(1e-5+(1-1e-5)*tt)*noise
    reader.context=cond.context
    output=native(x,t*1000,cond.features).float();target=(1-1e-5)*noise-x0
    loss=(output-target).square().mean();assert torch.isfinite(loss)
    return loss,{'t':float(t),'drop':dropped,'noise_key':noise_key,'object':r['scene_id']+':'+str(r['object_id'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--phase',choices=['small','large'],default='small');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    c=setup(a.arm,a.seed);plan,native,branch,reader,params=c
    if a.phase=='large':
        assert read(OUT/'SMALL_DECISION.json')['launch_large_matched'] is True
        records=[r for r in read(OLD/'PAIRED_NATIVE_MANIFEST.json')['records'] if r['split']=='adapter_TRAIN']
    else:records=plan['selected_records']
    order=list(range(len(records)));random.Random(971800).shuffle(order)
    opt=torch.optim.AdamW(params,**{k:plan['optimizer'][k] for k in ['lr','weight_decay','eps']},betas=tuple(plan['optimizer']['betas']))
    d=OUT/a.phase/a.arm/f'seed{a.seed}'/('preflight' if a.preflight else 'train');d.mkdir(parents=True,exist_ok=False)
    source={'arm':a.arm,'seed':a.seed,'phase':a.phase,'parent':plan['parent_checkpoints'][str(a.seed)],
        'initial_branch_sha256':hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in branch.state_dict().values())).hexdigest(),
        'trainable_parameters':sum(p.numel() for p in params),'lora_parameters':sum(p.numel() for p in lora_parameters(native)),
        'original_native_weights_frozen':True,'order':order,'record_count':len(records),'geometry_images_read':False,
        'v7_confidence_contract_at_TRAIN_and_inference':True,'flow_only':True,'TF32_matmul':False,'TF32_cudnn':True}
    save(d/'TRAIN_START.json',source)
    updates=2 if a.preflight else plan['small' if a.phase=='small' else 'large_conditional']['updates']
    checkpoints=[] if a.preflight else plan['small' if a.phase=='small' else 'large_conditional']['checkpoints']
    tick0=time.time();exposure={};rows=[]
    with (d/'train_log.jsonl').open('x') as f:
        for update in range(1,updates+1):
            tick=time.time();opt.zero_grad(set_to_none=True);losses=[];slots=[]
            for micro in range(2):
                r=records[order[((update-1)*2+micro)%len(order)]]
                loss,slot=step(r,971000+(a.seed-953101)*100000+(update-1)*2+micro,c,a.preflight)
                losses.append(float(loss.detach()));slots.append(slot)
                # Frozen baseline has no gradients on unconditional/drop slots;
                # LoRA must still learn the SAME native objective on these slots.
                if loss.requires_grad:(loss/2).backward()
                exposure[slot['object']]=exposure.get(slot['object'],0)+1
            head=branch.map_condition_head[2].weight.grad
            lg=[p.grad for p in lora_parameters(native) if p.grad is not None]
            hnorm=0 if head is None else float(head.norm())
            lnorm=float(torch.sqrt(sum(g.float().square().sum() for g in lg))) if lg else 0.
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            assert np.isfinite(norm) and all(p.grad is None for n,p in native.named_parameters() if not is_lora_name(n))
            if a.preflight:assert hnorm>0 and (a.arm!='structure_lora' or lnorm>0)
            opt.step();row={'update':update,'sec_update':time.time()-tick,'flow_MSE':float(np.mean(losses)),
                'head_grad_norm':hnorm,'lora_grad_norm':lnorm,'preclip_grad_norm':norm,'slots':slots,
                'allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
            f.write(json.dumps(row)+'\n');f.flush()
            if a.preflight:rows.append(row)
            if a.preflight or update%20==0:print('SS_LEARN_UPDATE',a.phase,a.arm,a.seed,row,flush=True)
            if update in checkpoints:
                path=d/f'checkpoint_{update:06d}.pt'
                torch.save({'branch_state':branch.state_dict(),'lora_state':lora_state_dict(native),'optimizer':opt.state_dict(),
                    'updates':update,'arm':a.arm,'seed':a.seed,'phase':a.phase,'plan_sha256':sha(OUT/'PLAN.json'),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'original_weights_frozen':True},path)
                save(d/f'CHECKPOINT_{update:06d}.json',{'path':str(path),'sha256':sha(path),'updates':update,'exposure':exposure})
                values={f'branch::{k}':v.detach().float().cpu().numpy() for k,v in branch.state_dict().items()}
                values.update({f'lora::{k}':v.detach().float().cpu().numpy() for k,v in lora_state_dict(native).items()})
                with (d/f'checkpoint_{update:06d}.npz').open('xb') as handle:np.savez_compressed(handle,**values)
    reader.context=None
    save(d/('PREFLIGHT.json' if a.preflight else 'TRAIN_COMPLETE.json'),{**source,'actual_updates':updates,
        'actual_seconds':time.time()-tick0,'exposure':exposure,'rows':rows,'discarded_and_reset_before_training':a.preflight})

if __name__=='__main__':main()

"""One bounded adaptation of existing earliest-failing Structure parameters."""
from trellis2_stage_v16 import OUT,V15,V12,SEEDS
from paths import JOB,read,save,sha
import argparse,random,time,json,torch,numpy as np

def register():
    diagnosis=read(OUT/'DIAGNOSIS_COMPLETE.json');assert not diagnosis['errors']
    old=read(V12/'PLAN.json');p=read(V15/'PLAN.json')
    records=old['selected_records'];assert len(records)==24 and all(r['split']=='adapter_TRAIN' for r in records)
    assert not ({r['family_id'] for r in records}&{r['family_id'] for r in old['heldout_records']})
    frozen={str(JOB/n):sha(JOB/n) for n in ['trellis2_stage_v16_train.py','trellis2_stage_v16_repair_owner.py']}
    for r in records:
        for key,hkey in [('map_input','map_sha256'),('native_target','native_target_sha256')]:
            assert sha(r[key])==r[hkey];frozen[r[key]]=r[hkey]
    for c in p['SS_checkpoints'].values():assert sha(c['path'])==c['sha256'];frozen[c['path']]=c['sha256']
    order=list(range(24));random.Random(old['order_seed']).shuffle(order)
    save(OUT/'REPAIR_PLAN.json',{'intervention':'One Structure-only existing adapter+LoRA adaptation, no new mechanism',
        'reason':'Map10 nonbuilding supports, floating sheets and empty output already occur before Shape',
        'implementation_bug_demonstrated':False,'updates':100,'seeds':SEEDS,'records':records,
        'data':'Existing v12 fixed 24 legitimate TRAIN objects; no new targets, no Map10 training',
        'order':order,'parents':p['SS_checkpoints'],'optimizer':old['optimizer'],
        'effective_batch':2,'noise_base':99616000,'same_noise_distribution_as_existing_trainer':True,
        'loss':'unchanged native flow MSE','trainable':'existing Structure map adapter and existing Structure cross-attention LoRA only',
        'frozen':['native original Structure weights','Shape adapter','native Shape Flow','decoders'],
        'checks':[20,100],'primary':100,'no_checkpoint_selection_or_extension':True,
        'controls':'v15 original Map10 outputs and v12 six_object_rehearsal TRAIN/heldout pure-noise outputs',
        'no_map10_targets':True,'no_materials':True,'no_geometry_or_scale_changes':True})
    frozen[str(OUT/'REPAIR_PLAN.json')]=sha(OUT/'REPAIR_PLAN.json')
    save(OUT/'REPAIR_SOURCE_FREEZE.json',frozen)
    print('REPAIR_PREREGISTERED_STRUCTURE_ONLY_100',flush=True)

def train(seed):
    import trellis2_structure_learnability_v8_train as t
    p=read(OUT/'REPAIR_PLAN.json');c=t.setup('structure_lora',seed)
    plan,native,branch,reader,params=c;parent=p['parents'][str(seed)]
    state=torch.load(parent['path'],map_location='cpu',weights_only=False)
    branch.load_state_dict(state['branch_state'],strict=True);t.load_lora_state_dict(native,state['lora_state']);del state
    opt=torch.optim.AdamW(params,**{k:p['optimizer'][k] for k in ['lr','weight_decay','eps']},betas=tuple(p['optimizer']['betas']))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    d=OUT/'repair'/f'seed{seed}';d.mkdir(parents=True,exist_ok=False)
    save(d/'START.json',{'parent':parent,'parameters':sum(v.numel() for v in params),
        'frozen_backbone':True,'Shape_modified':False,'TRAIN_objects':24,'effective_batch':2,
        'optimizer_reset':True,'train_seed':seed,'training_data_hashes':sha(OUT/'REPAIR_SOURCE_FREEZE.json')})
    start=time.time();exposure={}
    with (d/'train_log.jsonl').open('x') as f:
        for update in range(1,101):
            opt.zero_grad(set_to_none=True);losses=[];tick=time.time();slots=[]
            for micro in range(2):
                r=p['records'][p['order'][((update-1)*2+micro)%24]]
                loss,slot=t.step(r,p['noise_base']+(seed-SEEDS[0])*100000+(update-1)*2+micro,c)
                if loss.requires_grad:(loss/2).backward()
                losses.append(float(loss.detach()));slots.append(slot)
                exposure[slot['object']]=exposure.get(slot['object'],0)+1
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert np.isfinite(norm)
            assert all(v.grad is None for n,v in native.named_parameters() if not t.is_lora_name(n))
            opt.step();row={'update':update,'loss':float(np.mean(losses)),'preclip_norm':norm,
                'seconds':time.time()-tick,'slots':slots,'max_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            f.write(json.dumps(row)+'\n');f.flush()
            if update%20==0:print('STRUCTURE_REPAIR_UPDATE',seed,update,row['loss'],flush=True)
            if update in [20,100]:
                path=d/f'checkpoint_{update:06d}.pt'
                torch.save({'branch_state':branch.state_dict(),'lora_state':t.lora_state_dict(native),
                    'optimizer':opt.state_dict(),'updates':update,'seed':seed,'parent':parent,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                    'Shape_frozen':True,'plan_sha256':sha(OUT/'REPAIR_PLAN.json')},path)
                save(d/f'CHECKPOINT_{update:06d}.json',{'path':str(path),'sha256':sha(path),'updates':update,'exposures':exposure})
    reader.context=None;save(d/'TRAIN_COMPLETE.json',{'updates':100,'seconds':time.time()-start,
        'exposures':exposure,'no_more_training':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args();register() if a.register else train(a.seed)

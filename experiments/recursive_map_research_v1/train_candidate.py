"""Cumulative20/100/300 branch-only training; finite, identical two-seed data."""
import argparse,json,os,time,traceback
from runtime import *

def record(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['g1_a','g1_b','g2_a'],required=True)
    ap.add_argument('--seed',type=int,choices=[1701,1702],required=True)
    ap.add_argument('--until',type=int,choices=[20,100,300],required=True);ap.add_argument('--resume',type=int,choices=[20,100])
    ap.add_argument('--probe-only',action='store_true')
    ap.add_argument('--attempt',default='original')
    a=ap.parse_args();stage='structure' if a.candidate in ['g1_a','g2_a'] else 'detail'
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.probe_only:
        assert torch.ones(16,device='cuda').sum().item()==16
        print('GPU_IMPORT_AND_CUDA_OK',torch.cuda.get_device_name(),flush=True)
        return
    out=JOB/'runs'/a.candidate/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    model,branch,parent,restore=initialize(a.candidate,a.seed)
    model.train();optimizer=torch.optim.AdamW(list(branch.values()),lr=1e-4,weight_decay=0,fused=True)
    ema={k:p.detach().clone() for k,p in branch.items()};start=0
    if a.resume:
        info=read(out/f'CHECKPOINT_{a.resume:06d}.json');assert sha(info['path'])==info['sha256']
        checkpoint=torch.load(info['path'],map_location='cpu',weights_only=False)
        assert checkpoint['candidate']==a.candidate and checkpoint['seed']==a.seed and checkpoint['update']==a.resume
        assert set(branch)==set(checkpoint['branch_state'])
        with torch.no_grad():
            for k,p in branch.items():p.copy_(checkpoint['branch_state'][k])
        optimizer.load_state_dict(checkpoint['optimizer']);ema={k:v.cuda() for k,v in checkpoint['EMA'].items()}
        restore=checkpoint['rng'];start=a.resume;del checkpoint
    else:assert not (out/'TRAIN_CURVE.jsonl').exists(),'duplicate training start'
    assert start<a.until
    data=TrainData(stage);decoder=structure_decoder() if stage=='structure' else None
    if stage=='detail':
        assert data.train and all(r['split']=='train' and r['detail'] is not None for r in data.train),'Detail requires legitimate original TRAIN SLAT only'
    runtime=auxiliary(data) if stage=='structure' else None
    slots=read(JOB/'TRAINING_SLOTS.json')['updates'] if runtime else None
    schedule={g['added_update']:g for g in read(C4/'AUXILIARY_SCHEDULE.json')['groups']} if runtime else None
    coefficient=read(C4/'AUXILIARY_CONTRACT.json')['coefficient'] if runtime else None
    restore_rng(restore)
    frozen={k:p for k,p in model.named_parameters() if not p.requires_grad}
    source={str(p):sha(p) for p in (JOB/'candidates'/a.candidate).glob('*.py')}
    record(out/f'RUN_{start:03d}_{a.until:03d}_{a.attempt}.json',{'candidate':a.candidate,'seed':a.seed,'parent':parent,'source':source,
        'trainables':{k:list(p.shape) for k,p in branch.items()},'new_parameters':sum(p.numel() for p in branch.values()),
        'batch16':True,'sample_count_budget':a.until*16,'cumulative_updates':True,'EMA_is_primary':False,
        'stage':stage,'no_postprocessing':True,'no_DEV_or_test_training':True})
    first=True;torch.cuda.reset_peak_memory_stats()
    for update in range(start+1,a.until+1):
        tick=time.time();optimizer.zero_grad(set_to_none=True);sequence=[];flows=[];dices=[];aux_losses=[]
        for index in range(16):
            if runtime:
                slot=slots[update-1]['slots'][index];sid=slot['P0_scene'];sample=slot['sample_id'];r=data.lookup[sid]
            else:
                sample=80000+(update-1)*16+index;r=data.train[((update-1)*16+index)%len(data.train)]
                assert r['split']=='train' and r['detail'] is not None
            _,noisy,target,t,text,m,surface,seq=data.pack([r],[sample]);sequence.extend(seq)
            topology=runtime.zero if runtime and seq[0][3] else runtime.topology(r['scene_id']) if runtime else None
            if runtime:assert seq[0][3]==slot['drop'] and abs(float(t[0])-slot['t'])<1e-6
            if first:
                # Same native model/noise/map, with new branch switched off.
                # This is a no-optimizer identity check, not a discarded update.
                before_preflight=rng_state()
                if not runtime and start==0:
                    from native_parent_identity import check_native_parent_identity
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        proof=check_native_parent_identity(model,noisy,t*1000,text,m)
                    record(out/f'NATIVE_PARENT_IDENTITY_{a.attempt}.json',proof)
                    assert proof['exact_identity'] and proof['rng_restored']
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    args=(noisy,t*1000,text,m,topology) if runtime else (noisy,t*1000,text,m)
                    initialized=model(*args)
                    old_read=model.branch.read
                    model.branch.read=lambda hidden,prepared,site:torch.zeros_like(hidden)
                    try:reference=model(*args)
                    finally:model.branch.read=old_read
                x=initialized if runtime else initialized.feats;y=reference if runtime else reference.feats
                if start==0:assert torch.equal(x,y),('ZERO_INIT_IDENTITY',float((x-y).abs().max()))
                record(out/f'PREFLIGHT_{start:03d}_{a.attempt}.json',{'initialized_exact_identity':bool(torch.equal(x,y)),
                       'identity_required':start==0,'resume':start,'max_abs_delta':float((x-y).abs().max()),
                       'TRAIN_scene':r['scene_id'],'sample_id':sample,'peak_allocated':torch.cuda.max_memory_allocated()})
                del initialized,reference,args,x,y;first=False
                restore_rng(before_preflight)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(noisy,t*1000,text,m,topology) if runtime else model(noisy,t*1000,text,m)
                flow=per_scene_flow(prediction,target,'structure' if runtime else 'slat')
                dice=decoded_dice(decoder,noisy,t,prediction,surface,True) if runtime and update%4==0 else torch.zeros_like(flow)
                loss=(flow+.1*dice).sum()/16
            assert torch.isfinite(loss);loss.backward();flows.append(float(flow.mean()));dices.append(float(dice.mean()))
            del noisy,target,t,text,m,surface,prediction,flow,dice,loss
        if runtime and update%4==0:
            preserve_rng=rng_state();group=schedule[update]
            for entry in group['entries']:
                batch,topology,masks=runtime.pack_auxiliary(entry);_,noisy,target,t,text,m,surface,seq=batch
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(noisy,t*1000,text,m,topology)
                    logits=decode(decoder,noisy,t,prediction)
                    _,positive,negatives=terms(logits,surface,masks,runtime.groups[(update//4)-1])
                    loss=coefficient*(positive+negatives['O1'])
                assert torch.isfinite(loss);loss.backward();aux_losses.append(float(loss))
                del batch,noisy,target,t,text,m,surface,prediction,logits,positive,negatives,loss,masks
            restore_rng(preserve_rng)
        assert all(p.grad is None for p in frozen.values()),'parent gradient'
        missing=[k for k,p in branch.items() if p.grad is None];assert not missing,missing
        norm=torch.nn.utils.clip_grad_norm_(list(branch.values()),1.);assert torch.isfinite(norm)
        optimizer.step()
        with torch.no_grad():
            for k,p in branch.items():ema[k].lerp_(p,1-.9999)
        torch.cuda.synchronize()
        row={'update':update,'candidate':a.candidate,'seed':a.seed,'flow':float(np.mean(flows)),'dice':float(np.mean(dices)),
             'auxiliary_loss':sum(aux_losses),'preclip_grad_norm':float(norm),'postclip_grad_norm_bound':min(float(norm),1.),
             'seconds':time.time()-tick,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'sequence':sequence,'optimizer_update_completed':True}
        with (out/'TRAIN_CURVE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('UPDATE',json.dumps({k:v for k,v in row.items() if k!='sequence'}),flush=True)
        if update in [20,100,300]:
            path=out/f'checkpoint_{update:06d}.pt';assert not path.exists()
            torch.save({'candidate':a.candidate,'seed':a.seed,'update':update,'stage':stage,'parent':parent,
                 'branch_state':{k:p.detach().cpu() for k,p in branch.items()},'optimizer':optimizer.state_dict(),
                 'EMA':{k:v.cpu() for k,v in ema.items()},'rng':rng_state(),'source':source},path)
            record(out/f'CHECKPOINT_{update:06d}.json',{'path':str(path),'sha256':sha(path),'update':update,'candidate':a.candidate,'seed':a.seed})
    record(out/f'COMPLETED_{start:03d}_{a.until:03d}.json',{'update':a.until,'epoch':time.time(),'hard_stop_reached':a.until==300})

if __name__=='__main__':main()

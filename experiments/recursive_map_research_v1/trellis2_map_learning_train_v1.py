"""Only train an additive object-map branch; official native prior stays frozen."""
from paths import JOB, read, save, sha
from trellis2_prepare_v1 import DEST as INSTALL, REPO
from trellis2_map_learning_prepare_fix1 import OUT
import os, sys, argparse, random, time, json
os.environ['CUDA_HOME']=str(INSTALL/'environment')
os.environ['PATH']=str(INSTALL/'environment/bin')+':'+os.environ['PATH']
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from PIL import Image
from trellis2 import models
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
from trellis2_map_object_module_v1 import Trellis2MapObjectConditioner, native_world_grid, INJECTION_BLOCKS

DINO='/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/facebook/dinov3-vitl16-pretrain-lvd1689m'


def native_model():
    m=models.from_pretrained(str(INSTALL/'weights/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16'))
    return m.cuda().eval().requires_grad_(False)


def load_map(record):
    assert sha(record['map_input'])==record['map_sha256']
    with np.load(record['map_input']) as z:
        return {k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}


def clean(record):
    assert sha(record['structure_latent'])==record['structure_sha256']
    z=torch.load(record['structure_latent'],map_location='cpu',weights_only=True)
    assert z['codec_version']=='original immutable native sparse structure codec' and z['not_SLAT_teacher']
    x=z['S0'].float().cuda()
    assert x.shape==(1,8,16,16,16) and torch.isfinite(x).all()
    return x,z['surface_active'].cuda()


class BoundMapReader:
    def __init__(self, native, branch):
        self.native=native
        self.branch=branch
        self.context=None
        self.hooks=[]
        for i in INJECTION_BLOCKS:
            def hook(_module,_arguments,output,index=i):
                if self.context is None:
                    return output
                return self.branch.add_to(output,self.context,index)
            self.hooks.append(native.blocks[i].register_forward_hook(hook))

    def __call__(self,x,t,cond,mapbatch=None):
        self.context=None if mapbatch is None else self.branch.prepare(mapbatch,native_world_grid(device=x.device))
        return self.native(x,t,cond)


def image_cond(plan):
    encoder=DinoV3FeatureExtractor(DINO).cuda()
    encoder.model.eval().requires_grad_(False)
    with torch.no_grad():
        c=encoder([Image.open(plan['appearance_condition']).convert('RGB')]).detach()
    del encoder
    torch.cuda.empty_cache()
    return c


def heldout(reader,records,cond):
    result=[]
    reader.branch.eval()
    with torch.no_grad():
        for index,r in enumerate(records[:12]):
            x0,_=clean(r)
            noise=torch.randn(x0.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(993100+index))
            t=torch.full((1,),.6,device='cuda')
            x=(1-t[:,None,None,None,None])*x0+(1e-5+(1-1e-5)*t[:,None,None,None,None])*noise
            target=(1-1e-5)*noise-x0
            mapbatch=load_map(r)
            baseline=reader(x,t*1000,cond,None)
            correct=reader(x,t*1000,cond,mapbatch)
            # Same per-object and local-map values, wrong common spatial address.
            shuffled={k:(v.flip(-1,-2) if v.ndim==4 else v) for k,v in mapbatch.items()}
            shuf=reader(x,t*1000,cond,shuffled)
            result.append({'scene_id':r['scene_id'],'family_id':r['family_id'],
                'baseline_flow_MSE':float((baseline.float()-target).square().mean()),
                'correct_flow_MSE':float((correct.float()-target).square().mean()),
                'shuffled_flow_MSE':float((shuf.float()-target).square().mean())})
    reader.context=None
    reader.branch.train()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    plan=read(OUT/'PLAN.json');assert a.seed in plan['seeds']
    torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    target=OUT/f'seed{a.seed}';target.mkdir(parents=True,exist_ok=True)
    cond=image_cond(plan)
    native=native_model()
    branch=Trellis2MapObjectConditioner().cuda()
    reader=BoundMapReader(native,branch)
    train=[r for r in plan['records'] if r['split']=='adapter_TRAIN']
    val=[r for r in plan['records'] if r['split']=='adapter_HELDOUT']
    assert not ({r['family_id'] for r in train}&{r['family_id'] for r in val})
    r=train[0];x0,surface=clean(r);mapbatch=load_map(r)
    t=torch.full((1,),.6,device='cuda')
    noise=torch.randn_like(x0);x=.4*x0+(1e-5+(1-1e-5)*.6)*noise
    if a.preflight:
        start=time.time()
        dec=models.from_pretrained(str(INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
        enc=models.from_pretrained(str(JOB.parent/'map_native_scene_dit_v1/weights/codec/ckpts/ss_enc_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
        with torch.no_grad():
            reencoded=enc(surface.float(),sample_posterior=False).float()
            encoder_relative_error=float((reencoded-x0).abs().mean()/x0.abs().mean().clamp_min(1e-8))
            decoded=dec(x0)>0
            intersection=(decoded&surface.bool()).sum()
            union=(decoded|surface.bool()).sum()
            iou=float(intersection/union.clamp_min(1))
            no_map=reader(x,t*1000,cond,None)
            zero_map=reader(x,t*1000,cond,mapbatch)
            difference=float((no_map-zero_map).abs().max())
        # Codec reconstruction error is NOT an interface incompatibility.
        # Compare the actual native encoder target instead of inventing an IoU gate.
        assert encoder_relative_error<.02, ('NATIVE_TARGET_ENCODING_INCOMPATIBLE',encoder_relative_error)
        assert difference==0, ('NOT_STRICT_IDENTITY',difference)
        del dec,enc
        predicted=reader(x,t*1000,cond,mapbatch)
        loss=(predicted.float()-((1-1e-5)*noise-x0)).square().mean()
        loss.backward()
        grads={n:float(p.grad.float().norm()) for n,p in branch.named_parameters() if p.grad is not None}
        assert all(np.isfinite(g) for g in grads.values()) and grads['Wo.weight']>0
        assert all(p.grad is None for p in native.parameters())
        save(target/'PREFLIGHT.json',{'status':'NATIVE_TARGET_AND_REAL_BACKWARD_PASS',
            'source_TRAIN':r['scene_id'],'native_target_decoder_IoU':iou,
            'native_target_encoder_relative_error':encoder_relative_error,
            'strict_zero_identity_maxdiff':difference,'branch_gradients':grads,
            'frozen_parent_gradients_none':True,'training_updates':0,
            'trainable_parameters':sum(p.numel() for p in branch.parameters()),
            'frozen_prior_parameters':sum(p.numel() for p in native.parameters()),
            'seconds':time.time()-start,'allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
        print('NATIVE_BACKWARD_PASS',iou,grads['Wo.weight'],flush=True)
        return
    assert (OUT/f'seed{plan["seeds"][0]}/PREFLIGHT.json').exists()
    optimizer=torch.optim.AdamW(branch.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'],betas=(.9,.95),eps=1e-8)
    order=list(range(len(train)));random.Random(993000).shuffle(order)
    trajectory=target/'train_log.jsonl'
    assert not trajectory.exists(),'Duplicate optimizer owner'
    initial=heldout(reader,val,cond)
    save(target/'HELDOUT_000000.json',{'units':initial,'diagnostic_only':True})
    start=time.time();sample_exposure={}
    with trajectory.open('x') as log:
        for update in range(1,plan['max_updates']+1):
            tick=time.time();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(plan['effective_batch']):
                idx=order[((update-1)*plan['effective_batch']+micro)%len(order)]
                r=train[idx];assert r['split']=='adapter_TRAIN'
                x0,_=clean(r);m=load_map(r)
                g=torch.Generator(device='cuda').manual_seed(983000+(update-1)*2+micro)
                noise=torch.randn(x0.shape,device='cuda',generator=g)
                t=torch.sigmoid(torch.randn((1,),device='cuda',generator=g)+1)
                tb=t[:,None,None,None,None]
                x=(1-tb)*x0+(1e-5+(1-1e-5)*tb)*noise
                drop=float(torch.rand((),device='cuda',generator=g))<.1
                pred=reader(x,t*1000,torch.zeros_like(cond) if drop else cond,None if drop else m)
                target_flow=(1-1e-5)*noise-x0
                loss=(pred.float()-target_flow).square().mean()/plan['effective_batch']
                if not drop:
                    loss.backward()
                assert torch.isfinite(loss)
                losses.append(float(loss.detach())*plan['effective_batch'])
                sample_exposure[r['scene_id']]=sample_exposure.get(r['scene_id'],0)+1
                del x0,x,m,pred,loss
            preclip=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1.0))
            assert np.isfinite(preclip) and all(p.grad is None for p in native.parameters())
            optimizer.step()
            row={'update':update,'flow_MSE':float(np.mean(losses)),'preclip_grad_norm':preclip,
                 'sec_update':time.time()-tick,'allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            log.write(json.dumps(row)+'\n');log.flush()
            if update%10==0:print('UPDATE',a.seed,row,flush=True)
            if update in plan['checkpoints']:
                ckpt=target/f'branch_{update:06d}.pt'
                torch.save({'branch_state':branch.state_dict(),'optimizer':optimizer.state_dict(),
                    'updates':update,'seed':a.seed,'plan_sha256':sha(OUT/'PLAN.json'),
                    'native_prior_frozen':True,'map10_TRAIN':False},ckpt)
                save(target/f'CHECKPOINT_{update:06d}.json',{'path':str(ckpt),'sha256':sha(ckpt),
                    'update':update,'seed':a.seed,'training_sample_exposure':sample_exposure,
                    'elapsed_seconds':time.time()-start})
                save(target/f'HELDOUT_{update:06d}.json',{'units':heldout(reader,val,cond),'diagnostic_only':True})
    save(target/'TRAIN_COMPLETE.json',{'updates':plan['max_updates'],'seed':a.seed,
        'native_parent_frozen':True,'sample_exposure':sample_exposure,
        'whole_world_success':False,'next':'real-map reloaded native asset comparisons'})


if __name__=='__main__':main()

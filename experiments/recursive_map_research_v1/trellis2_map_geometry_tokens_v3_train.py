"""Train the shared map-condition module against two frozen native flow stages."""
import argparse
import json
import random
import time
import hashlib
import numpy as np
import torch
from PIL import Image
from paths import read, save, sha
from trellis2_map_geometry_tokens_v3_prepare import OUT
from trellis2_map_learning_train_fix1 import base
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid
from trellis2.modules.sparse import SparseTensor
from trellis2 import models


def initialization(arm, seed):
    plan = read(OUT / 'PLAN.json')
    pairs = read(OUT / 'PAIRED_NATIVE_MANIFEST.json')['records']
    parent = plan['parent_checkpoints'][str(seed)]
    assert sha(parent['path']) == parent['sha256']
    native_ss = base.native_model()
    native_shape = models.from_pretrained(str(base.INSTALL / 'weights/TRELLIS.2-4B/ckpts' / plan['fine_flow'])).cuda().eval().requires_grad_(False)
    image_conditions = {}
    if arm == 'original':
        encoder = base.DinoV3FeatureExtractor(base.DINO)
        encoder.cuda()
        encoder.model.eval().requires_grad_(False)
        image = Image.open(plan['appearance_condition']).convert('RGB')
        for stage, resolution in [('SS',512), ('shape',1024)]:
            encoder.image_size = resolution
            with torch.no_grad():
                image_conditions[stage] = encoder([image]).detach()
        del encoder
        torch.cuda.empty_cache()
    # Exact same learned-parent state and head initialization in the matched pair.
    torch.manual_seed(seed)
    branch = Trellis2MapGeometryTokenConditioner().cuda()
    checkpoint = torch.load(parent['path'], map_location='cpu', weights_only=False)
    missing, unexpected = branch.load_state_dict(checkpoint['branch_state'], strict=False)
    assert not unexpected and set(missing) == {'map_condition_head.0.weight','map_condition_head.0.bias',
         'map_condition_head.2.weight','map_condition_head.2.bias'}
    reader = base.BoundMapReader(native_ss, branch)
    config = read(base.JOB / 'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    mean = torch.tensor(config['shape_slat_normalization']['mean'], device='cuda')[None]
    std = torch.tensor(config['shape_slat_normalization']['std'], device='cuda')[None]
    return plan, pairs, branch, reader, native_ss, native_shape, image_conditions, mean, std


def clean_shape(record, mean, std):
    assert sha(record['native_shape_target']) == record['native_shape_sha256']
    with np.load(record['native_shape_target']) as target:
        features = torch.from_numpy(target['feats'].copy()).cuda().float()
        coords = torch.from_numpy(target['coords'].copy()).cuda().int()
    assert features.shape[1] == 32 and bool(((coords[:,1:]>=0)&(coords[:,1:]<64)).all())
    return SparseTensor(feats=(features-mean)/std, coords=coords)


def stage_prediction(record, stage, seed_key, t_fixed, arm, components, m=None, train=True):
    plan, pairs, branch, reader, native_ss, native_shape, images, mean, std = components
    m = base.load_map(record) if m is None else m
    context = branch.prepare(m, native_world_grid(device='cuda'))
    generator = torch.Generator(device='cuda').manual_seed(seed_key)
    t = (torch.sigmoid(torch.randn((1,),device='cuda',generator=generator)+1)
         if t_fixed is None else torch.tensor([t_fixed],device='cuda'))
    dropped = train and float(torch.rand((),device='cuda',generator=generator)) < .1
    condition = branch(context, images.get(stage), arm='reference' if arm=='original' else 'separated',
                       negative_cfg=dropped)
    if stage == 'SS':
        x0,_ = base.clean(record)
        noise = torch.randn(x0.shape,device='cuda',generator=generator)
        tb=t[:,None,None,None,None]
        x=(1-tb)*x0+(1e-5+(1-1e-5)*tb)*noise
        reader.context=condition.context
        prediction=native_ss(x,t*1000,condition.features).float()
        target=(1-1e-5)*noise-x0
    else:
        x0=clean_shape(record,mean,std)
        noise=torch.randn(x0.feats.shape,device='cuda',generator=generator)
        x=x0.replace((1-t)*x0.feats+(1e-5+(1-1e-5)*t)*noise)
        prediction=native_shape(x,t*1000,condition.features).feats.float()
        target=(1-1e-5)*noise-x0.feats
    loss=(prediction-target).square().mean()
    assert torch.isfinite(loss)
    return loss, dropped


def heldout(arm, components):
    branch=components[2]
    branch.eval()
    records=[r for r in components[1] if r['split']=='adapter_HELDOUT'][:12]
    result=[]
    with torch.no_grad():
        for index,record in enumerate(records):
            correct=base.load_map(record)
            shuffled={k:(v.flip(-1,-2) if v.ndim==4 else v) for k,v in correct.items()}
            row={'scene_id':record['scene_id'],'family_id':record['family_id']}
            for stage_index,stage in enumerate(['SS','shape']):
                for name,m in [('correct',correct),('shuffled',shuffled)]:
                    loss,_=stage_prediction(record,stage,993100+index*2+stage_index,.6,arm,components,m,train=False)
                    row[name+'_'+stage+'_flow_MSE']=float(loss)
            result.append(row)
    components[3].context=None
    branch.train()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['original','separated'],required=True)
    parser.add_argument('--seed',type=int,choices=[953101,953102],required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    components=initialization(args.arm,args.seed)
    plan,pairs,branch,reader,native_ss,native_shape,images,mean,std=components
    train=[r for r in pairs if r['split']=='adapter_TRAIN']
    optimizer=torch.optim.AdamW(branch.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.95),eps=1e-8)
    directory=OUT/args.arm/f'seed{args.seed}'
    directory.mkdir(exist_ok=True)
    order=list(range(len(train)))
    random.Random(993000).shuffle(order)
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in branch.state_dict().values())).hexdigest()
    source={'arm':args.arm,'seed':args.seed,'starting_branch_sha256':initial_hash,
            'parent':plan['parent_checkpoints'][str(args.seed)],
            'trainable_parameters':sum(p.numel() for p in branch.parameters()),
            'native_parameters_frozen':sum(p.numel() for n in [native_ss,native_shape] for p in n.parameters()),
            'geometry_condition_sources':['reference+map' if args.arm=='original' else 'map_only']*2,
            'reference_image_read_in_separated_training':False,
            'native_shape_TARGET_coords_training_only':True}
    if not args.preflight:
        assert (OUT/'PREFLIGHTS_COMPLETE.json').exists()
        save(directory/'TRAIN_START.json',source)
        save(directory/'HELDOUT_000000.json',{'units':heldout(args.arm,components),'diagnostic_only':True})
    updates=2 if args.preflight else 300
    exposures={}
    preflight_rows=[]
    filename='preflight_log.jsonl' if args.preflight else 'train_log.jsonl'
    start=time.time()
    with (directory/filename).open('x') as log:
        for update in range(1,updates+1):
            tick=time.time()
            optimizer.zero_grad(set_to_none=True)
            values={'SS':[],'shape':[]}
            increments={'SS':[],'shape':[]}
            for micro in range(2):
                record=train[0] if args.preflight else train[order[((update-1)*2+micro)%len(order)]]
                for phase,stage in enumerate(['SS','shape']):
                    before=branch.map_condition_head[2].weight.grad
                    before=None if before is None else before.detach().clone()
                    loss,dropped=stage_prediction(record,stage,983000+(update-1)*4+micro*2+phase,
                                                  .6 if args.preflight else None,args.arm,components,train=not args.preflight)
                    values[stage].append(float(loss.detach()))
                    if not dropped:
                        (loss/4).backward()
                        after=branch.map_condition_head[2].weight.grad
                        increment=after if before is None else after-before
                        increments[stage].append(float(increment.float().norm()))
                exposures[record['scene_id']]=exposures.get(record['scene_id'],0)+1
            preclip=float(torch.nn.utils.clip_grad_norm_(branch.parameters(),1))
            assert np.isfinite(preclip) and all(p.grad is None for n in [native_ss,native_shape] for p in n.parameters())
            optimizer.step()
            row={'update':update,'SS_flow_MSE':float(np.mean(values['SS'])),
                 'shape_flow_MSE':float(np.mean(values['shape'])),
                 'SS_head_gradient':float(np.mean(increments['SS'])) if increments['SS'] else 0,
                 'shape_head_gradient':float(np.mean(increments['shape'])) if increments['shape'] else 0,
                 'preclip_grad_norm':preclip,'sec_update':time.time()-tick,
                 'allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                 'reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
            log.write(json.dumps(row)+'\n');log.flush()
            if args.preflight:
                assert row['SS_head_gradient']>0 and row['shape_head_gradient']>0
                preflight_rows.append(row)
            if update%10==0 or args.preflight:
                print('JOINT_NATIVE_UPDATE',args.arm,args.seed,row,flush=True)
            if not args.preflight and update in [20,100,300]:
                path=directory/f'branch_{update:06d}.pt'
                torch.save({'branch_state':branch.state_dict(),'optimizer':optimizer.state_dict(),
                    'updates':update,'starting_parent_updates':100,'seed':args.seed,'arm':args.arm,
                    'plan_sha256':sha(OUT/'PLAN.json'),'parent_checkpoint':plan['parent_checkpoints'][str(args.seed)],
                    'native_parents_frozen':True,'map10_TRAIN':False,
                    'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()},path)
                save(directory/f'CHECKPOINT_{update:06d}.json',{'path':str(path),'sha256':sha(path),
                     'additional_updates':update,'starting_parent_updates':100,'exposures':exposures})
                save(directory/f'HELDOUT_{update:06d}.json',{'units':heldout(args.arm,components),'diagnostic_only':True})
    reader.context=None
    if args.preflight:
        save(directory/'PREFLIGHT.json',{**source,'status':'BOTH_NATIVE_STAGES_REAL_OPTIMIZER_PASS',
             'discarded_preflight_updates':2,'reset_before_primary_training':True,
             'rows':preflight_rows,'seconds':time.time()-start})
    else:
        save(directory/'TRAIN_COMPLETE.json',{**source,'additional_updates':300,'exposures':exposures,
             'no_automatic_extension':True,'seconds':time.time()-start})


if __name__=='__main__':
    main()

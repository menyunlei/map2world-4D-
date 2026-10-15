"""One zero-training component crossover. Existing weights, fixed TRAIN probes."""
from paths import JOB,read,save,sha
from trellis2_coverage_v12_prepare import OUT as V12,SEEDS,ARMS
import argparse,copy,time,numpy as np
OUT=V12/'diagnosis_v1/crossover'
GROUPS=[('coverage_map_parent_lora',1000),('parent_map_coverage_lora',1000)]

def register():
    assert not OUT.exists();p=copy.deepcopy(read(V12/'PLAN.json'))
    p.update({'purpose':'causal localization, not a new trained model or generalization result',
      'probe_indices':[0,1,6,7],'mesh_indices':[0,1],'training_updates':0,
      'combinations':GROUPS,'GPU_cap_seconds':900,'one_fixed_probe_no_retries_or_seed_search':True,
      'stop':'stop after the two crossed combinations; do not train automatically',
      'only_GPUs':[0,1],'prior_CPU_evidence':str(V12/'diagnosis_v1/CPU_EVIDENCE.json')})
    save(OUT/'PLAN.json',p)
    files=['trellis2_coverage_diagnosis_v13.py','trellis2_coverage_diagnosis_v13_owner.py',
      'trellis2_coverage_v12_eval.py','trellis2_coverage_v12_train.py','trellis2_structure_learnability_v8_train.py']
    frozen={str(JOB/f):sha(JOB/f) for f in files};frozen[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    for seed in SEEDS:
        for ck in [p['parent_checkpoints'][str(seed)],read(V12/'small/bounded_coverage'/f'seed{seed}/train/CHECKPOINT_001000.json')]:
            assert sha(ck['path'])==ck['sha256'];frozen[ck['path']]=ck['sha256']
    save(OUT/'SOURCE_FREEZE.json',frozen);print('CROSSOVER_REGISTERED',flush=True)

def structure(seed):
    import torch
    from trellis2_coverage_v12_train import setup,train
    from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
    from trellis2_map_object_module_v1 import native_world_grid
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2 import models
    from trellis2_map_learning_train_fix1 import base
    from trellis2_structure_learnability_v8_eval import tile,iou
    from trellis2_coverage_diagnosis_v13_cpu import support_stats
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    p=read(OUT/'PLAN.json');_,native,branch,reader,_=setup(ARMS[1],seed)
    parent=torch.load(p['parent_checkpoints'][str(seed)]['path'],map_location='cpu',weights_only=False)
    coverage=torch.load(read(V12/'small/bounded_coverage'/f'seed{seed}/train/CHECKPOINT_001000.json')['path'],map_location='cpu',weights_only=False)
    native.eval().requires_grad_(False);branch.eval().requires_grad_(False)
    assert not any(x.requires_grad for x in list(native.parameters())+list(branch.parameters()))
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=p['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    for name,update in GROUPS:
        branch.load_state_dict((coverage if name==GROUPS[0][0] else parent)['branch_state'],strict=True)
        train.load_lora_state_dict(native,(parent if name==GROUPS[0][0] else coverage)['lora_state'])
        root=OUT/'evaluation'/f'seed{seed}'/name/f'update{update:06d}';rows=[]
        for gen in p['generation_seeds']:
            noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(gen)).cuda()
            for mode in ['correct','edit']:
                for index in p['probe_indices']:
                    r=p['edit_records' if mode=='edit' else 'evaluation_records'][index];m=train.load_record(r);wrapper.mapbatch=m
                    with torch.no_grad():
                        features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
                        latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'existing_checkpoint_crossover'},
                          neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**cfg['params'],verbose=False).samples
                        support=(decoder(latent)>0)[0,0].cpu().numpy()
                    d=root/f'noise{gen}'/mode/f'object{index:03d}';d.mkdir(parents=True,exist_ok=False)
                    with (d/'SS_GENERATED.npz').open('xb') as h:np.savez_compressed(h,latent=latent.float().cpu().numpy(),native_support64=support)
                    fp=m['footprints'][0,int(m['target_object'])].cpu().numpy()>.5
                    with np.load(r['native_target']) as z:gt=z['support64'][0,0].astype(bool)
                    row={'seed':seed,'arm':name,'index':index,'noise':gen,'mode':mode,'object':r['scene_id']+':'+str(r['object_id']),
                      'map_projection_IoU':iou(support.any(2).T,fp),'target_support_IoU_diagnostic':iou(support,gt),
                      'footprint_coverage':float((support.any(2).T&fp).sum()/max(1,fp.sum())),
                      'map_features_RMS':float(features.float().square().mean().sqrt()),**support_stats(support),
                      'pure_noise':True,'training_updates':0,'not_a_new_learned_checkpoint':True}
                    save(d/'METRICS.json',row);rows.append(row)
                    tile(f'{name} object{index} {mode} XY={row["map_projection_IoU"]:.3f}',support,gt,fp).save(d/'STRUCTURE.png')
                    print('CROSSOVER',seed,name,index,gen,mode,row['map_projection_IoU'],flush=True)
        save(root/'RESULTS.json',{'rows':rows,'existing_weights_only':True,'no_target_in_sampling':True})
    save(OUT/'evaluation'/f'seed{seed}/STRUCTURE_COMPLETE.json',{'train_objects_only':True,'updates':0})

def reuse(mode,seed):
    import trellis2_coverage_v12_eval as old
    old.OUT=OUT;old.GROUPS=GROUPS
    {'mesh':old.mesh,'views':old.views}[mode](seed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['register','structure','mesh','views'],required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS);a=parser.parse_args()
    if a.mode=='register':register()
    elif a.mode=='structure':structure(a.seed)
    else:reuse(a.mode,a.seed)

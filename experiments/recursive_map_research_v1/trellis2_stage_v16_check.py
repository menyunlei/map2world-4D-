"""Small fixed pure-noise preservation check; no added training or Shape work."""
from trellis2_stage_v16 import OUT,V12,V15,SEEDS
from trellis2_structure_learnability_v8_train import setup,load_lora_state_dict,load_record
from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
from trellis2_map_object_module_v1 import native_world_grid
from trellis2_map_learning_train_fix1 import base
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import read,save,sha
import argparse,torch,numpy as np

def run(seed):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    p=read(V12/'PLAN.json');old,native,branch,reader,_=setup('structure_lora',seed)
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=old['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    root=OUT/'preservation'/f'seed{seed}';root.mkdir(parents=True,exist_ok=False);rows=[]
    indices=[0,1,24,25]
    for update in [0,20,100]:
        if update:
            ck=read(OUT/'repair'/f'seed{seed}/CHECKPOINT_{update:06d}.json');assert sha(ck['path'])==ck['sha256']
            state=torch.load(ck['path'],map_location='cpu',weights_only=False)
            branch.load_state_dict(state['branch_state']);load_lora_state_dict(native,state['lora_state']);del state
            branch.eval().requires_grad_(False);native.eval().requires_grad_(False)
        for i in indices:
            r=p['evaluation_records'][i];m=load_record(r);fp=m['footprints'][0,int(m['target_object'])].cpu().numpy()>.5
            with np.load(r['native_target']) as z:target=z['support64'][0,0].astype(bool)
            src=V12/f'evaluation/seed{seed}/six_object_rehearsal/update001000/noise970201/correct/object{i:03d}/SS_GENERATED.npz'
            if not update:
                with np.load(src) as z:q=z['native_support64'].astype(bool)
            else:
                noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(970201)).cuda()
                wrapper.mapbatch=m
                with torch.no_grad():
                    features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
                    latent=sampler.sample(wrapper,noise,cond={'features':features,'mapped':True,'source':'same_contract'},
                        neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'zero'},**cfg['params'],verbose=False).samples
                    q=(decoder(latent)>0)[0,0].cpu().numpy()
                with (root/f'update{update:06d}_object{i:03d}.npz').open('xb') as f:np.savez_compressed(f,native_support64=q,latent=latent.float().cpu().numpy())
            xy=q.any(2).T
            row={'seed':seed,'update':update,'index':i,'family':r['family_id'],'scene':r['scene_id'],
                'group':'TRAIN' if i<24 else 'heldout_DEV','cells':int(q.sum()),'empty':not bool(q.any()),
                'map_IoU':float((xy&fp).sum()/max(1,(xy|fp).sum())),
                'native_target_IoU_diagnostic':float((q&target).sum()/max(1,(q|target).sum())),
                'target_used_for_generation':False,'generation_seed':970201,'baseline_reused':not bool(update)}
            rows.append(row);print('PRESERVATION_PURE_NOISE',row,flush=True)
    save(root/'RESULTS.json',{'rows':rows,'fixed_indices':indices,'not_generalization_proof':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();run(a.seed)

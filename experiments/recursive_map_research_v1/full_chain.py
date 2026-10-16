"""Predicted Q -> learned native Detail -> unchanged native Gaussian decoder.

No target Q/SLAT/geometry, masking, mesh repair or procedural replacement.
Fixed cameras are evaluative observations, not conditioning of the generator.
"""
from runtime import *
import argparse,time
import imageio.v2 as imageio
from PIL import Image
from trellis.modules.sparse import SparseTensor
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from map_native_flow import MapCondition
from codec_adapt_graph import native_model

def frozen_detail():
    from dreamx_scene_flow import DreamXSceneFlow
    flow=DreamXSceneFlow('detail','D2')
    meta=read(D2/'D2_local_map/detail/CHECKPOINT_005000.json')
    assert sha(meta['path'])==meta['sha256']
    state=torch.load(meta['path'],map_location='cpu',weights_only=False)
    pars={k:p for k,p in flow.named_parameters() if p.requires_grad}
    assert set(pars)==set(state['trainable_state'])
    for k,p in pars.items():p.copy_(state['trainable_state'][k])
    del state
    return flow.eval().requires_grad_(False),meta

def native_gaussian_decoder():
    meta=read(NATIVE/'audit/CODEC_FREEZE_V1.json')
    state=torch.load(meta['checkpoint_path'],map_location='cpu',weights_only=False)
    decoder=native_model('slat_dec_gs_swin8_B_64l8gs32_fp16').eval().requires_grad_(False)
    decoder.load_state_dict(state['decoder'],strict=True)
    return decoder,sha(meta['checkpoint_path'])

def save_gaussian(g,path):
    fields=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']
    torch.save({'init_params':g.init_params,**{k:None if getattr(g,k) is None else getattr(g,k).detach().cpu() for k in fields}},path)

def render_native(g,out,center):
    from scene_render import camera,render_gs
    ox,oy=center
    # Same fixed 3D observation path for all candidates/seeds; no target height.
    W,H=512,384
    static={'aerial':([ox+.6,oy-.6,16],[ox,oy,0]),
            'oblique':([ox+10,oy-10,9],[ox,oy,3]),
            'street':([ox+8,oy-8,2],[ox,oy,2])}
    for name,(eye,look) in static.items():
        E,K,_=camera(eye,look,W,H,55)
        rgb=render_gs(g,E,K,W,H)
        assert torch.isfinite(rgb).all()
        Image.fromarray((rgb.float().clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(out/f'{name}.png')
    poses=[]
    with imageio.get_writer(out/'native_oblique_4s.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1) as video:
        for i in range(64):
            theta=-.9+.7*i/63
            eye=[ox+12*np.sin(theta),oy-12*np.cos(theta),8.]
            look=[ox,oy,2.5];E,K,_=camera(eye,look,W,H,55)
            rgb=render_gs(g,E,K,W,H)
            assert torch.isfinite(rgb).all()
            video.append_data((rgb.float().clamp(0,1).cpu().numpy()*255).astype(np.uint8))
            poses.append({'eye_m':eye,'look_m':look})
    save(out/'RENDER.json',{'poses':poses,'native_frames':64,'fps':16,'duration':4,
         'short_native_3D_inspection_not_world_exploration_demo':True,'depth_or_height_not_target_conditioned':True,
         'no_interpolation_or_frame_copy':True,'human_review':'PENDING'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--update',type=int,required=True);a=ap.parse_args()
    torch.set_grad_enabled(False);torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.candidate in ['g1_a','g2_a']:flow,meta=frozen_detail()
    else:
        assert a.candidate=='g1_b'
        flow,branch,_,_=initialize(a.candidate,a.seed)
        meta=read(JOB/'runs'/a.candidate/f'seed{a.seed}'/f'CHECKPOINT_{a.update:06d}.json')
        assert sha(meta['path'])==meta['sha256']
        state=torch.load(meta['path'],map_location='cpu',weights_only=False)
        assert set(branch)==set(state['branch_state'])
        for k,p in branch.items():p.copy_(state['branch_state'][k])
        del state;flow.eval().requires_grad_(False)
    branch_checks={'mapped_nonzero':0,'unknown_exact_zero':0}
    if a.candidate=='g1_b':
        original_read=flow.branch.read
        def checked_read(hidden,prepared,site):
            value=original_read(hidden,prepared,site)
            if site==0:
                if not prepared.valid.any():
                    assert value.count_nonzero()==0,'unknown-map Detail branch must remain exactly zero'
                    branch_checks['unknown_exact_zero']+=1
                elif value.count_nonzero()>0:
                    branch_checks['mapped_nonzero']+=1
            return value
        flow.branch.read=checked_read
    decoder,codec_sha=native_gaussian_decoder()
    norm=read(NATIVE/'teacher_latents_frozen_v1/NORMALIZATION.json')
    mean=torch.tensor(norm['mean'],device='cuda');std=torch.tensor(norm['std'],device='cuda')
    packet=read(JOB/'packets'/f'{a.candidate}_{a.update}.json')
    output=JOB/'native'/a.candidate/f'seed{a.seed}'/f'plus{a.update:03d}'
    sampler=FlowEulerGuidanceIntervalSampler(1e-5);contexts={};complete=[]
    for r in packet['rows']:
        if not r.get('native_witness',False):continue
        qpath=Path(r['structure_source']) if a.candidate=='g1_b' else JOB/'evaluation'/a.candidate/f'seed{a.seed}'/f'plus{a.update:03d}'/r['scene_id']/f'seed{r["seed_index"]}'/'RAW_STRUCTURE.pt'
        suffix=r.get('structure_arm','own')
        out=output/suffix/r['scene_id']/f'seed{r["seed_index"]}';out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():
            saved=read(out/'COMPLETE.json')
            for name,h in saved['files'].items():assert sha(out/name)==h
            complete.append(saved);continue
        assert not (out/'RAW_NATIVE_GENERATION.pt').exists(),'partial native output requires additive recovery'
        tick=time.time();q=torch.load(qpath,map_location='cpu',weights_only=True)['predicted_Q'].int()
        if len(q)==0:
            save(out/'EMPTY_STRUCTURE.json',{'Q_source':str(qpath),'Q_sha256':sha(qpath),'not_repaired':True})
            print('EMPTY_STRUCTURE',str(out),flush=True);continue
        coords=q.cuda();tensors=[]
        for s in [r['text'],r['negative']]:
            if s['sha256'] not in contexts:
                assert sha(s['path'])==s['sha256']
                contexts[s['sha256']]=torch.load(s['path'],map_location='cpu',weights_only=True)['context'][None].cuda()
            tensors.append(contexts[s['sha256']])
        assert sha(r['map_path'])==r['map_sha256']
        with np.load(r['map_path']) as z:m=torch.from_numpy(z['map_tensor'].copy())[None].float().cuda()
        unknown=torch.zeros_like(m);unknown[:,5]=1
        world=torch.randn((64,64,64,8),device='cuda',generator=torch.Generator(device='cuda').manual_seed(r['detail_seed']))
        noise=SparseTensor(world[coords[:,1],coords[:,2],coords[:,3]],coords);del world
        torch.save({'Q':q,'detail_noise':noise.feats.cpu(),'detail_seed':r['detail_seed'],'Q_source':str(qpath),'Q_sha256':sha(qpath)},out/'RAW_INPUT_NOISE.pt')
        def predict(x,t,c):return flow(x,t,c.text,c.map)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            L=sampler.sample(predict,noise,MapCondition(tensors[0],m),neg_cond=MapCondition(tensors[1],unknown),
                steps=25,rescale_t=3,cfg_strength=7.5,cfg_interval=[.5,.95],verbose=False).samples
            assert torch.equal(L.coords,coords),'Detail moved Q'
            assert torch.isfinite(L.feats).all()
            latent=L.replace(L.feats.float()*std+mean);g=decoder(latent)[0]
        torch.save({'Q':q,'generated_L_normalized':L.feats.cpu(),'generated_L_codec_units':latent.feats.cpu(),
           'target_SLAT_input':False,'target_Q_input':False,'learned_Detail_checkpoint':meta,'codec_sha256':codec_sha},out/'RAW_NATIVE_GENERATION.pt')
        save_gaussian(g,out/'NATIVE_GAUSSIANS.pt')
        render_native(g,out,r['map_center_xy'])
        evidence={'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'seconds':time.time()-tick,
          'Q_unchanged':True,'Q_cells':len(q),'no_oracle':True,'HUMAN_REVIEW_PENDING':True,'packet':r}
        save(out/'COMPLETE.json',evidence);complete.append(evidence)
        print('NATIVE_FULL_CHAIN',r['scene_id'],suffix,r['seed_index'],time.time()-tick,flush=True)
        del L,latent,g,noise,coords;torch.cuda.empty_cache()
    save(output/'COMPLETE.json',{'units':len(complete),'candidate':a.candidate,'update':a.update,'checkpoint':meta,'human_review':'PENDING'})
    if a.candidate=='g1_b':
        assert branch_checks['mapped_nonzero'] and branch_checks['unknown_exact_zero']
        save(output/'TRAINED_MAP_BRANCH_CHECKS.json',branch_checks)

if __name__=='__main__':main()

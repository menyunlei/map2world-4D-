"""Matched, bounded CityTopia road-stage audit. Frozen original checkpoints.

No new road surface, texture, training, camera registration or image enhancement.
Diagnostic hooks observe the native model and leave its outputs unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'citytopia_map10_checkpoint_trial_20261027_v1'
spec = importlib.util.spec_from_file_location('prior_trial', PRIOR / 'run_trial.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sha, save, progress = base.sha, base.save, base.progress


def quant(a):
    import numpy as np
    a = np.asarray(a); a = a[np.isfinite(a)]
    if not a.size:
        return {'count': 0}
    return dict(count=int(a.size), mean=float(a.mean()), **dict(zip(
        ['min', 'p05', 'median', 'p95', 'max'], map(float, np.percentile(a, [0, 5, 50, 95, 100])))))


def image(path, rgb):
    import cv2
    assert not path.exists()
    assert cv2.imwrite(str(path), rgb[..., ::-1])


def rgb(tensor):
    import numpy as np
    import utils.helpers as helpers
    return (helpers.tensor_to_image(tensor, 'RGB') * 255).astype(np.uint8)


def setup(out):
    import cv2
    import numpy as np
    import torch
    os.chdir(base.REPO)
    sys.path[:0] = [str(base.REPO), str(base.REPAIR)]
    from overlay_loader import load
    native = load(base.REPO)
    torch.manual_seed(953101); np.random.seed(953101)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    reference = json.loads((PRIOR / 'run_v1/CONTRACT.json').read_text())
    hashes = reference['source_hashes']
    for path, expected in hashes.items():
        assert sha(path) == expected, ('preserved source changed', path)
    bg, bldg, car = native.get_models('CITY_SAMPLE', str(base.WEIGHTS/'CityDreamer4D-BG.pth'), str(base.WEIGHTS/'CityDreamer4D-BLDG.pth'), None)
    assert car is None
    for name, model in [('CityDreamer4D-BG.pth', bg), ('CityDreamer4D-BLDG.pth', bldg)]:
        ck = torch.load(base.WEIGHTS/name, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['gancraft_g'], strict=True)
        model.eval().requires_grad_(False)
        if name.endswith('BG.pth'):
            save(out/'CHECKPOINT_TRAIN_CONFIG.json', dict(ck['cfg'].TRAIN.GANCRAFT))
        del ck
    projections, footprints = native.get_city_sample_layout(str(base.SPATIAL), [10, 30000])
    volumes = {k: native.get_cfg_value(v, 'CITY_SAMPLE') for k, v in [('LAYOUT','VOL_SIZE/LAYOUT'), ('BLDG','VOL_SIZE/BLDG'), ('EXT','VOL_SIZE/EXTEND'), ('CAR','VOL_SIZE/CAR')]}
    cy, cx = np.array(projections['REST']['TD_HF'].shape)//2
    common = {'INST_RANGE':[10,30000], 'MAX_HEIGHT':2560, 'VOL_SIZE':volumes['EXT'], 'CLASS_MAPPER':native._bldg_class_callback}
    hf, ids = native.get_hf_seg_tensor(projections['REST'], int(cx), int(cy), 9, dict(common,FACADE_CID=8), bg.output_device)
    hf_bldg, _ = native.get_hf_seg_tensor(projections['REST'], int(cx), int(cy), 9, dict(common,FACADE_CID=7), bldg.output_device)
    stats = native.get_part_bldg_stats(ids, footprints, int(cx), int(cy), [10,30000])
    saved = torch.load(PRIOR/'run_v1/WORLD_STYLES.pt',map_location='cpu',weights_only=False)
    styles = {int(k):v.to(bg.output_device) for k,v in saved['building_zs'].items()}
    cameras = json.loads((PRIOR/'run_v1/CAMERAS.json').read_text())['cameras']
    cfg = {'HEIGHT':540, 'WIDTH':960, 'PADDING':8, 'VFOV':30.}
    inst = {'DATASET':'CITY_SAMPLE', 'BLDG_INST_RANGE':[10,30000], 'BLDG_INST_MULTIPLIER':4, 'ROAD_CID':1,
            'BG_FACADE_CID':8, 'BG_CAR_CID':0, 'FACADE_CID':7, 'ROOF_CID':8, 'ROOF_OFFSET':1,
            'CAR_INST_RANGE':[30000,32767], 'N_CAR_CLASSES':6, 'CAR_BEV_FILE':None, 'N_VOXEL_SAMPLES':6}
    rest = projections['REST']; road = rest['INS_BEV']==1
    save(out/'INPUT_AUDIT.json', {'source_hashes':hashes, 'road_top_network':quant(rest['TD_HF'][road]),
        'road_bottom_network':quant(rest['BU_HF'][road]), 'road_pixels':int(road.sum()),
        'units_per_metre':4, 'map_registration':'Reused approximately registered Map10; no new georegistration',
        'semantics':{int(k):int(v) for k,v in zip(*np.unique(rest['INS_BEV'][rest['INS_BEV']<10],return_counts=True))},
        'unavailable_conditions':['verified sidewalks','surveyed local ground elevation','ground-truth street RGB'],
        'height_note':'Flat road support is an existing encoding assumption, not surveyed truth.'})
    crop = (slice(cy-384,cy+384),slice(cx-384,cx+384))
    sem = rest['INS_BEV'][crop]; visual = np.full((*sem.shape,3),64,np.uint8)
    visual[sem==1]=[75,105,130]; visual[sem>=100]=[185,170,140]; visual[sem==0]=0
    for c in cameras:
        p=c['citydreamer_pose']['cam_position']; x=int(p['x']-1536+384); y=int(p['y']-1536+384)
        if 0<=x<768 and 0<=y<768: cv2.circle(visual,(x,y),2,(40,255,100),-1)
    image(out/'map_conditions.png',visual)
    progress(phase='loading_full_preserved_volume')
    volume = native.get_seg_volume(projections,None,int(cx),int(cy),volumes,
        {'ROOF_HEIGHT':1,'ROOF_OFFSET':1,'INST_RANGE':[10,30000],'MAX_HEIGHT':2560})
    return dict(native=native,bg=bg,bldg=bldg,hf=hf,hf_bldg=hf_bldg,stats=stats,styles=styles,
                cameras=cameras,cfg=cfg,inst=inst,volumes=volumes,volume=volume,hashes=hashes)


class Trace:
    """Read-only native ray trace; no target or corrected values enter inference."""
    def __init__(self, model):
        import torch
        self.model=model; self.state={}; self.maps={}; self.samples=[]
        self.original={name:getattr(model,name) for name in ['_sample_depth_batched','_get_sampled_coordinates','_forward_perpix_sub','_volum_rendering_relu']}
        def sample(m,*a,**kw):
            result=self.original['_sample_depth_batched'](*a,**kw)
            self.state['sample_depth'], self.state['sample_dists'], self.state['sample_index']=result
            return result
        def coord(m,*a,**kw):
            result=self.original['_get_sampled_coordinates'](*a,**kw)
            self.state['normalized_xyz']=result[0]
            return result
        def sub(m,*a,**kw):
            result=self.original['_forward_perpix_sub'](*a,**kw)
            self.state['sigma']=result[0]
            self.state['saturated']=(result[1].abs()>=1).float().mean(-1,keepdim=True)
            return result
        def weights(m,*a,**kw):
            result=self.original['_volum_rendering_relu'](*a,**kw)
            self.state['weights']=result
            return result
        for name,fun in zip(self.original,[sample,coord,sub,weights]):
            setattr(model,name,types.MethodType(fun,model))

    def collect(self, bounds, rays, opacity):
        import numpy as np
        import torch
        sx,ex,sy,ey,psx,pex,psy,pey=bounds
        sl=(slice(sy-psy,ey-psy),slice(sx-psx,ex-psx))
        s=self.state; w=s['weights']; t=s['sample_depth']/4
        total=w.sum(-2).clamp_min(1e-8)
        depth=(w*t).sum(-2)/total
        var=(w*(t-depth.unsqueeze(-2))**2).sum(-2)/total
        values={'depth':depth[0,...,0], 'depth_std':var.sqrt()[0,...,0], 'opacity':opacity[0],
                'sigma_mean':(torch.relu(s['sigma']).mean(-2))[0,...,0],
                'feature_clip_fraction':s['saturated'].mean(-2)[0,...,0],
                'clipped_coordinates':(s['normalized_xyz'].abs()>=1).any(-1).float().mean(-1)[0]}
        for k,v in values.items():
            if k not in self.maps:self.maps[k]=np.zeros((540,960),np.float32)
            self.maps[k][sy:ey,sx:ex]=v[sl].detach().cpu().numpy()
        labels=torch.gather(rays[0],-2,s['sample_index'])
        near=(rays[0][0,:,:,0,0]==1)
        near[:sy-psy]=False; near[ey-psy:]=False
        near[:,:sx-psx]=False; near[:,ex-psx:]=False
        mask=near.clone(); mask[1::2]=False; mask[:,1::2]=False
        # Subsample diagnostic rays, not render rays.
        ys,xs=torch.where(mask); ys,xs=ys[::31],xs[::31]
        if len(ys):
            self.samples.append({k:v[0,ys,xs].detach().cpu().numpy() for k,v in dict(
                depth_metres=t, sigma=s['sigma'], weights=w, labels=labels, normalized_xyz=s['normalized_xyz']).items()})
        self.state.clear()

    def close(self):
        for name,fn in self.original.items():setattr(self.model,name,fn)


def background(env,rays,trace):
    import torch
    import torchvision
    native=env['native']; cfg=env['cfg']; model=env['bg']; vs=env['volumes']
    hf=env['hf'][:,:,vs['BLDG']:-vs['BLDG'],vs['BLDG']:-vs['BLDG']]
    ids=native.get_background_voxel_ids(rays[0],env['inst'])
    raw=torch.zeros(1,3,540,960,device=model.output_device); blurred=raw.clone()
    blurrer=torchvision.transforms.GaussianBlur(kernel_size=3,sigma=(2,2))
    for sy in range(0,540,135):
        for sx in range(0,960,240):
            ey,ex=sy+135,sx+240
            psx,pex,psy,pey=native.get_pad_img_bbox(sx,ex,sy,ey,540,960,8)
            rr=(ids[:,psy:pey,psx:pex],rays[1][:,psy:pey,psx:pex],rays[2][:,psy:pey,psx:pex],rays[3])
            result,opacity=model(hf_seg=hf,voxel_id=rr[0],depth2=rr[1],raydirs=rr[2],cam_origin=rr[3],ftp_stats=None,z=None,deterministic=True)
            bounds=(sx,ex,sy,ey,psx,pex,psy,pey)
            if trace is not None:trace.collect(bounds,rr,opacity)
            raw[:,:,sy:ey,sx:ex]=native.get_img_without_pad(result,*bounds,8)
            mask=(rr[0][:,None,:,:,0,0]==1).float()
            result=blurrer(result)*mask+result*(1-mask)
            blurred[:,:,sy:ey,sx:ex]=native.get_img_without_pad(result,*bounds,8)
    return raw,blurred


def compose(env,rays,bg_images):
    native=env['native']; results=[t.clone() for t in bg_images]
    visible=native.get_visible_buildings(rays[0][0,:,:,0,0],env['inst'])
    for b in visible.tolist():
        fg,mask=native.render_bldg((135,240),env['bldg'],b,env['hf_bldg'],*rays,
            env['stats'][b],env['styles'][b],env['volumes'],env['cfg'],env['inst'])
        results=[r*(1-mask)+fg*mask for r in results]
    return results


def normal_from_depth(depth,dirs,origin):
    import numpy as np
    p=origin+dirs*depth[...,None]
    dy,dx=np.gradient(p,axis=(0,1))
    n=np.cross(dx,dy); norm=np.linalg.norm(n,axis=-1,keepdims=True)
    n=np.divide(n,norm,out=np.zeros_like(n),where=norm>1e-8)
    n[n[...,2]<0]*=-1
    return n,p


def save_diagnostics(out,index,env,rays,trace,raw,blurred,final):
    import cv2
    import numpy as np
    import torch
    folder=out/f'{index:04d}';folder.mkdir()
    maps=trace.maps
    ids=rays[0][0,:,:,0,0].cpu().numpy()
    first=rays[1][0,:,:,0,0,0].cpu().numpy()/4
    dirs=rays[2][0,:,:,0,:].cpu().numpy(); origin=rays[3][0].cpu().numpy()/4
    road=ids==1; valid=road & (maps['opacity']>.05) & np.isfinite(first)
    core=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    n,p=normal_from_depth(maps['depth'],dirs,origin)
    gn,gp=normal_from_depth(np.nan_to_num(first),dirs,origin)
    angles=np.degrees(np.arccos(np.clip(n[...,2],-1,1)))
    angle_input=np.degrees(np.arccos(np.clip(gn[...,2],-1,1)))
    data=dict(maps,first_id=ids,first_depth=first,raydirs=dirs,origin=origin,normals=n,input_normals=gn)
    np.savez_compressed(folder/'geometry.npz',**data)
    if trace.samples:
        np.savez_compressed(folder/'density_rays.npz',**{k:np.concatenate([s[k] for s in trace.samples]) for k in trace.samples[0]})
    imraw,imblur,imfinal=rgb(raw),rgb(blurred),rgb(final)
    image(folder/'background_native_preblur.png',imraw)
    image(folder/'background_native.png',imblur)
    image(folder/'final_native.png',imfinal)
    image(folder/'density_normals.png',np.where(valid[...,None],(n*.5+.5)*255,0).astype(np.uint8))
    image(folder/'input_normals.png',np.where(road[...,None],(gn*.5+.5)*255,0).astype(np.uint8))
    for key in ['opacity','depth','sigma_mean']:
        value=maps[key]; scale={'opacity':1,'depth':40,'sigma_mean':20}[key]
        c=cv2.applyColorMap((np.clip(value/scale,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
        c[~road]=0; image(folder/(key+'.png'),c)
    before_path=PRIOR/'run_v1/checkpoint_declared'/f'{index:04d}.png'
    previous_equal=None
    if before_path.exists():
        previous_equal=np.array_equal(cv2.imread(str(before_path))[...,::-1],imfinal)
        assert previous_equal, 'Instrumented native baseline differs from saved baseline'
    result={'index':index,'road_pixels':int(road.sum()),'core_road_pixels':int(core.sum()),
        'baseline_pixel_exact':previous_equal,'native_final_sha256':sha(folder/'final_native.png'),
        'input_road_world_z_m':quant(gp[...,2][road]),'input_road_normal_angle_deg':quant(angle_input[core]),
        'density_road_world_z_m':quant(p[...,2][valid]),'density_road_normal_angle_deg':quant(angles[core]),
        'density_depth_minus_coarse_entry_m':quant((maps['depth']-first)[valid]),
        'native_road_blur_change_255':quant(np.abs(imraw.astype(float)-imblur)[road]),
        'compositor_road_change_255':quant(np.abs(imblur.astype(float)-imfinal)[road]),
        **{k:quant(v[road]) for k,v in maps.items()}}
    save(folder/'METRICS.json',result)
    return result


def execute(out,mode,indices):
    import torch
    started=time.perf_counter(); env=setup(out); rows=[]
    with torch.inference_mode():
        for index in indices:
            cam=next(c for c in env['cameras'] if c['index']==index)
            progress(phase='road_native_audit',index=index)
            rays=env['native'].get_voxel_intersection_perspective(env['volume'],cam['citydreamer_pose'],env['cfg'],30.,6)
            trace=Trace(env['bg'].module)
            try:raw,blurred=background(env,rays,trace)
            finally:trace.close()
            final=compose(env,rays,[blurred])[0]
            row=save_diagnostics(out,index,env,rays,trace,raw,blurred,final)
            rows.append(row);progress(phase='frame_complete',**row)
            del raw,blurred,final,rays,trace
    assert all(sha(p)==h for p,h in env['hashes'].items())
    save(out/'RESULT.json',{'mode':mode,'seconds':time.perf_counter()-started,
        'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'frames':rows,'preserved_sources':True,
        'depth_warning':'Opacity-normalized density expectation, not measured ground truth. Compare against the supplied support only.',
        'source_code_sha256':{str(p):sha(p) for p in [Path(__file__),base.REPO/'models/gancraft.py',base.REPO/'core/train.py']}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[4,5],required=True)
    p.add_argument('--mode',choices=['diagnose'],default='diagnose');p.add_argument('--max-seconds',type=int,default=600)
    a=p.parse_args();assert 0<a.max_seconds<=1200
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==base.UUIDS[a.gpu]
    gpu=subprocess.check_output(['nvidia-smi','-i',base.UUIDS[a.gpu],'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    used,free=map(int,gpu.strip().split(','));assert used<256 and free>75000,('occupied GPU; do not preempt',gpu)
    out=ROOT/a.mode;out.mkdir(exist_ok=False)
    save(out/'RUN_CONFIG.json',{'mode':a.mode,'gpu':a.gpu,'max_seconds':a.max_seconds,'seed':953101,
        'frames':[14,31,45],'checkpoint_and_map':'unchanged from previous checkpoint trial','correction':'NONE: diagnosis only'})
    def deadline(*unused):raise TimeoutError('bounded road experiment deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(a.max_seconds)
    try:execute(out,a.mode,[14,31,45])
    except BaseException as e:
        save(out/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

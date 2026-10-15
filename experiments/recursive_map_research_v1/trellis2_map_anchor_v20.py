"""One frozen-prior map-anchor feasibility test. No training or material stage."""
from paths import JOB, read, save, sha
from trellis2_prepare_v1 import DEST, REPO
from pathlib import Path
import sys, os, argparse, time, hashlib, gc
import numpy as np

OUT=JOB/'map_object_world_v1/trellis2_map_anchor_v20'
V17=JOB/'map_object_world_v1/trellis2_native_street_v17'
V19=JOB/'map_object_world_v1/trellis2_map_volume_v19'
V11=JOB/'map_object_world_v1/trellis2_map10_block_v11'
SSENC=JOB.parent/'map_native_scene_dit_v1/weights/codec/ckpts/ss_enc_conv3d_16l8_fp16'
SEED=942217
AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)


def setup():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME=str(DEST/'environment'))
    os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
    sys.path.insert(0,str(REPO))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    return torch


def prepare():
    assert not OUT.exists()
    from PIL import Image,ImageDraw
    p=read(V17/'PLAN.json'); old=read(V19/'PLAN.json');meta=read(V11/'inputs/MAP_INPUT.json')
    OUT.mkdir(parents=True)
    for d in ['logs','audit','gpu_cost','work']:(OUT/d).mkdir()
    # Selection uses map input only. Keep the established native/world transform.
    chosen=[2,3]
    for mode in ['correct','edit']:
        with np.load(old['maps'][mode]['path']) as z:
            fps=z['footprints'][chosen]>.5
            roads=(z['roads'][0]>.5)&(z['roads_known'][0]>0)
        folder=OUT/mode;folder.mkdir()
        np.savez_compressed(folder/'MAP.npz',footprints=fps,roads=roads)
    floor=old['guidance']['floor_native_reference'];floor_cell=int(np.floor((floor+.5)*64))
    plan={'method':'Predicted-clean Structure projection and native re-encoding during Euler sampling',
        'native_structural_inpainting_supported':False,
        'native_model':'Original pretrained TRELLIS.2; Structure forward(x,t,cond), no spatial mask argument',
        'source_commit':p['source_commit'],'runtime':p['runtime'],'input':p['input'],'input_sha256':p['input_sha256'],
        'seed':SEED,'selected_map_objects':chosen,'map_objects':[meta['objects'][i] for i in chosen],
        'map_source':str(V11/'inputs/MAP_INPUT.json'),'map_source_sha256':sha(V11/'inputs/MAP_INPUT.json'),
        'existing_world_transform':meta['coordinate_transform'],'world_z_translation':old['world_from_native_z_translation'],
        'floor_native_reference':floor,'floor_cell':floor_cell,'anchor_cells':[floor_cell,floor_cell+1],
        'constraints':{'horizontal':'two selected roof-proxy footprints prescribed as object XY support',
          'base':'two voxel layers of footprint boundary only, at the existing assumed reference floor',
          'above_base':'retain only native-predicted occupancy inside selected object XY columns; heights/roofs/interiors unfilled',
          'outside':'foreground-only two-building generation scope, not an assertion that unannotated real map context is empty',
          'roads_free_air':'existing positive roads_known pixels above assumed ground; other map context stays unmeasured',
          'projection_times':'all 12 original Euler steps, one native decode/project/encode per step',
          'final':'raw final native decoder output, with NO final occupancy mask or mesh clipping'},
        'sampler':'v17 native 12-step CFG and time schedule unchanged; fixed dense noise and coordinate-indexed Shape noise',
        'new_training':False,'materials':False,'geometry_repair':False,'posthoc_output_fitting':False,
        'Shape':'original image-conditioned 1024 cascade and decoder; no texture models loaded',
        'GPU_cap_seconds':7200,'per_generation_limit_seconds':1500,'fixed_modes':['correct','edit'],
        'metric':'all-geometry support is a diagnostic only; building/roof labels must exclude base/ground/vegetation independently of target',
        'semantic_eval':'independent AI roof annotation on generated-only orthographic/oblique views, frozen before map scoring; uncertainty retained',
        'GO_requires':['two recognizable nondegenerate building masses in both outputs',
          'independently labeled roof-union IoU >=0.5 in both outputs in fixed world XY',
          'known-road above-ground obstruction <=0.05',
          'local footprint edit changes intended mass and does not visibly damage the other',
          'no major side-view sheets/fragmentation'],
        'STOP':'any required gate fails or building semantics cannot be resolved; no further trial or training',
        'limitations':'Map10 oblique screenshot roof proxies, not measured footprints or ground truth height; map supplied XY/base boundary, model supplied remaining support and fine surfaces; no generalization claim'}
    save(OUT/'PLAN.json',plan)
    evidence={
      'old_metric':{'source':str(JOB/'trellis2_mapguided_v18_overlay_fix1.py'),'rule':'any triangle vertex above fixed height; no semantic classes',
        'verdict':'NOT_BUILDING_ONLY','ground':'low horizontal ground may be excluded; crossing/elevated base may remain',
        'vegetation':'not excluded','historical_numbers':'preserved as height-filtered scene-projection diagnostics; not building footprint accuracy'},
      'native_interface':{'flow':str(REPO/'trellis2/models/sparse_structure_flow.py')+':224',
        'signature':'forward(x, t, cond)','pipeline':str(REPO/'trellis2/pipelines/trellis2_image_to_3d.py')+':193',
        'mask_argument':False,'encoder':str(SSENC),'encoder_mode':'deterministic mean, sample_posterior=False',
        'candidate_is_custom_sampler':'No claim that the native checkpoint is trained for structural inpainting'},
      'map_uncertainty':'Roof-proxy coordinates retained exactly; no generated-output registration or per-object movement'}
    save(OUT/'METRIC_INTERFACE_AUDIT.json',evidence)
    sourcefiles=[Path(__file__),JOB/'trellis2_map_anchor_v20_owner.py',JOB/'trellis2_map_anchor_v20_views.py',
      REPO/'trellis2/pipelines/trellis2_image_to_3d.py',REPO/'trellis2/models/sparse_structure_flow.py',
      REPO/'trellis2/models/sparse_structure_vae.py',REPO/'trellis2/pipelines/samplers/flow_euler.py',
      Path(str(SSENC)+'.json'),Path(str(SSENC)+'.safetensors'),Path(p['runtime'])/'pipeline.json',Path(p['input']),OUT/'PLAN.json']
    sourcefiles.extend(OUT/m/'MAP.npz' for m in ['correct','edit'])
    save(OUT/'SOURCE_FREEZE.json',{str(f):sha(f) for f in sourcefiles})
    image=Image.open(V11/'inputs/source_crop.png').convert('RGB');draw=ImageDraw.Draw(image)
    for i in chosen:
        xy=[(x-72,y-910) for x,y in meta['objects'][i]['outer_source_px']]
        draw.line(xy+[xy[0]],fill=(255,75,50),width=4);draw.text(xy[0],f'B{i+1}',fill='yellow')
    for r in meta['roads']:
        xy=[(x-72,y-910) for x,y in r['outer_source_px']];draw.line(xy+[xy[0]],fill=(70,220,255),width=3)
    image.save(OUT/'FIXED_MAP_REGION.png')
    print('V20_REGISTERED',chosen,SEED,flush=True)


def verify():
    for f,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)


def generate(mode):
    torch=setup();verify();p=read(OUT/'PLAN.json');folder=OUT/mode
    from trellis2 import models
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2_mapguided_v18_guidance import AddressedSparseNoise
    from scipy.ndimage import binary_erosion
    from PIL import Image,ImageDraw
    import trimesh
    cfg=read(Path(p['runtime'])/'pipeline.json')['args']
    cfgsam=cfg['sparse_structure_sampler']
    with np.load(folder/'MAP.npz') as z:fps=z['footprints'].copy();roads=z['roads'].copy()
    footprint=fps.any(0).T
    # A boundary ring conveys known XY, while all vertical shape remains predicted.
    ring=np.logical_or.reduce([x & ~binary_erosion(x,structure=np.ones((3,3))) for x in fps]).T
    allowed=torch.tensor(footprint,device='cuda')[None,None,:,:,None]
    anchors=torch.zeros((1,1,64,64,64),device='cuda',dtype=torch.bool)
    for zi in p['anchor_cells']:anchors[0,0,:,:,zi]=torch.tensor(ring,device='cuda')
    above=torch.arange(64,device='cuda')[None,None,None,None,:]>=p['floor_cell']
    road=torch.tensor(roads.T,device='cuda')[None,None,:,:,None]
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False).cuda()
          for k in ['sparse_structure_flow_model','sparse_structure_decoder']}
    encoder=models.from_pretrained(str(SSENC)).cuda().eval().requires_grad_(False)
    decoder=mods['sparse_structure_decoder']
    image_encoder=getattr(image_feature_extractor,cfg['image_cond_model']['name'])(**cfg['image_cond_model']['args'])
    pipe=Trellis2ImageTo3DPipeline(models=mods,image_cond_model=image_encoder,low_vram=True);pipe.cuda()
    with torch.no_grad():cond=pipe.get_cond([Image.open(p['input']).convert('RGB')],512)
    class AnchorSampler(FlowEulerGuidanceIntervalSampler):
        def __init__(self):super().__init__(**cfgsam['args']);self.trace=[]
        @torch.no_grad()
        def sample_once(self,model,x_t,t,t_prev,cond=None,**kwargs):
            out=super().sample_once(model,x_t,t,t_prev,cond,**kwargs)
            predicted=decoder(out.pred_x_0).float()>0
            constrained=(predicted & allowed & above) | anchors
            constrained &= ~(road & above)
            encoded=encoder(constrained.float(),sample_posterior=False).float()
            back=decoder(encoded).float()>0
            self.trace.append({'t':float(t),'native_predicted_cells':int(predicted.sum()),
              'projected_cells':int(constrained.sum()),'redecoded_cells':int(back.sum()),
              'redecoded_outside_cells':int((back & ~allowed & above).sum()),
              'native_retained_above_anchor_cells':int((predicted & constrained & ~anchors).sum()),
              'codec_IoU':float((back&constrained).sum()/(back|constrained).sum().clamp_min(1)),
              'anchor_added_cells':int((anchors&~predicted).sum()),
              'clean_latent_change_RMS':float((encoded-out.pred_x_0).square().mean().sqrt())})
            out.pred_x_0=encoded
            out.pred_x_prev=x_t-(t-t_prev)*self._xstart_to_pred(x_t,t,encoded)
            return out
    torch.manual_seed(SEED);noise=torch.randn((1,8,16,16,16)).cuda()
    sampler=AnchorSampler()
    with torch.no_grad():
        latent=sampler.sample(mods['sparse_structure_flow_model'],noise,**cond,**cfgsam['params'],verbose=False).samples
        support=(decoder(latent)>0)[0,0]
        coords=torch.argwhere(torch.nn.functional.max_pool3d(support[None,None].float(),2,2)>.5)[:,[0,2,3,4]].int()
    arr=support.cpu().numpy();ss=np.argwhere(arr)
    np.savez_compressed(folder/'SS.npz',latent=latent.cpu().numpy(),support64=arr,coords32=coords.cpu().numpy())
    proj=arr[:,:,p['floor_cell']+2:].any(2).T;union=fps.any(0)
    s={'mode':mode,'support_cells':int(arr.sum()),'coords32':len(coords),'native_extents':(np.ptp(ss,axis=0)+1).tolist() if len(ss) else [0,0,0],
      'all_support_projection_IoU_diagnostic':float((proj&union).sum()/max(1,(proj|union).sum())),
      'road_obstruction':float((proj&roads).sum()/max(1,roads.sum())),
      'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),'trace':sampler.trace,
      'final_support_unmasked':True,'model_training_updates':0,'materials':False}
    save(folder/'STRUCTURE.json',s)
    rgb=np.zeros((64,64,3),np.uint8)+20;rgb[proj]=[60,150,220];rgb[union]=[220,60,50];rgb[proj&union]=[230,210,80]
    Image.fromarray(rgb[::-1]).resize((512,512),Image.Resampling.NEAREST).save(folder/'STRUCTURE_OVERLAY.png')
    print('V20_STRUCTURE',mode,len(coords),s['all_support_projection_IoU_diagnostic'],flush=True)
    if not len(coords):
        save(folder/'EMPTY.json',{'stage':'Structure','no_retry_or_replacement':True});return
    del encoder,decoder,mods,pipe,sampler,latent,noise,support,anchors,allowed,above,road;gc.collect();torch.cuda.empty_cache()
    # Native Shape flow and decoder, with unmodified final Structure support.
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False)
          for k in ['shape_slat_flow_model_512','shape_slat_flow_model_1024','shape_slat_decoder']}
    pipe=Trellis2ImageTo3DPipeline(models=mods,image_cond_model=image_encoder,low_vram=True,
      shape_slat_sampler=AddressedSparseNoise(FlowEulerGuidanceIntervalSampler(**cfg['shape_slat_sampler']['args']),SEED+100000),
      shape_slat_sampler_params=cfg['shape_slat_sampler']['params'],shape_slat_normalization=cfg['shape_slat_normalization']);pipe.cuda()
    with torch.no_grad():
        cond1024=pipe.get_cond([Image.open(p['input']).convert('RGB')],1024)
        shape,resolution=pipe.sample_shape_slat_cascade(cond,cond1024,mods['shape_slat_flow_model_512'],mods['shape_slat_flow_model_1024'],32,1024,coords,max_num_tokens=49152)
        meshes,_=pipe.decode_shape_slat(shape,resolution)
    mesh=meshes[0];v=mesh.vertices.cpu().float().numpy();f=mesh.faces.cpu().numpy()
    np.savez_compressed(folder/'RAW_MESH.npz',vertices=v,faces=f)
    torch.save({'coords':shape.coords.cpu(),'feats':shape.feats.cpu(),'resolution':resolution},folder/'SHAPE_SLAT.pt')
    if not len(f):save(folder/'EMPTY.json',{'stage':'Shape','no_retry_or_replacement':True});return
    asset=folder/'geometry.glb'
    trimesh.Scene(trimesh.Trimesh(vertices=v@AXIS.T,faces=f,process=False)).export(asset)
    save(folder/'ASSET.json',{'asset':str(asset),'sha256':sha(asset),'vertices':len(v),'faces':len(f),
      'native_bounds':[v.min(0).tolist(),v.max(0).tolist()],'generated_height_roof_and_surfaces':True,
      'map_supplied':'two footprint boundary layers and allowed XY support during Structure sampling',
      'no_final_clipping_or_repair':True,'no_texture_generation':True,'native_to_glTF':AXIS.tolist(),
      'same_global_display_translation':p['world_z_translation'],'per_object_placement':False,
      'max_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
    print('V20_ASSET',mode,len(v),len(f),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--mode',choices=['correct','edit']);args=a.parse_args()
    prepare() if args.prepare else generate(args.mode)

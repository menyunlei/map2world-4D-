"""Resume saved v20 Structure with the native 512->1024 cascade contract."""
from trellis2_map_anchor_v20 import OUT,SEED,AXIS,setup,verify
from paths import JOB,read,save,sha
from pathlib import Path
import numpy as np,sys,hashlib

torch=setup();verify()
from trellis2 import models
from trellis2.modules import image_feature_extractor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2_mapguided_v18_guidance import AddressedSparseNoise
from PIL import Image
import trimesh

mode=sys.argv[sys.argv.index('--mode')+1];d=OUT/mode;p=read(OUT/'PLAN.json')
assert not (d/'geometry.glb').exists()
cfg=read(Path(p['runtime'])/'pipeline.json')['args']
with np.load(d/'SS.npz') as z:coords=torch.from_numpy(z['coords32'].copy()).int().cuda()
torch.manual_seed(SEED)
mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False)
    for k in ['shape_slat_flow_model_512','shape_slat_flow_model_1024','shape_slat_decoder']}
image_encoder=getattr(image_feature_extractor,cfg['image_cond_model']['name'])(**cfg['image_cond_model']['args'])
sampler=AddressedSparseNoise(FlowEulerGuidanceIntervalSampler(**cfg['shape_slat_sampler']['args']),SEED+100000)
pipe=Trellis2ImageTo3DPipeline(models=mods,image_cond_model=image_encoder,low_vram=True,
    shape_slat_sampler=sampler,shape_slat_sampler_params=cfg['shape_slat_sampler']['params'],
    shape_slat_normalization=cfg['shape_slat_normalization']);pipe.cuda()
with torch.no_grad():
    cond=pipe.get_cond([Image.open(p['input']).convert('RGB')],512)
    cond1024=pipe.get_cond([Image.open(p['input']).convert('RGB')],1024)
    shape,resolution=pipe.sample_shape_slat_cascade(cond,cond1024,mods['shape_slat_flow_model_512'],mods['shape_slat_flow_model_1024'],512,1024,coords,max_num_tokens=49152)
    meshes,_=pipe.decode_shape_slat(shape,resolution)
mesh=meshes[0];v=mesh.vertices.cpu().float().numpy();f=mesh.faces.cpu().numpy()
np.savez_compressed(d/'RAW_MESH.npz',vertices=v,faces=f)
torch.save({'coords':shape.coords.cpu(),'feats':shape.feats.cpu(),'resolution':resolution},d/'SHAPE_SLAT.pt')
save(d/'CASCADE_INTERFACE_FIX1.json',{'reason':'lr_resolution denotes mesh resolution512, not the sparse coord-grid resolution32; original wrong call expanded HR coordinates16x',
    'reused_SS_sha256':sha(d/'SS.npz'),'native_coords_unmodified':True,'new_Structure_sampling':False,
    'fixed_argument':{'before':32,'after':512},'official_source':'trellis2/pipelines/trellis2_image_to_3d.py:570',
    'noise_addressing_unchanged':True,'source_sha256':sha(Path(__file__))})
asset=d/'geometry.glb';trimesh.Scene(trimesh.Trimesh(vertices=v@AXIS.T,faces=f,process=False)).export(asset)
save(d/'ASSET.json',{'asset':str(asset),'sha256':sha(asset),'vertices':len(v),'faces':len(f),
    'native_bounds':[v.min(0).tolist(),v.max(0).tolist()],'generated_height_roof_and_surfaces':True,
    'map_supplied':'two footprint boundary layers and allowed XY support during Structure sampling',
    'no_final_clipping_or_repair':True,'no_texture_generation':True,'native_to_glTF':AXIS.tolist(),
    'same_global_display_translation':p['world_z_translation'],'per_object_placement':False,
    'max_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'cascade_interface_fix1':True})
print('V20_ASSET',mode,len(v),len(f),flush=True)

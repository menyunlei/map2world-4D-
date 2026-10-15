"""Real native dependencies and DINO feature extraction, on one leased A800."""
from trellis2_asset_trial_v3 import DEST, REPO, OUT, DINO, save
import os
import sys
import time
os.environ['CUDA_HOME']=str(DEST/'environment')
os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
sys.path.insert(0,str(REPO))
start=time.time()
import torch, flash_attn, cumesh, flex_gemm, o_voxel, nvdiffrast.torch as dr
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
from PIL import Image
model=DinoV3FeatureExtractor(str(DINO));model.cuda()
features=model([Image.open(OUT/'inputs/map10_block/input.png')])
assert torch.isfinite(features).all()
context=dr.RasterizeCudaContext()
vertices=torch.tensor([[[-.5,-.5,0,1],[.5,-.5,0,1],[0,.5,0,1]]],device='cuda')
faces=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
pixels,_=dr.rasterize(context,vertices,faces,resolution=[32,32])
assert torch.isfinite(pixels).all() and (pixels[...,3]>0).any()
save(OUT/'GPU_PREFLIGHT.json',{'device':torch.cuda.get_device_name(),'visible_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
     'features_shape':list(features.shape),'all_features_finite':True,'native_rasterization_pass':True,
     'seconds':time.time()-start,'training_updates':0})
print('REAL_GPU_PREFLIGHT_PASSED',flush=True)

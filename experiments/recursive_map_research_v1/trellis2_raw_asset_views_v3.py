"""Fresh native PBR rendering of already generated saved assets; no model sampling."""
from trellis2_asset_trial_v3 import OUT, DEST, REPO, SEEDS, save, sha
import os
import sys
import math
os.environ['CUDA_HOME']=str(DEST/'environment')
os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
sys.path.insert(0,str(REPO))
import torch
import numpy as np
import cv2
import utils3d
from PIL import Image, ImageDraw
from trellis2.representations import MeshWithVoxel
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

directions=[('front',[0,-1,.05]),('back',[0,1,.05]),('left',[-1,0,.05]),('right',[1,0,.05]),
            ('top',[.001,0,1]),('oblique',[1,-1,.75]),('closeup',[1,-1,.75]),
            ('lateral_left',[-.55,-1,.25]),('lateral_right',[.55,-1,.25])]
image=cv2.cvtColor(cv2.imread(str(REPO/'assets/hdri/studio.exr'),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
envmap=EnvMap(torch.tensor(image,dtype=torch.float32,device='cuda'))
all_records=[]
for seed in SEEDS:
    root=OUT/f'seed{seed}'/'official_building'
    raw=root/'RAW_NATIVE.pt'
    data=torch.load(raw,map_location='cpu',weights_only=False)
    resolution=int(data['resolution'])
    spatial=(data['coords'].max(0).values+1).tolist()
    mesh=MeshWithVoxel(data['vertices'].cuda(),data['faces'].cuda(),[-.5,-.5,-.5],
        data['voxel_size'],data['coords'].cuda(),data['attrs'].cuda(),
        torch.Size([1,data['attrs'].shape[1],*spatial]),data['layout'])
    target=root/'native_views';target.mkdir(exist_ok=False)
    renderer=render_utils.get_renderer(mesh,resolution=768,near=.005,far=10,ssaa=1)
    records=[]
    for name,direction in directions:
        radius=1.6 if name=='closeup' else 2.6
        eye=torch.tensor(direction,dtype=torch.float32,device='cuda');eye=eye/eye.norm()*radius
        extr=utils3d.torch.extrinsics_look_at(eye,torch.zeros(3,device='cuda'),torch.tensor([0,0,1],dtype=torch.float32,device='cuda'))
        intr=utils3d.torch.intrinsics_from_fov_xy(torch.tensor(math.pi/4,device='cuda'),torch.tensor(math.pi/4,device='cuda'))
        with torch.no_grad():result=renderer.render(mesh,extr,intr,envmap=envmap)
        for key in ('shaded','base_color','normal','alpha'):
            tensor=result[key].detach().float().cpu()
            if tensor.ndim==2:tensor=tensor[None].expand(3,-1,-1)
            array=np.clip(tensor.numpy().transpose(1,2,0)*255,0,255).astype('uint8')
            Image.fromarray(array).save(target/f'{name}_{key}.png')
        records.append({'view':name,'eye':eye.tolist(),'look':[0,0,0],'fov':45})
        print('NATIVE_SAVED_ASSET_VIEW',seed,name,flush=True)
    contact=Image.new('RGB',(3*384,3*410),'#202020');draw=ImageDraw.Draw(contact)
    for index,(name,_) in enumerate(directions):
        im=Image.open(target/f'{name}_shaded.png');im.thumbnail((384,384))
        x,y=(index%3)*384,(index//3)*410;contact.paste(im,(x,y));draw.text((x+5,y+386),name,fill='white')
    contact.save(target/'contact.jpg',quality=92)
    save(target/'RELOAD.json',{'seed':seed,'RAW_NATIVE_sha256':sha(raw),'native_faces':len(mesh.faces),
         'native_vertices':len(mesh.vertices),'voxel_shape':list(mesh.voxel_shape),'resolution':resolution,
         'shader':'official PbrMeshRenderer','HDRI':str(REPO/'assets/hdri/studio.exr'),
         'fresh_process':True,'geometry_or_material_modified':False,'camera_records':records})
    all_records.append({'seed':seed,'contact':str(target/'contact.jpg')})
    del mesh,data;torch.cuda.empty_cache()
save(OUT/'NATIVE_RELOAD_COMPLETE.json',{'cases':all_records,'no_model_sampling':True,'training_updates':0})

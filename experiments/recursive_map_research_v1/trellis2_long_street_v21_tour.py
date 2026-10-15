"""Actual saved-scene street camera translation and look-back; no model inference."""
from trellis2_long_street_v21 import OUT
from paths import read,save,sha
from trellis2_map_edit_mesh_v9_views import camera_matrices,MeshRenderer
from trellis2.representations import MeshWithVoxel
import torch,numpy as np,math
from PIL import Image,ImageDraw
import imageio.v2 as imageio

torch.set_num_threads(4)
route=read(OUT/'STREET_ROUTE.json');scale=read(OUT/'ASSUMED_SCALE.json')['global_native_to_meter_scale']
a=read(OUT/'ASSET.json');assert sha(a['asset'])==a['sha256']
data=torch.load(OUT/'RAW_NATIVE.pt',map_location='cpu',weights_only=False)
spatial=(data['coords'].max(0).values+1).tolist()
mesh=MeshWithVoxel(data['vertices'].cuda(),data['faces'].cuda(),[-.5,-.5,-.5],data['voxel_size'],
    data['coords'].cuda(),data['attrs'].cuda(),torch.Size([1,data['attrs'].shape[1],*spatial]),data['layout'])
r=MeshRenderer();r.rendering_options.resolution=768;r.rendering_options.ssaa=1
r.rendering_options.near=.001;r.rendering_options.far=10;r.rendering_options.antialias=False;r.rendering_options.chunk_size=8000000
start=np.array(route['start_native_xy']);end=np.array(route['end_native_xy']);ground=route['ground_native_z']
forward=(end-start)/np.linalg.norm(end-start);eye_z=ground+1.6/scale
points=[];mini=Image.open(OUT/'ORTHO_TOP.png').convert('RGB').resize((176,176))
vd=OUT/'tour_views';vd.mkdir()
with imageio.get_writer(str(OUT/'street_20s_24fps.mp4'),fps=24,codec='libx264',quality=8) as writer:
    for i in range(480):
        if i<384:xy=start+(end-start)*(i/383);direction=forward;phase='forward'
        elif i<432:
            xy=end;theta=math.pi*(i-384)/47
            direction=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])@forward;phase='look back'
        else:xy=end-forward*(10/scale)*((i-432)/47);direction=-forward;phase='return 10m'
        eye=[*xy,eye_z];look=[*(xy+direction*.25),eye_z]
        K,w=camera_matrices(eye,look,[0,0,1],70)
        with torch.no_grad():z=r.render(mesh,torch.tensor(w,dtype=torch.float32,device='cuda'),torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['attr','mask','depth'])
        rgb=z['base_color'].permute(1,2,0).cpu().numpy();mask=z['mask'].cpu().numpy()
        rgb=(rgb+.16*(1-mask[...,None])).clip(0,1)
        # Center crop, never stretch: 768-square native render -> 768x512 viewport.
        im=Image.fromarray((rgb*255).astype(np.uint8)).crop((0,128,768,640));draw=ImageDraw.Draw(im)
        mp=mini.copy();md=ImageDraw.Draw(mp)
        def mxy(v):return ((v[0]+.5)*176,(.5-v[1])*176)
        md.line([mxy(start),mxy(end)],fill='#f4c45e',width=2)
        px,py=mxy(xy);md.ellipse((px-3,py-3,px+3,py+3),fill='#ff4c42')
        md.line([(px,py),mxy(xy+direction*.045)],fill='#ff4c42',width=2)
        im.paste(mp,(580,12));draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,568,32),fill='#22262c');draw.text((9,10),f'Saved native 3D | {phase} | eye 1.6m (assumed scale)',fill='white')
        draw.rectangle((570,190,768,214),fill='#22262c');draw.text((575,197),'asset top view, NOT input map',fill='white')
        writer.append_data(np.asarray(im))
        if i in [0,96,192,288,383,431,479]:im.save(vd/f'{i:04d}.png')
        center_depth=z['depth'][384,384].item() if z['depth'].ndim==2 else float('nan')
        points.append({'frame':i,'eye':list(map(float,eye)),'look':list(map(float,look)),
            'center_ray_depth_native':center_depth if np.isfinite(center_depth) else None})
        if i%96==0:print('V21_TOUR_FRAME',i,flush=True)
sheet=Image.new('RGB',(3*384,2*280),'#252525');draw=ImageDraw.Draw(sheet)
for j,i in enumerate([0,96,192,288,431,479]):
    sheet.paste(Image.open(vd/f'{i:04d}.png').resize((384,256)),((j%3)*384,(j//3)*280+24))
    draw.text(((j%3)*384+5,(j//3)*280+5),f'{i/24:.1f}s',fill='white')
sheet.save(OUT/'street_contact.jpg',quality=95)
save(OUT/'TOUR.json',{'asset_sha256':a['sha256'],'native_PBR_sha256':sha(OUT/'RAW_NATIVE.pt'),
    'frames':480,'fps':24,'duration_seconds':20,'forward_distance_assumed_m':float(np.linalg.norm(end-start)*scale),
    'return_distance_assumed_m':10,'eye_height_assumed_m':1.6,'not_map_tracking':True,
    'no_model_generation_or_geometry_edit':True,'route':route,'poses':points})
print('V21_STREET_TOUR_COMPLETE',flush=True)

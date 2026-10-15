"""Reload native geometry; fixed cameras and target-free orthographic roof evidence."""
from trellis2_map_anchor_v20 import OUT,AXIS,setup
from paths import read,save,sha
import sys,numpy as np
torch=setup()
from trellis2.representations import Mesh
from trellis2.renderers import MeshRenderer
from trellis2_map_entry_views_v2 import camera_matrices
import nvdiffrast.torch as dr
import trimesh
from PIL import Image,ImageDraw
mode=sys.argv[sys.argv.index('--mode')+1];folder=OUT/mode
if (folder/'EMPTY.json').exists():print('EMPTY_ASSET_PRESERVED',mode);sys.exit(0)
p=read(OUT/'PLAN.json');a=read(folder/'ASSET.json');assert sha(a['asset'])==a['sha256']
with np.load(folder/'RAW_MESH.npz') as z:v=z['vertices'].copy();f=z['faces'].copy()
s=trimesh.load(a['asset'],force='scene',process=False);g=list(s.geometry.values())[0]
assert np.array_equal(g.faces,f) and np.allclose(g.vertices,v@AXIS.T,atol=1e-7)
del s,g
mesh=Mesh(torch.from_numpy(v).cuda(),torch.from_numpy(f.astype(np.int32)).cuda())
renderer=MeshRenderer();renderer.rendering_options.resolution=512;renderer.rendering_options.ssaa=1
renderer.rendering_options.near=.001;renderer.rendering_options.far=10
renderer.rendering_options.antialias=False;renderer.rendering_options.chunk_size=8000000
# Cameras are derived from the original map region, never recentered on output.
center=[.04,-.16,-.21];floor=p['floor_native_reference'];eye_z=floor+.045
views=[('top',[.04,-.16,1.0],center,[0,1,0],45),
 ('oblique',[.85,-1.05,.5],center,[0,0,1],45),
 ('front',[.04,-.85,eye_z],[.04,-.05,eye_z],[0,0,1],65),
 ('side',[.65,-.16,eye_z],[-.15,-.16,eye_z],[0,0,1],65),
 ('back',[.04,.65,-.15],center,[0,0,1],55)]
vd=folder/'views';vd.mkdir()
def render(eye,look,up,fov):
    K,w=camera_matrices(eye,look,up,fov)
    with torch.no_grad():r=renderer.render(mesh,torch.tensor(w,device='cuda',dtype=torch.float32),torch.tensor(K,device='cuda',dtype=torch.float32),return_types=['normal','depth','mask'])
    mask=r['mask'].cpu().numpy();n=r['normal'].permute(1,2,0).cpu().numpy()*2-1
    light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
    shade=.28+.72*np.abs((n*light).sum(-1)).clip(0,1)
    rgb=np.array([.76,.79,.82])*shade[...,None]*mask[...,None]+.08*(1-mask[...,None])
    return Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8))
for name,eye,look,up,fov in views:render(eye,look,up,fov).save(vd/f'{name}.png')
import imageio.v2 as imageio
with imageio.get_writer(str(folder/'lateral_2s_24fps.mp4'),fps=24,codec='libx264',quality=7) as writer:
    for i,x in enumerate(np.linspace(-.20,.30,48)):
        im=render([float(x),-.85,eye_z],[float(x),-.05,eye_z],[0,0,1],65)
        writer.append_data(np.asarray(im))
        if i in [0,24,47]:im.save(vd/f'lateral_{i:02d}.png')
# Fixed native XY orthographic raster; record all faces and topmost height.
verts=mesh.vertices.float();faces=mesh.faces.int()
clip=torch.cat([verts[:,:2]*2,-verts[:,2:3],torch.ones_like(verts[:,:1])],-1)[None].contiguous()
context=dr.RasterizeCudaContext();height=torch.full((512,512),-2.,device='cuda');occupied=torch.zeros((512,512),device='cuda',dtype=torch.bool)
for start in range(0,len(faces),8000000):
    fc=faces[start:start+8000000].contiguous()
    with torch.no_grad():
        rast,_=dr.rasterize(context,clip,fc,resolution=[512,512]);z,_=dr.interpolate(verts[:,2:3][None].contiguous(),rast,fc)
    hit=rast[0,:,:,3]>0;height=torch.maximum(height,torch.where(hit,z[0,:,:,0],-2.));occupied|=hit
ht=height.cpu().numpy();occ=occupied.cpu().numpy()
np.savez_compressed(folder/'ORTHO_GEOMETRY.npz',height=ht,occupied=occ)
# Height visualization is target-free and semantically UNLABELED.
gray=(.2+.75*((ht+.5)/.8).clip(0,1))*occ
rgb=np.repeat(gray[...,None],3,-1)+.08*(~occ)[...,None]
Image.fromarray((rgb[::-1].clip(0,1)*255).astype(np.uint8)).save(folder/'ORTHO_UNLABELED.png')
sheet=Image.new('RGB',(3*320,2*344),'#151922');draw=ImageDraw.Draw(sheet)
for i,name in enumerate(['top','oblique','front','side','back','lateral_47']):
    sheet.paste(Image.open(vd/f'{name}.png').resize((320,320)),((i%3)*320,(i//3)*344+24));draw.text(((i%3)*320+4,(i//3)*344+4),name,fill='white')
sheet.save(folder/'CONTACT.jpg',quality=94)
with np.load(folder/'MAP.npz') as z:fp=z['footprints'].any(0);roads=z['roads']
target=np.repeat(np.repeat(fp,8,0),8,1);road=np.repeat(np.repeat(roads,8,0),8,1)
high=occ&(ht>floor+2/64)
save(folder/'RELOAD.json',{'asset_sha256':a['sha256'],'new_process_exact_geometry':True,'views':views,'no_output_registration':True,
  'all_geometry_above_floor_IoU_DIAGNOSTIC_ONLY':float((high&target).sum()/max(1,(high|target).sum())),
  'known_road_obstruction':float((high&road).sum()/max(1,road.sum())),
  'building_only_IoU':None,'reason':'No automatic height threshold is claimed to identify buildings. Await target-blind semantic labels.',
  'neutral_geometry_no_materials':True,'lateral_frames':48,'fps':24})
print('V20_SAVED_ASSET_VIEWS_COMPLETE',mode,flush=True)

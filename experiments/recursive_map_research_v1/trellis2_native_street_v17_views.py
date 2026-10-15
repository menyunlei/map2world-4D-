"""Fresh-process rendering of saved native geometry and appearance, no resampling."""
from trellis2_native_street_v17 import OUT
from paths import read,save,sha
from trellis2_map_edit_mesh_v9_views import camera_matrices,MeshRenderer
from trellis2.representations import MeshWithVoxel
import torch,numpy as np,trimesh
from PIL import Image,ImageDraw

torch.set_num_threads(4)
meta=read(OUT/'ASSET.json');assert sha(meta['asset'])==meta['sha256']
data=torch.load(OUT/'RAW_NATIVE.pt',map_location='cpu',weights_only=False)
with np.load(OUT/'RAW_MESH.npz') as z:
    assert np.array_equal(z['faces'],data['faces'].numpy()) and np.allclose(z['vertices'],data['vertices'].numpy())
scene=trimesh.load(meta['asset'],force='scene',process=False);g=list(scene.geometry.values())[0]
assert len(g.faces)==len(data['faces']) and np.array_equal(g.faces,data['faces'].numpy())
axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
assert np.allclose(g.vertices,data['vertices'].numpy()@axis.T,atol=1e-7)
del g,scene
spatial=(data['coords'].max(0).values+1).tolist()
mesh=MeshWithVoxel(data['vertices'].cuda(),data['faces'].cuda(),[-.5,-.5,-.5],data['voxel_size'],
    data['coords'].cuda(),data['attrs'].cuda(),torch.Size([1,data['attrs'].shape[1],*spatial]),data['layout'])
renderer=MeshRenderer();renderer.rendering_options.resolution=512;renderer.rendering_options.ssaa=1
renderer.rendering_options.near=.001;renderer.rendering_options.far=10
renderer.rendering_options.antialias=False;renderer.rendering_options.chunk_size=8000000
vd=OUT/'views';vd.mkdir(exist_ok=False)
for name in ['appearance','neutral','frames']:(vd/name).mkdir()
low,high=data['vertices'][:,2].min().item(),data['vertices'][:,2].max().item()
street_z=low+.08*(high-low)
views=[('front',[0,-1.65,.05],[0,0,-.05],[0,0,1],45),
       ('back',[0,1.65,.05],[0,0,-.05],[0,0,1],45),
       ('left',[-1.65,0,.05],[0,0,-.05],[0,0,1],45),
       ('right',[1.65,0,.05],[0,0,-.05],[0,0,1],45),
       ('top',[0,0,1.65],[0,0,-.05],[0,1,0],45),
       ('oblique',[1.1,-1.3,.9],[0,0,-.05],[0,0,1],45),
       ('closeup',[.65,-.85,.35],[0,0,-.12],[0,0,1],45),
       ('street_front',[0,-1.1,street_z],[0,0,street_z],[0,0,1],55),
       ('street_side',[1.1,0,street_z],[0,0,street_z],[0,0,1],55)]

def render(eye,look,up,fov):
    K,w2c=camera_matrices(eye,look,up,fov)
    with torch.no_grad():p=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),
        torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['attr','normal','mask','depth'])
    mask=p['mask'].cpu().numpy();attr=p['attr'].permute(1,2,0).cpu().numpy()
    rgb=attr[...,data['layout']['base_color']]+.16*(1-mask[...,None])
    n=p['normal'].permute(1,2,0).cpu().numpy()*2-1;light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
    gray=(.3+.7*np.abs((n*light).sum(-1)).clip(0,1))*mask
    neutral=np.array([.76,.79,.82])*gray[...,None]+.16*(1-mask[...,None])
    return Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)),Image.fromarray((neutral.clip(0,1)*255).astype(np.uint8)),p

for name,eye,look,up,fov in views:
    image,neutral,p=render(eye,look,up,fov);image.save(vd/'appearance'/f'{name}.png');neutral.save(vd/'neutral'/f'{name}.png')
    if name in ['street_front','top']:np.savez_compressed(vd/f'{name}_depth.npz',depth=p['depth'].cpu().numpy(),mask=p['mask'].cpu().numpy())
    print('NATIVE_SAVED_STREET_VIEW',name,flush=True)
import imageio.v2 as imageio
poses=[]
with imageio.get_writer(str(OUT/'lateral_4s_24fps.mp4'),fps=24,codec='libx264',quality=7) as writer:
    for i,x in enumerate(np.linspace(-.25,.25,96)):
        eye=[float(x),-1.1,street_z];look=[float(x),0,street_z]
        image,_,_=render(eye,look,[0,0,1],55);image.save(vd/'frames'/f'{i:04d}.png')
        writer.append_data(np.asarray(image));poses.append({'eye':eye,'look':look,'fov':55})
for mode in ['appearance','neutral']:
    sheet=Image.new('RGB',(3*320,3*344),'#252525');draw=ImageDraw.Draw(sheet)
    for i,(name,*_) in enumerate(views):
        sheet.paste(Image.open(vd/mode/f'{name}.png').resize((320,320)),((i%3)*320,(i//3)*344+24))
        draw.text(((i%3)*320+6,(i//3)*344+4),name,fill='white')
    sheet.save(OUT/f'{mode}_contact.jpg',quality=94)
save(OUT/'RELOAD.json',{'asset_sha256':meta['sha256'],'native_PBR_sha256':sha(OUT/'RAW_NATIVE.pt'),
    'fresh_process':True,'GLB_faces_and_vertices_checked_against_native':True,
    'renderer':'Original TRELLIS.2 MeshRenderer, chunked, native base-color and separate neutral normals',
    'not_full_PBR_light_render':True,'no_geometry_modified':True,'views':views,
    'street_height_rule':'native min Z + 0.08 * scene height; illustrative asset-scale viewpoint, not measured eye height',
    'video_frames':96,'fps':24,'duration_seconds':4,'lateral_poses':poses})
print('NATIVE_STREET_VIEWS_COMPLETE',flush=True)

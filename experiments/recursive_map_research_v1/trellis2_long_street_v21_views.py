"""Reuse fixed v17 reload views; add native-coordinate orthophoto for route inspection."""
from pathlib import Path
source=Path(__file__).with_name('trellis2_native_street_v17_views.py')
code=source.read_text().replace('from trellis2_native_street_v17 import OUT','from trellis2_long_street_v21 import OUT')
code=code.replace("attr=p['attr'].permute(1,2,0).cpu().numpy()", "attr=p['base_color'].permute(1,2,0).cpu().numpy()")
code=code.replace("rgb=attr[...,data['layout']['base_color']]","rgb=attr")
exec(compile(code,str(source),'exec'))

import nvdiffrast.torch as dr
verts=mesh.vertices.float();faces=mesh.faces.int()
clip=torch.cat([verts[:,:2]*2,-verts[:,2:3],torch.ones_like(verts[:,:1])],-1)[None].contiguous()
ctx=dr.RasterizeCudaContext();height=torch.full((512,512),-2.,device='cuda');occupied=torch.zeros((512,512),device='cuda',dtype=torch.bool)
for start in range(0,len(faces),8000000):
    fc=faces[start:start+8000000].contiguous()
    with torch.no_grad():rast,_=dr.rasterize(ctx,clip,fc,resolution=[512,512]);z,_=dr.interpolate(verts[:,2:3][None].contiguous(),rast,fc)
    hit=rast[0,:,:,3]>0;height=torch.maximum(height,torch.where(hit,z[0,:,:,0],-2.));occupied|=hit
y,x=torch.meshgrid((torch.arange(512,device='cuda')+.5)/512-.5,(torch.arange(512,device='cuda')+.5)/512-.5,indexing='ij')
pos=torch.stack([x,y,height],-1)
with torch.no_grad():attrs=mesh.query_attrs(pos[occupied]);colors=attrs[:,data['layout']['base_color']].clamp(0,1)
rgb=torch.ones((512,512,3),device='cuda')*.16;rgb[occupied]=colors
Image.fromarray((rgb.cpu().numpy()[::-1]*255).astype(np.uint8)).save(OUT/'ORTHO_TOP.png')
np.savez_compressed(OUT/'ORTHO_WORLD.npz',height=height.cpu().numpy(),occupied=occupied.cpu().numpy())
print('V21_ORTHOPHOTO_SAVED',flush=True)

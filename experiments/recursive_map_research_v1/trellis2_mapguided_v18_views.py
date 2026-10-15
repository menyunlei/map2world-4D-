"""Same fixed cameras, fresh-process asset verification and actual mesh overlay."""
from pathlib import Path
import sys
from trellis2_mapguided_v18 import OUT
mode=sys.argv[sys.argv.index('--mode')+1]
assert mode in ['correct','edit']
source=Path(__file__).with_name('trellis2_native_street_v17_views.py')
code=source.read_text().replace('from trellis2_native_street_v17 import OUT','')
code=code.replace("attr=p['attr'].permute(1,2,0).cpu().numpy()", "attr=p['base_color'].permute(1,2,0).cpu().numpy()")
code=code.replace("rgb=attr[...,data['layout']['base_color']]", "rgb=attr")
code=code.replace("low,high=data['vertices'][:,2].min().item(),data['vertices'][:,2].max().item()",
    'low,high=-0.3383564352989197,0.3336969017982483')
code=code.replace("'native min Z + 0.08 * scene height; illustrative asset-scale viewpoint, not measured eye height'",
    "'Fixed v17 reference floor/height for both outputs; not recentered on each result; not measured eye height'")
scope={'__name__':'v18_views_reuse','__file__':str(source),'OUT':OUT/mode}
exec(compile(code,str(source),'exec'),scope)

# Orthographic projection uses every saved triangle above the fixed floor band.
# This is a diagnostic only, never used to modify the asset.
import torch,numpy as np,nvdiffrast.torch as dr
from PIL import Image,ImageDraw
from paths import read,save
p=read(OUT/'PLAN.json');d=scope['data'];v=d['vertices'].cuda().float();f=d['faces'].cuda().int()
zstart=p['guidance']['building_z_start']/64-.5
keep=(v[f.long(),2].amax(-1)>=zstart);faces=f[keep]
clip=torch.cat([v[:,:2]*2,-v[:,2:3]*.5,torch.ones_like(v[:,:1])],-1)[None]
context=dr.RasterizeCudaContext()
with torch.no_grad():rast,_=dr.rasterize(context,clip,faces,resolution=[512,512])
projection=(rast[0,:,:,3]>0).cpu().numpy()
with np.load(p['maps'][mode]['path']) as z:fps=z['footprints'].copy()>.5
union=fps.any(0);target=np.repeat(np.repeat(union,8,0),8,1)
rgb=np.zeros((512,512,3),np.uint8)+20
rgb[projection]=[55,155,230];rgb[target]=[230,65,55];rgb[projection&target]=[235,215,90]
image=Image.fromarray(rgb[::-1].copy());image.save(OUT/mode/'actual_mesh_map_overlay.png')
save(OUT/mode/'MESH_MAP.json',{'map_kind':'Approximate manually annotated roof proxies, not surveyed GT',
    'projection':'All saved triangles above fixed reference floor band, orthographic XY; no per-object placement',
    'above_floor_projection_IoU':float((projection&target).sum()/max(1,(projection|target).sum())),
    'per_footprint_coverage':[float((projection&np.repeat(np.repeat(fp,8,0),8,1)).sum()/max(1,fp.sum()*64)) for fp in fps],
    'outside_fraction':float((projection&~target).sum()/max(1,projection.sum())),
    'not_a_geometry_or_visual_quality_pass':True})

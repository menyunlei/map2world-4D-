"""Freeze one visibly identified generated road centerline, not an input-map route."""
from paths import read,save,sha
from trellis2_long_street_v21 import OUT
import numpy as np
from PIL import Image,ImageDraw

# Chosen once from the saved target-free top view: road between x≈-.05 and+.04.
# Entire central segment retained, not selected frame-by-frame to conceal defects.
start=np.array([-.007,-.45]);end=np.array([-.007,.45]);t=np.linspace(0,1,257)
points=start[None]+t[:,None]*(end-start)[None]
with np.load(OUT/'ORTHO_WORLD.npz') as z:h=z['height'];occ=z['occupied']
ix=np.clip(((points[:,0]+.5)*512).astype(int),0,511);iy=np.clip(((points[:,1]+.5)*512).astype(int),0,511)
valid=occ[iy,ix];heights=h[iy,ix];scale=read(OUT/'ASSUMED_SCALE.json')['global_native_to_meter_scale']
ground=float(np.median(heights[valid]));eye=ground+1.6/scale
profile={'valid_ground_ray_fraction':float(valid.mean()),'ground_height_native_quantiles':np.quantile(heights[valid],[0,.1,.5,.9,1]).tolist(),
    'top_surface_above_eye_fraction':float((heights[valid]>eye).mean()),
    'ground_variation_m_q10_q90':float(np.quantile(heights[valid],.9)-np.quantile(heights[valid],.1))*scale,
    'not_complete_collision_certification':True}
route={'start_native_xy':start.tolist(),'end_native_xy':end.tolist(),'ground_native_z':ground,
    'assumed_meter_scale':scale,'forward_distance_assumed_m':float(np.linalg.norm(end-start)*scale),
    'eye_height_assumed_m':1.6,'frames':480,'fps':24,'route_rule':'16s forward along entire marked road-center segment;2s look-back;2s reverse10m',
    'selection_basis':'Actual generated ORTHO_TOP, not a target map; fixed once before first street render',
    'geometry_changes':False,'input_map_control':False,'ground_profile':profile,
    'source_top_sha256':sha(OUT/'ORTHO_TOP.png'),'source_height_sha256':sha(OUT/'ORTHO_WORLD.npz')}
save(OUT/'STREET_ROUTE.json',route)
im=Image.open(OUT/'ORTHO_TOP.png').convert('RGB');d=ImageDraw.Draw(im)
px=[((x+.5)*512,(.5-y)*512) for x,y in [start,end]];d.line(px,fill='yellow',width=3)
d.text((px[0][0]+6,px[0][1]-15),'START',fill='yellow');d.text((px[1][0]+6,px[1][1]+5),'END',fill='yellow')
im.save(OUT/'STREET_ROUTE_TOP.png');print(route,flush=True)

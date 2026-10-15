"""Actual saved-mesh depth measurements, never mask or repair output geometry."""
from pathlib import Path
import json,hashlib,numpy as np
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_boundary_online_v24';NEW=OUT/'chunks/road-boundary-0--1'
OLD=JOB/'map_object_world_v1/trellis2_online_v23/chunks/online-0--1';PARENT=JOB/'map_object_world_v1/trellis2_long_street_v21'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=read(NEW/'PLAN.json');b=p['boundary'];s=b['fixed_world_frame']['global_native_to_meter_scale'];floor=b['fixed_world_frame']['native_floor_for_display'];tz=p['fixed_chunk_translation_GLTF'][2]
with np.load(NEW/'FINAL_HEIGHTFIELD.npz') as a:nh=a['height'];no=a['occupied']
with np.load(PARENT/'ORTHO_WORLD.npz') as a:ph=a['height'];po=a['occupied']
def sample(h,o,x,z,offset):
    nx=np.asarray(x)/s;ny=(offset-np.asarray(z))/s
    valid=(nx>=-.5)&(nx<.5)&(ny>=-.5)&(ny<.5)
    ix=np.clip(((nx+.5)*512).astype(int),0,511);iy=np.clip(((ny+.5)*512).astype(int),0,511)
    hit=valid&o[iy,ix];height=h[iy,ix]*s-floor*s
    return hit,np.where(hit,height,np.nan)
ground=(b['ground_native_z']-floor)*s
xs=np.linspace(-.007*s-2,-.007*s+2,21);zs=np.linspace(-s/2-.2,-s*1.5+.2,501)
x,z=np.meshgrid(xs,zs);hit,height=sample(nh,no,x,z,tz)
near=np.abs(height-ground)<=.5
front_hit,front_y=sample(ph,po,xs,np.full_like(xs,-s/2+.25),0)
back_hit,back_y=sample(nh,no,xs,np.full_like(xs,-s/2-.25),tz)
both=front_hit&back_hit;step=np.abs(front_y-back_y)[both]
with np.load(NEW/'FORWARD_VISIBILITY.npz') as a:blocked=a['occupied']
ix=np.clip(((xs/s+.5)*512).astype(int),0,511)
visibility={}
for name,h in [('low',ground+.6),('body',ground+1.6)]:
    iy=int(h/5*512);rows=np.arange(max(0,iy-2),min(512,iy+3));bad=blocked[rows[:,None],ix[None,:]].any(0)
    visibility[name]={'camera_world_height':h,'blocked_ray_fraction':float(bad.mean()),'clear_ray_fraction':float((~bad).mean())}
new_bounds=read(NEW/'RELOAD.json')['world_bounds'];old_bounds=read(OLD/'RELOAD.json')['world_bounds']
parent_zmin=-49.9922
new_gap=max(0,parent_zmin-(new_bounds[1][2]+tz));old_gap=max(0,parent_zmin-(old_bounds[1][2]-100))
old_zz=np.linspace(-s/2+5,-s/2,1001);new_zz=np.linspace(-s/2,-s/2-5,1001)
old_hits,old_hh=sample(ph,po,np.full_like(old_zz,-.007*s),old_zz,0)
new_hits,new_hh=sample(nh,no,np.full_like(new_zz,-.007*s),new_zz,tz)
old_valid=old_hits&(np.abs(old_hh-ground)<=.5);new_valid=new_hits&(np.abs(new_hh-ground)<=.5)
road_gap=float(old_zz[old_valid].min()-new_zz[new_valid].max()) if old_valid.any() and new_valid.any() else None
with np.load(NEW/'GENERATED_STRUCTURE.npz') as a:c=a['coords']
road=c[(np.abs((c[:,1]+.5)/32-.5+.007)<.05)&(np.abs((c[:,3]+.5)/32-.5-b['ground_native_z'])<.06)]
coarse_y=sorted(set(int(x) for x in road[:,2]))
trace=read(NEW/'BOUNDARY_TRACE.json');assert len(trace)==12 and all(x['outside_mask_change_max']==0 for x in trace)
hashes={'parent':sha(PARENT/'large_street_100m.glb'),'v23':sha(OLD/'chunk_100m.glb'),'candidate':sha(NEW/'chunk_100m.glb')}
assert hashes['parent']=='b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe'
assert hashes['v23']=='373c81c0de7f2bc1e0bd3c85e858d30980f2224f2c2ff8f887eab16432771cb7'
result={'decision':'STOP_SINGLE_TEST_JOINT_CONTINUITY_NOT_ESTABLISHED','same_seed':942219,'one_generation':True,'model_training':False,
 'mechanism':'2.34375% local noisy-latent road observation replacement; all unmasked Euler outputs remain exact native results; fixed parent world frame before inference',
 'mechanism_interpretation':'Outside-mask replacement delta0 means no direct clamp outside the known region, not identity to the unconditioned baseline: native attention may propagate the observations. The local condition and fixed export frame jointly implement one world-address contract; this single test does not separately identify their contributions.',
 'previous_failed_v23_preserved':True,'hashes':hashes,'codec_preflight':read(NEW/'CODEC_PREFLIGHT.json'),
 'native_structure_diagnostic':{'coordinate_count':len(c),'road_near_support_longitudinal_slices':coarse_y,'covered_fraction':len(coarse_y)/32,'not_final_geometry_success':True},
 'final_mesh':{'new_longitudinal_aabb_gap_display_units':new_gap,'v23_aabb_gap_display_units':old_gap,'aabb_is_not_road_semantics':True,
   'actual_center_road_gap_display_units':road_gap,'actual_center_road_gap_method':'nearest saved-mesh depth pixels, 0.005-unit query spacing, true raster precision about0.195 display units',
   'protected_center_corridor_top_surface_hit_fraction':float(hit.mean()),'top_surface_within_half_unit_of_parent_road_fraction':float((hit&near).mean()),
   'height_difference_to_parent_ground_quantiles':np.nanquantile(height-ground,[0,.1,.5,.9,1]).tolist(),
   'seam_paired_ground_hit_fraction':float(both.mean()),'seam_top_surface_step_quantiles':np.quantile(step,[0,.5,.9,1]).tolist() if len(step) else None,
   'forward_visibility':visibility,'native_world_bounds_before_fixed_chunk_translation':new_bounds},
 'measurements_definition':{'corridor':'21 X positions over center+/-2 display units; 501 longitudinal positions inside the new fixed chunk',
   'seam':'same21X rays, parent and candidate 0.25 units to either side of fixed seam',
   'visibility':'actual saved native opaque triangles rasterized along +nativeY, full native cube depth; five pixel rows per height; no near-plane exclusion',
   'top_surface':'uppermost triangle surface, not semantic ground and not complete volumetric/collision certification','units':'inherited arbitrary display units, not surveyed metres'},
 'failure_localization':'The generated coarse road-near support spans all32 longitudinal slices, while the final native mesh road retreats from the shared border. This supports insufficient fine Shape/surface boundary conditioning; coarse support presence alone does not prove exact surface geometry.',
 'visual_review':'AI inspected actual online GLBs at street and overhead viewpoints; road direction now aligned, buildings remain recognizable, road has a visible gap; not human acceptance',
 'evidence':['street_after_arrival.png','top_after_arrival.png','during_generation.png','chunks/road-boundary-0--1/BOUNDARY_TRACE.json'],
 'GPU_cost':read(NEW/'GPU_COST.json'),'runtime_state':read(OUT/'STATE.json')}
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'decision':result['decision'],'structure':result['native_structure_diagnostic'],'final_mesh':result['final_mesh'],'gpu_seconds':result['GPU_cost']['gpu_seconds']},indent=2))

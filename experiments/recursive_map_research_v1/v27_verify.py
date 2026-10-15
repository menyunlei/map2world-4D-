"""Independent, frozen saved-mesh checker. Never imports the generator/sampler.

Only positively labelled road patches are known. Height is NOT building semantics.
Unresolved required object correspondence always prevents automatic promotion.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
JOB=Path(__file__).resolve().parent
OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def decision(report,contract):
    g=contract['gates'];fail=[]
    if not report['valid_geometry']:fail.append('INVALID_NATIVE_MESH')
    if report['road_air_column_obstruction_fraction']>g['road_air_column_obstruction_fraction_max']:fail.append('KNOWN_ROAD_OBSTRUCTED')
    if report['road_surface_near_display_ground_fraction']<g['road_surface_near_display_ground_fraction_min']:fail.append('KNOWN_ROAD_SURFACE_NOT_ESTABLISHED')
    if report['semantic_building_correspondence']!='PASS':fail.append('MAP_OBJECT_CORRESPONDENCE_UNRESOLVED')
    if report['protected_cache_unchanged'] is not True:fail.append('PRESERVATION_FAILED')
    if report['neighbor_compatibility']=='FAIL':fail.append('NEIGHBOR_INCOMPATIBLE')
    return dict(accepted=not fail,reasons=fail,uncertainty_is_not_pass=True)
def evaluate(d):
    # New process, load the actual saved asset, verify against raw native triangles.
    import torch,trimesh,nvdiffrast.torch as dr
    from PIL import Image
    d=Path(d);began=time.time();p=read(d/'PLAN.json');contract=read(OUT/'CONTRACT.json')
    with np.load(d/'RAW_MESH.npz') as a:v0=a['vertices'];f0=a['faces']
    saved=trimesh.load(d/'world.glb',force='scene',process=False)
    geom=list(saved.geometry.values());assert len(geom)==1
    axis=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    assert np.array_equal(geom[0].faces,f0) and np.allclose(geom[0].vertices,v0@axis.T,atol=1e-7)
    transformed=saved.to_geometry() if hasattr(saved,'to_geometry') else saved.dump(concatenate=True)
    frame=np.array(p['boundary']['fixed_world_frame']['glTF_root_matrix']).reshape(4,4).T
    expected=trimesh.transform_points(v0@axis.T,frame)
    assert np.allclose(transformed.vertices,expected,atol=1e-4),'World frame differs from registered map frame'
    valid=bool(len(f0)>0 and np.isfinite(v0).all() and np.ptp(v0,axis=0).min()>1e-5)
    v=torch.tensor(v0,device='cuda',dtype=torch.float32);faces=torch.tensor(f0,device='cuda',dtype=torch.int32)
    ctx=dr.RasterizeCudaContext();res=1024;clip=torch.cat([v[:,:2]*2,-v[:,2:3],torch.ones_like(v[:,:1])],-1)[None].contiguous()
    top=torch.full((res,res),-2.,device='cuda');hit=torch.zeros((res,res),device='cuda',dtype=torch.bool)
    color=torch.zeros((res,res,4),device='cuda')
    rgba=np.asarray(geom[0].visual.vertex_colors).astype(np.float32)/255
    attrs=torch.tensor(rgba,device='cuda',dtype=torch.float32)[None].contiguous()
    for start in range(0,len(faces),4000000):
        fc=faces[start:start+4000000].contiguous();rast,_=dr.rasterize(ctx,clip,fc,resolution=[res,res])
        z,_=dr.interpolate(v[:,2:3][None].contiguous(),rast,fc);rgb,_=dr.interpolate(attrs,rast,fc)
        mask=rast[0,:,:,3]>0;higher=mask&(z[0,:,:,0]>top)
        color=torch.where(higher[:,:,None],rgb[0],color);top=torch.where(higher,z[0,:,:,0],top);hit|=mask
    h=top.cpu().numpy();occupied=hit.cpu().numpy();rgb=color[:,:,:3].cpu().numpy();rgb[~occupied]=.12
    # Flip native +Y upward to the source-map image convention for display only.
    Image.fromarray((rgb[::-1].clip(0,1)*255).astype(np.uint8)).save(d/'TOP_NATIVE.png')
    q=np.load(OUT/'EVALUATION_QUERIES.npz')['road_xy_native'];ix=np.clip(((q[:,0]+.5)*res).astype(int),0,res-1);iy=np.clip(((q[:,1]+.5)*res).astype(int),0,res-1)
    found=occupied[iy,ix];heights=(h[iy,ix]-contract['coordinates']['floor'])*100
    # Slab clipping distinguishes road-support surfaces from overhead surfaces.
    # No camera/near-plane loophole: each query includes its entire registered slab.
    floor=contract['coordinates']['floor']
    def slab(lo,hi):
        projected=torch.cat([v[:,:2]*2,1-2*(v[:,2:3]-lo)/(hi-lo),torch.ones_like(v[:,:1])],-1)[None].contiguous()
        covered=torch.zeros((res,res),device='cuda',dtype=torch.bool)
        for off in range(0,len(faces),4000000):
            raster,_=dr.rasterize(ctx,projected,faces[off:off+4000000].contiguous(),resolution=[res,res])
            covered|=raster[0,:,:,3]>0
        return covered.cpu().numpy()[iy,ix]
    near=slab(floor-.005,floor+.005)
    blocked=slab(floor+.005,floor+.08)
    hist=read(OUT/'BASELINE_CACHE.json')
    preserved=all(sha(JOB/k)==v for k,v in hist.items())
    # No trusted building mask exists; never call everything above height a building.
    report=dict(attempt=p['attempt'],epoch=time.time(),asset_sha256=sha(d/'world.glb'),
        verifier_sha256=sha(__file__),contract_sha256=sha(OUT/'CONTRACT.json'),
        valid_geometry=valid,faces=len(f0),vertices=len(v0),native_bounds=[v0.min(0).tolist(),v0.max(0).tolist()],
        road_ray_count=len(q),road_surface_near_display_ground_fraction=float(near.mean()),
        road_air_column_obstruction_fraction=float(blocked.mean()),road_surface_hit_fraction=float(found.mean()),
        road_top_height_quantiles=([float(x) for x in np.quantile(heights[found],[0,.1,.5,.9,1])] if found.any() else None),
        semantic_building_correspondence='UNCERTAIN',semantic_building_IoU=None,
        footprint_test='UNRESOLVED: roof proxies and unlabeled native mesh. No height-based vegetation/platform shortcut.',
        horizontal_walkability='NOT_TESTED: vertical-column checks are not a collision or topology certificate',
        neighbor_compatibility='NOT_TESTED_NO_ACCEPTED_MAP_ALIGNED_NEIGHBOR',
        boundary_continuity=None,protected_cache_unchanged=preserved,
        measured_seconds=time.time()-began,render_semantics='actual opaque mesh, all original triangles, ground +/-0.5 and air 0.5..8 display-unit slabs; uppermost heights separately; not horizontal rays',
        evidence='TOP_NATIVE.png; RAY_RESULTS.npz; independent asset reload; no output repair')
    report['decision']=decision(report,contract)
    np.savez_compressed(d/'RAY_RESULTS.npz',xy=q,hit=found,height=heights,near_ground=near,road_blocked=blocked)
    bad=q[blocked];report['targeted_feedback']=dict(kind='known_positive_road_columns',
        known_road_blocked_queries=int(blocked.sum()),missing_road_ground_queries=int((~near).sum()),
        first_obstructed_world_xz=([[float(100*x),float(-100*y)] for x,y in bad[:16]]),
        permitted_retry='Reuse fixed full positive-road observation; do not select new query rays or change gates',
        semantic_limit='Retry cannot resolve missing object semantics; no automatic promotion from road improvement alone')
    dump(d/'REPORT.json',report);print(json.dumps(report),flush=True)
def selftest():
    c={'gates':{'road_air_column_obstruction_fraction_max':.05,'road_surface_near_display_ground_fraction_min':.95}}
    good=dict(valid_geometry=True,road_air_column_obstruction_fraction=0,road_surface_near_display_ground_fraction=1,
              semantic_building_correspondence='PASS',protected_cache_unchanged=True,neighbor_compatibility='PASS')
    tests={'valid_fixture':decision(good,c)['accepted']}
    for name,change in [('road_obstruction',{'road_air_column_obstruction_fraction':.2}),('missing_road',{'road_surface_near_display_ground_fraction':.2}),
                        ('semantic_unknown',{'semantic_building_correspondence':'UNCERTAIN'}),('changed_cache',{'protected_cache_unchanged':False}),('bad_neighbor',{'neighbor_compatibility':'FAIL'})]:
        tests[name]=not decision({**good,**change},c)['accepted']
    assert all(tests.values());return tests
if __name__=='__main__':
    if '--selftest' in sys.argv:print(json.dumps(selftest()))
    else:evaluate(Path(sys.argv[1]))

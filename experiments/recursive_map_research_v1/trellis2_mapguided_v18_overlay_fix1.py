"""Chunk only the diagnostic raster, like native MeshRenderer; assets unchanged."""
from paths import JOB,read,save,sha
from trellis2_mapguided_v18 import OUT,SEED
import sys

def worker(mode):
    import torch,numpy as np,nvdiffrast.torch as dr
    from PIL import Image
    torch.set_num_threads(4);p=read(OUT/'PLAN.json')
    assert (OUT/mode/'RELOAD.json').exists()
    with np.load(OUT/mode/'RAW_MESH.npz') as z:
        v=torch.from_numpy(z['vertices'].copy()).cuda().float()
        faces=torch.from_numpy(z['faces'].astype(np.int32)).cuda()
    clip=torch.cat([v[:,:2]*2,-v[:,2:3]*.5,torch.ones_like(v[:,:1])],-1)[None].contiguous()
    context=dr.RasterizeCudaContext();projection=np.zeros((512,512),bool)
    zstart=p['guidance']['building_z_start']/64-.5
    for offset in range(0,len(faces),8000000):
        f=faces[offset:offset+8000000]
        keep=(v[f.long(),2].amax(-1)>=zstart);f=f[keep].contiguous()
        if len(f):
            with torch.no_grad():rast,_=dr.rasterize(context,clip,f,resolution=[512,512])
            projection|=(rast[0,:,:,3]>0).cpu().numpy()
        print('UNCHANGED_MESH_PROJECTION_CHUNK',mode,offset,flush=True)
    with np.load(p['maps'][mode]['path']) as z:fps=z['footprints'].copy()>.5
    target=np.repeat(np.repeat(fps.any(0),8,0),8,1)
    rgb=np.zeros((512,512,3),np.uint8)+20
    rgb[projection]=[55,155,230];rgb[target]=[230,65,55];rgb[projection&target]=[235,215,90]
    Image.fromarray(rgb[::-1].copy()).save(OUT/mode/'actual_mesh_map_overlay.png')
    save(OUT/mode/'MESH_MAP.json',{'map_kind':'Approximate manually annotated roof proxies, not surveyed GT',
        'projection':'All saved triangles above fixed floor band; orthographic XY; native 8M-face raster chunks; no object placement',
        'above_floor_projection_IoU':float((projection&target).sum()/max(1,(projection|target).sum())),
        'per_footprint_coverage':[float((projection&np.repeat(np.repeat(fp,8,0),8,1)).sum()/max(1,fp.sum()*64)) for fp in fps],
        'outside_fraction':float((projection&~target).sum()/max(1,projection.sum())),
        'not_a_geometry_or_visual_quality_pass':True,'asset_unchanged':True})

def owner():
    import types,threading,concurrent.futures,fcntl
    import subcell_owner_v1 as lease
    from trellis2_mapguided_v18_owner import verify
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'COMPLETE.json').exists()
        for m in ['correct','edit']:assert (OUT/m/'RELOAD.json').exists() and not (OUT/m/'MESH_MAP.json').exists()
        save(OUT/'DIAGNOSTIC_RASTER_FIX.json',{'script':__file__,'sha256':sha(__file__),
            'reason':'Full 50M-face diagnostic raster exceeded rasterizer limit; native renderer uses chunks',
            'no_asset_regeneration_or_geometry_changes':True})
        scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
            'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'guard':threading.Lock(),'active':{},'halt':threading.Event()}
        fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
        fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
        gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs']
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(fn,gpus[i],m,SEED,'overlay_chunks','trellis2_mapguided_v18_overlay_fix1.py',['--mode',m],seconds=180)
                for i,m in enumerate(['correct','edit'])]
            for f in fs:f.result()
        save(OUT/'COMPLETE.json',{'generated_modes':['correct','edit'],'one_attempt_only':True,'no_more_jobs':True,
            'render_only_scheduling_resume':True,'diagnostic_raster_chunk_fix':True,
            'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
            'visual_and_map_review_pending':True,'new_model_training':False})

if __name__=='__main__':
    owner() if '--owner' in sys.argv else worker(sys.argv[sys.argv.index('--mode')+1])

"""Saved v27 stage localization; no model generation or training."""
from pathlib import Path
from paths import JOB,read,save,sha
import sys,os,time,json,subprocess,fcntl
import numpy as np
OUT=JOB/'map_object_world_v1/trellis2_road_stage_v28'
OLD=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def task():
    from trellis2_prepare_v1 import REPO,DEST
    sys.path.insert(0,str(REPO));os.environ['CUDA_HOME']=str(DEST/'environment')
    import torch
    from trellis2 import models
    torch.set_num_threads(4)
    p=read(OLD/'attempts/attempt-1/PLAN.json');contract=read(OLD/'CONTRACT.json');floor=contract['coordinates']['floor']
    path=read(Path(p['runtime'])/'pipeline.json')['args']['models']['sparse_structure_decoder']
    decoder=models.from_pretrained(path).cuda().eval().requires_grad_(False)
    z=np.load(OLD/'attempts/attempt-1/STRUCTURE_LATENT.npz')['latent']
    with torch.no_grad():dense=(decoder(torch.tensor(z,device='cuda')).float()>0)[0,0]
    assert list(dense.shape)==[64,64,64]
    pooled=torch.nn.functional.max_pool3d(dense[None,None].float(),2,2)>0
    coords=torch.argwhere(pooled)[:,[0,2,3,4]].int().cpu().numpy()
    saved=np.load(OLD/'attempts/attempt-1/GENERATED_STRUCTURE.npz')['coords']
    assert np.array_equal(coords,saved),'Not the actual Structure contract'
    d=dense.cpu().numpy();del decoder
    np.savez_compressed(OUT/'STRUCTURE64_REPLAY.npz',occupied=d,pooled_coords=coords)
    q=np.load(OLD/'EVALUATION_QUERIES.npz')['road_xy_native'];xy64=np.clip(((q+.5)*64).astype(int),0,63)
    col=d[xy64[:,0],xy64[:,1]]
    a=np.load(OLD/'attempts/attempt-1/ROAD_OBSERVATION.npz');ground=a['ground'];known=a['known']
    zi=int(np.flatnonzero(ground.any((0,1)))[0]);ground_cells64=np.argwhere(ground)
    def bounds(res,idx):return [float((idx/res-.5-floor)*100),float(((idx+1)/res-.5-floor)*100)]
    sections=[]
    for i in [0,1]:
        c=np.load(OLD/f'attempts/attempt-{i}/GENERATED_STRUCTURE.npz')['coords'][:,1:]
        occ=np.zeros((32,32,32),bool);occ[tuple(c.T)]=1;xy=np.clip(((q+.5)*32).astype(int),0,31);support=occ[xy[:,0],xy[:,1]]
        r=np.load(OLD/f'attempts/attempt-{i}/RAY_RESULTS.npz');bad=r['road_blocked'];h=r['height']
        center=((np.arange(32)+.5)/32-.5-floor)*100
        # Only cells ENTIRELY above the ground guard count as unambiguous air
        # support. A coarse cell intersecting ground cannot certify obstruction.
        bottom=(np.arange(32)/32-.5-floor)*100
        sections.append(dict(attempt=i,road_queries=len(q),saved_Structure_cells=len(c),
            ground_intersecting_Q32_support_fraction=float(support[:,(bottom<=.5)&(bottom+100/32>=-.5)].any(1).mean()),
            unambiguous_air_Q32_support_fraction=float(support[:,(bottom>=.5)&(bottom<8)].any(1).mean()),
            ground_cell_center_as_air_is_invalid=float(support[:,(center>.5)&(center<8)].any(1).mean()),
            mesh_blocked_query_count=int(bad.sum()),blocked_top_surface_quantiles=np.quantile(h[bad],[0,.25,.5,.75,1]).tolist() if bad.any() else None,
            blocked_top_below_3_display_units=int((bad&(h<3)).sum()),
            blocked_top_above_3_display_units=int((bad&(h>=3)).sum())))
    query_stage=dict(dense64_road_any_support_fraction=float(col.any(1).mean()),
        dense64_ground_bin_support_fraction=float(col[:,zi].mean()),
        dense64_above_ground_plus_2_voxels_fraction=float(col[:,zi+2:zi+6].any(1).mean()),
        native_dense64_ground_bin=zi,dense64_ground_bin_world_bounds=bounds(64,zi),
        pooled32_ground_bin=zi//2,pooled32_ground_bin_world_bounds=bounds(32,zi//2),
        pooled32_ground_center_world_height=sum(bounds(32,zi//2))/2,
        frozen_map_clearance_lower_bound=.5,
        surface_height_not_determined_by_support=True)
    result=dict(actual_dense_decoder_replayed=True,pooled32_exact_match=True,
        replayed_latent_sha256=sha(OLD/'attempts/attempt-1/STRUCTURE_LATENT.npz'),
        decoder=path,sections=sections,stage=query_stage,
        caution='Q occupancy is surface support, not solid volume. Ground-mixed cells cannot be called roof/wall obstruction.',
        preliminary_localization='Positive road-ground support is injected already at Structure, then downpooled into a 3.125-display-unit cell. Most final blocking surfaces are low; the coarse support cannot specify their exact height. Distinguish this from tall-building intrusion.',
        world_frame_unchanged=True,old_thresholds_unchanged=True,old_results_preserved=True)
    save(OUT/'STAGE_DIAGNOSIS.json',result);print(json.dumps(result),flush=True)
def owner():
    from v26_resident import GPUS
    OUT.mkdir(parents=True,exist_ok=True)
    index,uuid=GPUS[0]
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert f'{index}, {uuid}' in inventory and uuid not in owners
    h=(JOB/'gpu_leases/gpu0.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    save(OUT/'ONE_STAGE_CHECK.json',dict(epoch=time.time(),gpu=uuid,source_sha256=sha(__file__)))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ATTN_BACKEND':'flash_attn','SPCONV_ALGO':'native','OMP_NUM_THREADS':'4'}
    began=time.time()
    try:
        with (OUT/'stage_check.log').open('x') as f:r=subprocess.run([sys.executable,'-B','-u',__file__,'--work'],cwd=JOB,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
        save(OUT/'GPU_COST_STAGE.json',dict(index=index,uuid=uuid,gpu_seconds=time.time()-began,returncode=r.returncode))
        print(json.dumps(dict(returncode=r.returncode)),flush=True)
    finally:h.close()
if __name__=='__main__':
    if '--work' in sys.argv:task()
    else:owner()

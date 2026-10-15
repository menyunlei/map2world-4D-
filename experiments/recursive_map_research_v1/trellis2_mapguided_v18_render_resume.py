"""Render-only fallback if GPUs2/3 acquired an unrelated owner after preflight."""
from paths import JOB,read,save,sha
from trellis2_mapguided_v18 import OUT,SEED
from trellis2_mapguided_v18_owner import verify
import subcell_owner_v1 as lease
import types,threading,concurrent.futures,fcntl

assert (OUT/'FAILURE.json').exists() and not (OUT/'COMPLETE.json').exists()
with (OUT/'owner.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    modes=[m for m in ['correct','edit'] if not (OUT/m/'MESH_MAP.json').exists()]
    for mode in modes:
        assert (OUT/mode/'ASSET.json').exists() and not (OUT/mode/'views').exists(),('PARTIAL_RENDER_DO_NOT_OVERWRITE',mode)
    save(OUT/'RENDER_RESUME_SOURCE.json',{'path':__file__,'sha256':sha(__file__),
        'reason':'Render on released GPUs0/1; do not touch unrelated GPUs2/3',
        'no_model_resampling_or_asset_changes':True})
    scope={**lease.run.__globals__,'NEW':OUT,'frozen':verify,
        'costs':lambda:sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'guard':threading.Lock(),'active':{},'halt':threading.Event()}
    fn=types.FunctionType(lease.run.__code__,scope,lease.run.__name__,lease.run.__defaults__)
    fn.__code__=fn.__code__.replace(co_consts=tuple(7100 if c==114900 else c for c in fn.__code__.co_consts))
    gpus=read(JOB/'map_object_world_v1/trellis2_stage_attribution_v16/PLAN.json')['GPUs']
    with concurrent.futures.ThreadPoolExecutor(max(1,len(modes))) as pool:
        fs=[pool.submit(fn,gpus[i],m,SEED,'views_gpu01','trellis2_mapguided_v18_views.py',['--mode',m],seconds=600)
            for i,m in enumerate(modes)]
        for f in fs:f.result()
    save(OUT/'COMPLETE.json',{'generated_modes':['correct','edit'],'one_attempt_only':True,'no_more_jobs':True,
        'render_only_scheduling_resume':True,'GPU_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'visual_and_map_review_pending':True,'new_model_training':False})

"""Float camera arithmetic only; rerender saved meshes, never regenerate."""
from pathlib import Path
p=Path(__file__).with_name('v28_views.py');s=p.read_text()
s=s.replace("OUT/'native64/world.glb'","OUT/'native64_contiguous/world.glb'")
s=s.replace("dest=OUT/'views'/which","dest=OUT/'views_float'/which")
s=s.replace("eye=np.array(eye);forward=np.array(target)-eye", "eye=np.array(eye,dtype=np.float64);forward=np.array(target,dtype=np.float64)-eye")
s=s.replace("idx=1;uuid=GPUS[idx][1]", """owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    free=[(i,u) for i,u in GPUS if u not in owners]
    assert free,'No idle authorized GPU'
    idx,uuid=free[0]""")
s=s.replace("(JOB/'gpu_leases/gpu1.lock')", "(JOB/'gpu_leases'/f'gpu{idx}.lock')")
s=s.replace("f'ONE_RENDER_{which}.json'","f'ONE_RENDER_FIX_{which}.json'")
s=s.replace("f'GPU_COST_RENDER_{which}.json'","f'GPU_COST_RENDER_FIX_{which}.json'")
s=s.replace("f'render_{which}.log'","f'render_fix_{which}.log'")
s=s.replace("child.wait(timeout=120)","child.wait(timeout=(120-read(OUT/'GPU_COST_RENDER_baseline.json')['gpu_seconds'] if which=='baseline' else 120))")
exec(compile(s,str(p),'exec'),globals())

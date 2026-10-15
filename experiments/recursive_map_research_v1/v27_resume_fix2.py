"""Resume once missing immutable local scaffold copy has been transferred."""
from pathlib import Path
p=Path(__file__).with_name('v27_resume_fix1.py')
source=p.read_text().replace('FIX1','FIX2')
source=source.replace("value['lease_seconds']-=original_read(OUT/'GPU_COST.json')['gpu_seconds']",
    "value['lease_seconds']-=original_read(OUT/'GPU_COST.json')['gpu_seconds']+original_read(OUT/'GPU_COST_FIX1.json')['gpu_seconds']")
source=source.replace("(d/'TOP_NATIVE.png').rename(d/'TOP_NATIVE_BEFORE_HASH_FIX.png')",
    "(d/'TOP_NATIVE.png').rename(d/'TOP_NATIVE_BEFORE_DEPLOYMENT_FIX.png')")
exec(compile(source,str(p),'exec'),globals())

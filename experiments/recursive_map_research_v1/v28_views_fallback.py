"""Saved-asset render only; change GPU assignment, never geometry or cameras."""
from pathlib import Path
p=Path(__file__).with_name('v28_views.py')
s=p.read_text().replace("idx=1;uuid=GPUS[idx][1]", """owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    free=[(i,u) for i,u in GPUS if u not in owners]
    assert free,'No idle authorized GPU'
    idx,uuid=free[0]""").replace("(JOB/'gpu_leases/gpu1.lock')", "(JOB/'gpu_leases'/f'gpu{idx}.lock')")
exec(compile(s,str(p),'exec'),globals())

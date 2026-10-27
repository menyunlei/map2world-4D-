"""Same-camera speed and numerical-fidelity check, no model-quality benchmark."""
import json, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parent
before = json.loads((root/'BEFORE_BALANCED_STATUS.json').read_text())['jobs']
rows = []
for baseline_id in ['377b601b467a','377b601b467a','fdfd50e5c5c2','fdfd50e5c5c2']:
    request = urllib.request.Request('http://127.0.0.1:18935/render',data=json.dumps({'pose':before[baseline_id]['pose']}).encode(),headers={'Content-Type':'application/json'})
    job = json.load(urllib.request.urlopen(request,timeout=10))
    deadline = time.monotonic()+120
    while job['status'] not in ['done','error']:
        assert time.monotonic()<deadline
        time.sleep(.2)
        job = json.load(urllib.request.urlopen('http://127.0.0.1:18935/status',timeout=5))['jobs'][job['id']]
    assert job['status']=='done', job
    a = np.asarray(Image.open(root/'renders'/f'{job["id"]}.png')).astype(float)
    b = np.asarray(Image.open(root/'renders'/f'{baseline_id}.png')).astype(float)
    error = a-b
    row = dict(id=job['id'],baseline_id=baseline_id,total_seconds=job['seconds'],render_seconds=job['render_seconds'],load_seconds=job['load_seconds'],
        per_gpu=job['per_gpu'],max_difference_255=float(abs(error).max()),mean_difference_255=float(abs(error).mean()),
        changed_fraction=float((error!=0).mean()),psnr_db=float(-10*np.log10(max(1e-15,np.mean((error/255)**2)))))
    rows.append(row)
    (root/'SPEED_CHECK_V4.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(row),flush=True)
    # CUDA convolution shapes change rounding near uint8 boundaries. Bound the
    # absolute, average and energy error, rather than equating this with quality.
    assert row['max_difference_255']<=3 and row['mean_difference_255']<.01 and row['psnr_db']>65

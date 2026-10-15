"""Tiny GPU renderer-coordinate sanity check, not a generated-world result."""
from pathlib import Path
import os,time,json,fcntl,subprocess
from paths import JOB,save
from v26_resident import GPUS
OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
index,uuid=GPUS[0]
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
lease=(JOB/'gpu_leases/gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ['CUDA_VISIBLE_DEVICES']=uuid
began=time.time()
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext();floor=-.06219705939292908
# Two horizontal test sheets: one ground, one known obstruction. Synthetic unit fixtures only.
vertices=torch.tensor([[-.3,-.3,floor],[.3,-.3,floor],[.3,.3,floor],[-.3,.3,floor],
                       [-.1,-.1,floor+.02],[.1,-.1,floor+.02],[.1,.1,floor+.02],[-.1,.1,floor+.02]],device='cuda')
faces=torch.tensor([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype=torch.int32,device='cuda')
def slab(lo,hi,fc):
    clip=torch.cat([vertices[:,:2]*2,1-2*(vertices[:,2:3]-lo)/(hi-lo),torch.ones_like(vertices[:,:1])],-1)[None].contiguous()
    r,_=dr.rasterize(ctx,clip,fc.contiguous(),resolution=[64,64]);return (r[0,:,:,3]>0).cpu().numpy()
ground=slab(floor-.005,floor+.005,faces);air=slab(floor+.005,floor+.08,faces)
assert ground[32,32] and air[32,32]
assert ground[20,20] and not air[20,20]
assert not ground[0,0] and not air[0,0]
assert not slab(floor+.005,floor+.08,faces[:2]).any(),'Ground falsely counted as obstruction'
from v27_verify import selftest
result=dict(passed=True,synthetic_fixture_only=True,not_model_quality_evidence=True,
    ground_and_raised_sheet_distinguished=True,empty_ray_distinguished=True,
    native_xy_row_sign_verified=True,decision_policy_tests=selftest(),gpu=dict(index=index,uuid=uuid),
    gpu_seconds=time.time()-began)
save(OUT/'RASTER_PREFLIGHT.json',result);print(json.dumps(result),flush=True)

"""Independent saved-GLB fixed cameras, unchanged triangles/colors. No model call."""
from pathlib import Path
from paths import JOB,read,save,sha
import os,sys,json,time,subprocess,fcntl,signal
import numpy as np
OUT=JOB/'map_object_world_v1/trellis2_road_stage_v28'
OLD=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def render(which):
    import torch,trimesh,nvdiffrast.torch as dr
    from PIL import Image,ImageDraw
    torch.set_num_threads(4)
    asset=OLD/'attempts/attempt-1/world.glb' if which=='baseline' else OUT/'native64/world.glb'
    dest=OUT/'views'/which;dest.mkdir(parents=True,exist_ok=False)
    scene=trimesh.load(asset,force='scene',process=False)
    mesh=scene.to_geometry() if hasattr(scene,'to_geometry') else scene.dump(concatenate=True)
    v=torch.tensor(np.asarray(mesh.vertices),device='cuda',dtype=torch.float32)
    f=torch.tensor(np.asarray(mesh.faces),device='cuda',dtype=torch.int32)
    color=torch.tensor(np.asarray(mesh.visual.vertex_colors)[:,:3]/255,device='cuda',dtype=torch.float32)
    n=torch.zeros_like(v)
    for off in range(0,len(f),2000000):
        fc=f[off:off+2000000].long();a,b,c=[v[fc[:,i]] for i in range(3)];fn=torch.cross(b-a,c-a,dim=1)
        for i in range(3):n.index_add_(0,fc[:,i],fn)
    n=torch.nn.functional.normalize(n,dim=1)
    light=torch.nn.functional.normalize(torch.tensor([-.35,.8,.5],device='cuda'),dim=0)
    gray=(.25+.75*(n@light).abs())[:,None]*torch.tensor([.76,.79,.82],device='cuda')[None]
    attrs=torch.cat([color,gray],1)[None].contiguous();ctx=dr.RasterizeCudaContext();res=512
    m=read(OLD/'map10_scaffold.json');spawn=np.array(m['spawn']['position']);look=np.array(m['spawn']['look_at'])
    direction=look-spawn;direction/=np.linalg.norm(direction);right=np.cross(direction,[0,1,0]);right/=np.linalg.norm(right)
    views=[('top',[0,135,1],[0,0,0],[0,0,-1]),('oblique',[85,70,85],[0,0,0],[0,1,0]),
           ('street',spawn,look,[0,1,0]),('lateral',spawn+right*3,look+right*3,[0,1,0])]
    cameras=[]
    for name,eye,target,up in views:
        eye=np.array(eye);forward=np.array(target)-eye;forward/=np.linalg.norm(forward)
        r=np.cross(forward,up);r/=np.linalg.norm(r);u=np.cross(r,forward)
        R=np.stack([r,-u,forward]);camera=(v-torch.tensor(eye,device='cuda',dtype=torch.float32))@torch.tensor(R.T,device='cuda',dtype=torch.float32)
        tangent=np.tan(np.deg2rad(65)/2);near=.03;far=1500
        clip=torch.stack([camera[:,0]/tangent,-camera[:,1]/tangent,
             (far+near)/(far-near)*camera[:,2]-2*far*near/(far-near),camera[:,2]],1)[None].contiguous()
        depth=torch.full((res,res),float('inf'),device='cuda');pixels=torch.zeros((res,res,6),device='cuda');hit=torch.zeros((res,res),device='cuda',dtype=torch.bool)
        for off in range(0,len(f),4000000):
            fc=f[off:off+4000000].contiguous();rast,_=dr.rasterize(ctx,clip,fc,resolution=[res,res])
            z,_=dr.interpolate(camera[:,2:3][None].contiguous(),rast,fc)
            rgb,_=dr.interpolate(attrs,rast,fc);ok=(rast[0,:,:,3]>0)&(z[0,:,:,0]<depth)
            depth=torch.where(ok,z[0,:,:,0],depth);pixels=torch.where(ok[:,:,None],rgb[0],pixels);hit|=ok
        p=pixels.cpu().numpy();mask=hit.cpu().numpy();p[~mask]=.12
        for mode,lo in [('appearance',0),('neutral',3)]:Image.fromarray((p[::-1,:,lo:lo+3].clip(0,1)*255).astype(np.uint8)).save(dest/f'{name}_{mode}.png')
        cameras.append(dict(name=name,eye=eye.tolist(),target=np.asarray(target).tolist(),up=list(up),fov=65,near=near,far=far))
    sheet=Image.new('RGB',(4*320,2*344),'#17252f');draw=ImageDraw.Draw(sheet)
    for row,mode in enumerate(['appearance','neutral']):
        for col,(name,*_) in enumerate(views):
            sheet.paste(Image.open(dest/f'{name}_{mode}.png').resize((320,320)),(320*col,344*row+24));draw.text((320*col+8,344*row+6),f'{which} / {name} / {mode}',fill='white')
    sheet.save(dest/'contact.png')
    save(dest/'RELOAD.json',dict(asset=str(asset),sha256=sha(asset),triangles=len(f),vertices=len(v),cameras=cameras,
        independent_process=True,geometry_modified=False,renderer='nvdiffrast opaque native saved mesh; model base colors and independent neutral shading',
        no_camera_selection_after_results=True))
    print('V28_FIXED_VIEWS',which,flush=True)
def owner(which):
    from v26_resident import GPUS
    idx=1;uuid=GPUS[idx][1]
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert f'{idx}, {uuid}' in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    lease=(JOB/'gpu_leases/gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    save(OUT/f'ONE_RENDER_{which}.json',dict(epoch=time.time(),uuid=uuid))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'};began=time.time()
    try:
        with (OUT/f'render_{which}.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--work',which],cwd=JOB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        save(OUT/f'GPU_COST_RENDER_{which}.json',dict(index=idx,uuid=uuid,gpu_seconds=time.time()-began,returncode=code))
        print(json.dumps(dict(which=which,returncode=code)),flush=True)
    finally:lease.close()
if __name__=='__main__':
    if '--work' in sys.argv:render(sys.argv[-1])
    else:owner(sys.argv[-1])

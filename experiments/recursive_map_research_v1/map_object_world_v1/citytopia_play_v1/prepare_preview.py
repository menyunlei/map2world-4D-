"""Navigation-only RGB-D proxy from existing native Map10 outputs; no inference."""
from pathlib import Path
import json, shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'citytopia_road_diagnosis_20261027_v1/route_density5_v2'
OUT=ROOT/'preview'

def make():
    OUT.mkdir(exist_ok=True)
    cams=json.loads((SRC/'CAMERAS.json').read_text())['cameras']
    rows=[]
    for c in cams:
        i=c['index']; folder=SRC/'native'/f'{i:04d}'
        d=np.load(folder/'surface.npz')
        # First intersections are coarse conditioning geometry, NOT learned density.
        depth=d['first_depth'][::4,::4].copy(); ids=d['first_id'][::4,::4]
        h,w=depth.shape; yy,xx=np.mgrid[:h,:w]*4
        pose=c['citydreamer_pose']; p=pose['cam_position']; t=pose['cam_look_at']
        origin=np.array([p['y'],p['x'],p['z']],np.float32)
        forward=np.array([t['y']-p['y'],t['x']-p['x'],t['z']-p['z']],np.float32)
        forward/=np.linalg.norm(forward); side=np.cross(forward,[0,0,1]); side/=np.linalg.norm(side)
        up=np.cross(side,forward)
        rays=up*(270-yy)[...,None]+side*(xx-480)[...,None]+forward*(270/np.tan(np.pi/6))
        rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
        valid=np.isfinite(depth)&(depth>.05)&(depth<300)&(ids!=0)
        xyz=origin/4+rays*np.where(valid,depth,0)[...,None]
        pos=np.stack([xyz[...,0]-384,xyz[...,2]-3.5,-(xyz[...,1]-384)],-1).astype('<f4')
        uv=np.stack([(xx+.5)/960,1-(yy+.5)/540],-1).astype('<f4')
        n=np.arange(h*w).reshape(h,w); a=n[:-1,:-1]; b=n[:-1,1:]; z=n[1:,:-1]; e=n[1:,1:]
        tri=np.concatenate([np.stack([a,z,b],-1).reshape(-1,3),np.stack([b,z,e],-1).reshape(-1,3)])
        flat=depth.ravel(); labels=ids.ravel(); tv=valid.ravel()[tri].all(1)
        tv&=(labels[tri].max(1)==labels[tri].min(1))
        tv&=(flat[tri].max(1)-flat[tri].min(1)<np.maximum(.8,.04*flat[tri].min(1)))
        tri=tri[tv].astype('<u4')
        blob=pos.tobytes()+uv.tobytes()+tri.tobytes()
        (OUT/f'{i:04d}.bin').write_bytes(blob)
        shutil.copyfile(folder/'final.png',OUT/f'{i:04d}.png')
        rows.append(dict(index=i,tag=c['tag'],pose=pose,vertices=h*w,indices=tri.size,
                         mesh=f'/preview/{i:04d}.bin',rgb=f'/preview/{i:04d}.png'))
    shutil.copyfile(SRC.parent/'diagnose/INPUT_AUDIT.json',OUT/'INPUT_AUDIT.json') if (SRC.parent/'diagnose/INPUT_AUDIT.json').exists() else None
    (OUT/'manifest.json').write_text(json.dumps(dict(cameras=rows,default=14,
        source='Saved CityTopia Map10 native RGB + coarse first-intersection depth',
        preview_only=True,world_scale='4 native units/metre',native_image=[960,540],fov=60)))
    print(json.dumps(dict(preview_frames=len(rows),directory=str(OUT))),flush=True)

if __name__=='__main__':make()

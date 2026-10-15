"""Encode existing ID-labeled TRAIN geometry, not meshes generated from maps."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_data_v6_fix1 import OUT
from paths import read, save, sha, JOB
import argparse,time
import numpy as np
import torch
import o_voxel
from trellis2 import models
from trellis2.modules.sparse import SparseTensor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    encoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')).cuda().eval().requires_grad_(False)
    ss_encoder=models.from_pretrained(str(JOB.parent/'map_native_scene_dit_v1/weights/codec/ckpts/ss_enc_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    ss_decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    done=[]
    for i,r in enumerate(read(OUT/'SOURCE_GEOMETRIES.json')['records']):
        if i%4!=a.worker:continue
        tick=time.time();assert sha(r['source_geometry'])==r['sha256']
        with np.load(r['source_geometry']) as z:
            v=torch.from_numpy(z['vertices'].copy()).float();f=torch.from_numpy(z['faces'].copy()).int()
        coords,dual,intersected=o_voxel.convert.mesh_to_flexible_dual_grid(v,f,grid_size=1024,
            aabb=[[-.5,-.5,-.5],[.5,.5,.5]],face_weight=1.,boundary_weight=.2,regularization_weight=1e-2)
        assert len(coords)
        local=((dual*1024-coords).clamp(0,1)*255).to(torch.uint8).float()/255
        c=torch.cat([torch.zeros(len(coords),1,dtype=torch.int32),coords.int()],1).cuda()
        sparse=SparseTensor(feats=local.cuda(),coords=c)
        with torch.no_grad():
            latent=encoder(sparse,sparse.replace(intersected.float().cuda()),sample_posterior=False)
            assert latent.feats.shape[1]==32 and torch.isfinite(latent.feats).all()
            cc=latent.coords[:,1:].long()
            assert bool(((cc>=0)&(cc<64)).all())
            occupancy=torch.zeros((1,1,64,64,64),device='cuda')
            occupancy[0,0,cc[:,0],cc[:,1],cc[:,2]]=1
            s0=ss_encoder(occupancy,sample_posterior=False).float()
            decoded=ss_decoder(s0)>0
            iou=float((decoded&occupancy.bool()).sum()/(decoded|occupancy.bool()).sum().clamp_min(1))
        path=OUT/'targets'/f"{r['key']}.npz";path.parent.mkdir(exist_ok=True)
        with path.open('xb') as h:np.savez_compressed(h,coords=latent.coords.cpu().numpy(),
            feats=latent.feats.float().cpu().numpy(),S0=s0.cpu().numpy(),support64=occupancy.cpu().numpy())
        save(OUT/'target_records'/f"{r['key']}.json",{**r,'native_target':str(path),'native_target_sha256':sha(path),
            'shape_channels':32,'shape_cells':len(cc),'SS_shape':list(s0.shape),
            'SS_codec_reconstruction_IoU':iou,'seconds':time.time()-tick,
            'encoder_frozen':True,'model_training_updates':0,
            'SS_target':'native SS encoder of actual native ShapeEncoder support; no world ground merged'})
        done.append(r['key']);print('ENCODE_OBJECT',a.worker,len(done),r['key'],len(cc),iou,flush=True)
        del sparse,latent,coords,dual,local,intersected,occupancy,decoded,s0
    save(OUT/f'ENCODE_COMPLETE_WORKER{a.worker}.json',{'objects':done,'optimizer_updates':0})

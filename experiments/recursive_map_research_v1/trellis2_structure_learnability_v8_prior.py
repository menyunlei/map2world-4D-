"""Diagnostic source-mesh images and unchanged image-conditioned native prior.

Reference RGB is used here ONLY, never by map-only training/sampling. No new
geometry, materials, decoder, or model parameters are produced by this check.
"""
from trellis2_map_learning_train_fix1 import base
from trellis2_structure_learnability_v8_prepare import OUT
from trellis2_structure_learnability_v8_eval import iou,tile
from trellis2_map_entry_views_v2 import camera_matrices
from trellis2.representations import Mesh
from trellis2.renderers import MeshRenderer
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from trellis2 import models
from paths import read,save,sha
import torch,numpy as np
from PIL import Image,ImageDraw

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    plan=read(OUT/'PLAN.json');root=OUT/'native_prior_check';root.mkdir(parents=True,exist_ok=False)
    renderer=MeshRenderer();renderer.rendering_options.resolution=512;renderer.rendering_options.ssaa=1
    renderer.rendering_options.near=.001;renderer.rendering_options.far=10
    renderer.rendering_options.antialias=False
    images=[];targets=[];rows=[]
    for i,r in enumerate(plan['selected_records']):
        g=r['source_geometry_record'];assert sha(g['source_geometry'])==g['sha256'] and sha(g['source_mesh'])==g['source_mesh_sha256']
        with np.load(g['source_geometry']) as z:vertices=z['vertices'].copy();faces=z['faces'].copy()
        with np.load(g['source_mesh']) as z:
            sourcefaces=z['faces'][z['face_object_ids']==g['source_object_id']]
            used=np.unique(sourcefaces);colors=z['vertex_colors'][used].astype(np.float32)
        assert len(colors)==len(vertices)
        if colors.max()>1:colors/=255
        mesh=Mesh(torch.tensor(vertices,device='cuda'),torch.tensor(faces,device='cuda'),torch.tensor(colors,device='cuda'))
        K,w2c=camera_matrices([1.,-1.,.7],[0,0,-.15],[0,0,1],43)
        with torch.no_grad():view=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),
            torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['attr','mask'])
        mask=view['mask'].cpu().numpy();rgb=view['attr'][:3].permute(1,2,0).cpu().numpy()*mask[...,None]
        full=Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8));full.save(root/f'source_object{i}.png')
        y,x=np.where(mask>.8);assert len(x)>100
        cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2;side=max(x.max()-x.min(),y.max()-y.min())
        im=full.crop((int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))).resize((512,512),Image.Resampling.LANCZOS)
        im.save(root/f'official_image_input{i}.png');images.append(im)
        with np.load(r['native_target']) as z:targets.append((z['S0'].copy(),z['support64'][0,0].astype(bool)))
        rows.append({'index':i,'scene':r['scene_id'],'source_asset':r['source_asset_id'],
            'source_geometry_sha256':g['sha256'],'image_source':'fixed corner render of legitimate authored TRAIN source mesh',
            'standard_image_preprocessing':'foreground square crop, black alpha-composited background; no geometry modification',
            'diagnostic_only_not_map_only_inference':True})
    encoder=DinoV3FeatureExtractor(base.DINO,image_size=512);encoder.cuda()
    with torch.no_grad():features=[encoder([im]).detach() for im in images]
    encoder.cpu();del encoder;torch.cuda.empty_cache()
    native=base.native_model()
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=plan['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args']);tiles=[]
    noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(plan['generation_seeds'][0])).cuda()
    for i,(x0,gt) in enumerate(targets):
        with torch.no_grad():
            reconstruction=(decoder(torch.from_numpy(x0).cuda())>0)[0,0].cpu().numpy()
            latent=sampler.sample(native,noise.clone(),cond=features[i],neg_cond=torch.zeros_like(features[i]),**cfg['params'],verbose=False).samples
            support=(decoder(latent)>0)[0,0].cpu().numpy()
        coords=np.argwhere(support);extent=(coords.max(0)-coords.min(0)+1).tolist() if len(coords) else [0,0,0]
        rows[i].update({'native_SS_reconstruction_IoU':iou(reconstruction,gt),'image_conditioned_support_cells':int(support.sum()),
            'image_conditioned_extent_cells':extent,'image_generation_seed':plan['generation_seeds'][0],
            'image_prior_not_map_pose_control':'official asset normalization differs; no target-bbox alignment applied'})
        with (root/f'IMAGE_PRIOR_SS_{i}.npz').open('xb') as f:np.savez_compressed(f,latent=latent.cpu().float().numpy(),native_support64=support)
        tiles.append(tile(f"{i} IMAGE PRIOR {int(support.sum())} cells; SS codec IoU {iou(reconstruction,gt):.3f}",support,gt,np.zeros((64,64),bool)))
        print('NATIVE_PRIOR_CHECK',rows[i],flush=True)
    sheet=Image.new('RGB',(1536,846),'black')
    for i,t in enumerate(tiles):sheet.paste(t,((i%2)*768,(i//2)*282))
    sheet.save(root/'PRIOR_CONTACT.png')
    save(root/'RESULTS.json',{'rows':rows,'native_decoder_original':True,'no_material_generation':True,
        'full_3D_mesh_reconstruction_NOT_claimed':'this is native Structure support codec and image-conditioned Structure generation',
        'pure_noise':True,'source_target_inputs_diagnostic_only':True})
    assert all(r['native_SS_reconstruction_IoU']>.999 for r in rows)
    assert all(r['image_conditioned_support_cells']>0 and min(r['image_conditioned_extent_cells'])>=3 for r in rows)

if __name__=='__main__':main()

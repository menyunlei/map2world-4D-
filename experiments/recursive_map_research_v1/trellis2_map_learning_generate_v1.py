"""Matched native inference, real-map input only; no placement or mesh repair."""
from trellis2_map_learning_train_fix1 import base
from paths import JOB, read, save, sha
from trellis2_map_learning_prepare_fix1 import OUT
from trellis2_prepare_v1 import DEST as INSTALL, REPO
from trellis2_map_object_module_v1 import Trellis2MapObjectConditioner
import os, sys, argparse, time, random
import numpy as np
import torch
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg


class MapConditionedFlow(torch.nn.Module):
    def __init__(self,native,branch,mapbatch):
        super().__init__()
        self.native=native
        self.branch=branch
        self.reader=base.BoundMapReader(native,branch)
        self.mapbatch=mapbatch
        self.resolution=native.resolution
        self.in_channels=native.in_channels

    def forward(self,x,t,cond,**kw):
        # Match the trained CFG contract: no map on unconditional image-zero pass.
        active=self.mapbatch if bool(torch.count_nonzero(cond)) else None
        return self.reader(x,t,cond,active)


def export_native(mesh,latents,folder):
    vertices=mesh.vertices.detach().float().cpu()
    faces=mesh.faces.detach().cpu()
    assert len(faces)>0 and torch.isfinite(vertices).all()
    payload={'vertices':vertices,'faces':faces,'attrs':mesh.attrs.detach().cpu(),
        'coords':mesh.coords.detach().cpu(),'layout':mesh.layout,'voxel_size':mesh.voxel_size,
        'shape_slat':latents[0].feats.detach().cpu(),'shape_coords':latents[0].coords.detach().cpu(),
        'tex_slat':latents[1].feats.detach().cpu(),'resolution':latents[2]}
    raw=folder/'RAW_NATIVE.pt';assert not raw.exists();torch.save(payload,raw)
    rgba=[]
    for start in range(0,len(mesh.vertices),250000):
        attrs=mesh.query_attrs(mesh.vertices[start:start+250000])
        value=torch.cat([attrs[:,mesh.layout['base_color']],attrs[:,mesh.layout['alpha']]],-1)
        rgba.append((value.clamp(0,1)*255).byte().cpu().numpy())
    matrix=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
    native_v=vertices.numpy();v=native_v@matrix.T;f=faces.numpy()
    colored=trimesh.Trimesh(vertices=v,faces=f,vertex_colors=np.concatenate(rgba),process=False)
    path=folder/'world_basecolor.glb';assert not path.exists();trimesh.Scene(colored).export(path)
    neutral=folder/'generated_geometry.glb'
    trimesh.Scene(trimesh.Trimesh(vertices=v,faces=f,process=False)).export(neutral)
    fresh=trimesh.load(path,force='scene',process=False);g=list(fresh.geometry.values())[0]
    assert len(g.faces)==len(f) and len(g.vertices)==len(v) and np.max(abs(g.vertices-v))<1e-6
    save(folder/'ASSET.json',{'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
        'vertices':len(v),'faces':len(f),'geometry_and_appearance_model_generated':True,
        'source':'actual native TRELLIS.2 output, no procedural placement, pruning or added geometry',
        'independent_parser_pass':True,'original_geometry_preserved':True,
        'appearance_export':'unchanged-vertex basecolor/alpha preview; complete native O-Voxel PBR retained in RAW_NATIVE.pt',
        'native_decoder_standard_postprocessing':'unchanged official decode_latent fill_holes retained',
        'normalization':'unit cube, +Z up; one fixed glTF axis conversion only; no per-object transforms',
        'human_review':'PENDING','real_geographic_accuracy_verified':False})
    print('ACTUAL_NATIVE_ASSET_SAVED',folder.name,len(v),len(f),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[953101,953102],required=True);a=ap.parse_args()
    plan=read(OUT/'PLAN.json');assert (OUT/'real_input/MAP_INPUT.json').exists()
    meta=read(OUT/f'seed{a.seed}/CHECKPOINT_000100.json')
    assert sha(meta['path'])==meta['sha256']
    ckpt=torch.load(meta['path'],map_location='cpu',weights_only=False)
    assert ckpt['updates']==100 and ckpt['plan_sha256']==sha(OUT/'PLAN.json') and not ckpt['map10_TRAIN']
    rembg.UnusedBackgroundRemoval=lambda:None
    oldruntime=JOB/'map_object_world_v1/trellis2_resume_v3/runtime'
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(oldruntime))
    pipeline.cuda()
    branch=Trellis2MapObjectConditioner()
    branch.load_state_dict(ckpt['branch_state'],strict=True)
    branch.eval().requires_grad_(False)
    wrapped=MapConditionedFlow(pipeline.models['sparse_structure_flow_model'],branch,None)
    pipeline.models['sparse_structure_flow_model']=wrapped
    modes=[('baseline',None),('correct','map.npz'),('shuffled','shuffled_map.npz'),('edit','removed_last_building_map.npz')]
    inference_seed=953201+(a.seed-953101)
    save(OUT/f'seed{a.seed}/GENERATION_START.json',{'checkpoint':meta,'generation_seed':inference_seed,
        'pipeline':'1024_cascade','sampler_parameters':{'ss':pipeline.sparse_structure_sampler_params,
        'shape':pipeline.shape_slat_sampler_params,'texture':pipeline.tex_slat_sampler_params},
        'modes':[n for n,_ in modes],'appearance_sha256':plan['appearance_sha256'],
        'geometry_queries':'learned map conditions act inside shared native world SS denoising, not mesh placement',
        'map10_not_TRAIN':True,'real_map_input_limit':'oblique roof proxies, normalized layout; not measured ground footprints',
        'precision':{k:sorted({str(p.dtype) for p in m.parameters()}) for k,m in pipeline.models.items()}})
    for mode,name in modes:
        folder=OUT/f'seed{a.seed}/assets'/mode;folder.mkdir(parents=True,exist_ok=False)
        if name:
            with np.load(OUT/'real_input'/name) as z:
                wrapped.mapbatch={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
        else:wrapped.mapbatch=None
        random.seed(inference_seed);np.random.seed(inference_seed);torch.manual_seed(inference_seed)
        start=time.time()
        image=Image.open(plan['appearance_condition']).convert('RGB')
        with torch.no_grad():
            meshes,latents=pipeline.run(image,seed=inference_seed,pipeline_type='1024_cascade',
                preprocess_image=False,return_latent=True,max_num_tokens=49152)
            export_native(meshes[0],latents,folder)
        save(folder/'GENERATION.json',{'mode':mode,'training_seed':a.seed,'generation_seed':inference_seed,
            'seconds':time.time()-start,'actual_resolution':latents[2],
            'appearance_reference':'official stylized building, not recovered real facades',
            'same_weights_noise_image_and_sampler_except_map_graph':mode!='baseline',
            'baseline':'frozen prior with branch disabled' if mode=='baseline' else None,
            'map_input':name,'source_map':'10.jpg','map10_not_training':True})
        del meshes,latents;torch.cuda.empty_cache()
    save(OUT/f'seed{a.seed}/GENERATION_COMPLETE.json',{'native_assets':4,'training_updates':100,
        'manual_postplacement_or_repair':False,'quality_acceptance_pending':True})


if __name__=='__main__':main()

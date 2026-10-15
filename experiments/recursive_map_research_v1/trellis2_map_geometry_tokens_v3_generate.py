"""Native world generation with learned map tokens in EVERY geometry stage."""
from trellis2_map_learning_generate_v1 import (base,Trellis2ImageTo3DPipeline,rembg,export_native)
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_geometry_tokens_v3_prepare import OUT
from trellis2_map_object_module_v1 import native_world_grid
from paths import JOB,read,save,sha
from trellis2_map_geometry_entry_v2_generate import digest_tensor
import argparse
import random
import time
import numpy as np
import torch
from PIL import Image


class AddressedNoise:
    def __init__(self,sampler,seed,stage):
        self.sampler,self.seed,self.stage=sampler,seed,stage
        self.grids={}
    def sample(self,model,noise,*args,**kwargs):
        coords=noise.coords.cpu().long()
        assert bool((coords[:,0]==0).all()) and bool(((coords[:,1:]>=0)&(coords[:,1:]<64)).all())
        channels=noise.feats.shape[-1]
        if channels not in self.grids:
            self.grids[channels]=torch.randn((64,64,64,channels),generator=torch.Generator().manual_seed(self.seed+200000+self.stage*1000))
        features=self.grids[channels][coords[:,1],coords[:,2],coords[:,3]].to(noise.device,dtype=noise.feats.dtype)
        return self.sampler.sample(model,noise.replace(features),*args,**kwargs)


class MapSSFlow(torch.nn.Module):
    def __init__(self,native,branch):
        super().__init__()
        self.native,self.branch=native,branch
        self.reader=base.BoundMapReader(native,branch)
        self.mapbatch=None
        self.resolution,self.in_channels=native.resolution,native.in_channels
        self.audit=[]
    def forward(self,x,t,cond,**kwargs):
        features=cond['features']
        active=cond['mapped']
        if len(self.audit)<128:
            self.audit.append({'map_residual_active':active,'condition_source':cond['source'],
                               'native_geometry_tokens':features.shape[1]})
        return self.reader(x,t,features,self.mapbatch if active else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['original','separated'],required=True)
    parser.add_argument('--seed',type=int,choices=[953101,953102],required=True)
    args=parser.parse_args()
    plan=read(OUT/'PLAN.json')
    checkpoint_record=read(OUT/args.arm/f'seed{args.seed}/CHECKPOINT_000300.json')
    assert sha(checkpoint_record['path'])==checkpoint_record['sha256']
    checkpoint=torch.load(checkpoint_record['path'],map_location='cpu',weights_only=False)
    assert checkpoint['updates']==300 and checkpoint['map10_TRAIN'] is False
    rembg.UnusedBackgroundRemoval=lambda:None
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'))
    pipeline.cuda()
    branch=Trellis2MapGeometryTokenConditioner().eval().requires_grad_(False)
    branch.load_state_dict(checkpoint['branch_state'],strict=True)
    ss=MapSSFlow(pipeline.models['sparse_structure_flow_model'],branch)
    pipeline.models['sparse_structure_flow_model']=ss
    generation_seed=953201+args.seed-953101
    pipeline.shape_slat_sampler=AddressedNoise(pipeline.shape_slat_sampler,generation_seed,1)
    pipeline.tex_slat_sampler=AddressedNoise(pipeline.tex_slat_sampler,generation_seed,2)
    image=Image.open(plan['appearance_condition']).convert('RGB')
    image512=image1024=None
    if args.arm=='original':
        image512=pipeline.get_cond([image],512)
        image1024=pipeline.get_cond([image],1024)
    modes=[('correct','map.npz'),('edit','explicit_remove_last_building_map.npz')]
    if args.arm=='separated':modes.append(('shuffled','shuffled_map.npz'))
    for mode,filename in modes:
        root=OUT/args.arm/f'seed{args.seed}/assets'/mode
        root.mkdir(parents=True,exist_ok=False)
        with np.load(OUT/'real_input'/filename) as archive:
            ss.mapbatch={key:torch.from_numpy(archive[key].copy()).unsqueeze(0).cuda() for key in archive.files}
        branch.cuda()
        with torch.no_grad():
            context=branch.prepare(ss.mapbatch,native_world_grid(device='cuda'))
            kind='reference' if args.arm=='original' else 'separated'
            geometry512=branch(context,None if image512 is None else image512['cond'],arm=kind)
            geometry1024=branch(context,None if image1024 is None else image1024['cond'],arm=kind)
        source='reference_plus_map_tokens' if args.arm=='original' else 'learned_map_tokens_only'
        positive={'features':geometry512.features,'mapped':True,'source':source}
        negative={'features':torch.zeros_like(geometry512.features),'mapped':False,'source':'unconditional_zeros'}
        ss_condition={'cond':positive,'neg_cond':negative}
        fine_condition={'cond':geometry1024.features,'neg_cond':torch.zeros_like(geometry1024.features)}
        if args.arm=='separated':
            assert geometry512.features.shape==(1,256,1024)
            assert geometry1024.features.shape==(1,256,1024)
            assert image512 is None  # No reference encoder/pixels consumed for geometry.
        random.seed(generation_seed);np.random.seed(generation_seed);torch.manual_seed(generation_seed)
        ss.audit=[];start=time.time();trace={}
        def capture(_module,arguments,output):
            trace['latent']=arguments[0].float().cpu().numpy()
            trace['native_support64']=(output>0).cpu().numpy()
        hook=pipeline.models['sparse_structure_decoder'].register_forward_hook(capture)
        with torch.no_grad():
            coords=pipeline.sample_sparse_structure(ss_condition,64)
            hook.remove()
            with (root/'SS_GENERATED.npz').open('xb') as handle:
                np.savez_compressed(handle,**trace,native_coords64=coords.cpu().numpy())
            if not len(coords):
                save(root/'EMPTY_NATIVE_SUPPORT.json',{'native_support_cells':0,'mode':mode,
                     'native_asset':None,'no_fake_mesh_or_retry':True,'quality_verdict':'FAIL'})
                print('EMPTY_MODEL_SUPPORT_PRESERVED',args.arm,args.seed,mode,flush=True)
                continue
            shape=pipeline.sample_shape_slat(fine_condition,pipeline.models['shape_slat_flow_model_1024'],coords)
            shape_digest={'coarse_support_sha256':digest_tensor(coords),'shape_features_sha256':digest_tensor(shape.feats.float()),
                          'shape_coords_sha256':digest_tensor(shape.coords)}
            # In separated mode the reference-image encoder is called ONLY now,
            # after ALL geometry has been sampled. This cannot steer geometry.
            appearance=pipeline.get_cond([image],1024) if args.arm=='separated' else image1024
            texture=pipeline.sample_tex_slat(appearance,pipeline.models['tex_slat_flow_model_1024'],shape)
            meshes=pipeline.decode_latent(shape,texture,1024)
            export_native(meshes[0],(shape,texture,1024),root)
        save(root/'GENERATION.json',{'arm':args.arm,'seed':args.seed,'mode':mode,'additional_updates':300,
             'generation_seed':generation_seed,'seconds':time.time()-start,'pipeline':'official_direct_1024',
             'geometry_condition_source':source,'map10_TRAIN':False,
             'reference_encoder_first_used_after_geometry':args.arm=='separated',
             'fine_geometry_receives_learned_map_tokens':True,'shape_digest':shape_digest,
             'SS_CFG_calls':ss.audit,'model_support_cells':len(coords),
             'map_input':filename,'map_input_sha256':sha(OUT/'real_input'/filename),
             'samplers':{'SS':pipeline.sparse_structure_sampler_params,'shape':pipeline.shape_slat_sampler_params,
                          'texture':pipeline.tex_slat_sampler_params},
             'matched_world_address_noise':True,'teacher_coords_or_target_mesh_in_inference':False,
             'not_geographically_registered_or_real_facade_reconstruction':True,
             'appearance_reference':'same fixed stylized building, not recovered real facades'})
        print('SHARED_GEOMETRY_MAP_ASSET',args.arm,args.seed,mode,len(coords),flush=True)
        del shape,texture,meshes,coords
        torch.cuda.empty_cache()
    save(OUT/args.arm/f'seed{args.seed}/GENERATION_COMPLETE.json',{'fixed_modes':[m for m,_ in modes],
         'empty_native_outputs_preserved_as_failures':True,'no_postplacement_or_geometry_repair':True})


if __name__=='__main__':main()

"""Decode the registered condition-interface test through unchanged native stages."""
from trellis2_map_geometry_tokens_v3_generate import AddressedNoise
from trellis2_map_learning_generate_v1 import base,Trellis2ImageTo3DPipeline,rembg,export_native
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid
from paths import JOB,read,save,sha
from torch.nn import functional as F
import argparse,numpy as np,torch,time
from PIL import Image
OUT=JOB/'map_object_world_v1/trellis2_map_condition_contract_v4'
OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=['correct','edit'],required=True);a=p.parse_args()
    torch.set_num_threads(4);rembg.UnusedBackgroundRemoval=lambda:None
    plan=read(OLD/'PLAN.json');source=OUT/'native_norm'/f'seed{a.seed}'/('map10_'+a.mode)/'SS_GENERATED.npz'
    directory=OUT/'separated'/f'seed{a.seed}'/'assets'/a.mode;directory.mkdir(parents=True,exist_ok=False)
    with np.load(source) as f:coords=torch.from_numpy(f['coords'].copy()).cuda().int()
    if not len(coords):
        save(directory/'EMPTY_NATIVE_SUPPORT.json',{'quality':'FAIL','native_asset':None,'source':str(source),'no_retry':True})
        return
    checkpoint=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    pipe=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'));pipe.cuda()
    branch=Trellis2MapGeometryTokenConditioner().cuda().eval().requires_grad_(False)
    branch.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['branch_state'])
    filename='map.npz' if a.mode=='correct' else 'explicit_remove_last_building_map.npz'
    with np.load(OLD/'real_input'/filename) as f:m={k:torch.from_numpy(f[k].copy()).unsqueeze(0).cuda() for k in f.files}
    with torch.no_grad():
        raw=branch(branch.prepare(m,native_world_grid(device='cuda')),arm='separated').features
        condition=F.layer_norm(raw,raw.shape[-1:])
    fine={'cond':condition,'neg_cond':torch.zeros_like(condition)}
    del branch
    generation_seed=953201+a.seed-953101
    pipe.shape_slat_sampler=AddressedNoise(pipe.shape_slat_sampler,generation_seed,1)
    pipe.tex_slat_sampler=AddressedNoise(pipe.tex_slat_sampler,generation_seed,2)
    torch.manual_seed(generation_seed);start=time.time()
    with torch.no_grad():
        shape=pipe.sample_shape_slat(fine,pipe.models['shape_slat_flow_model_1024'],coords)
        # Appearance remains strictly downstream of geometry.
        appearance=pipe.get_cond([Image.open(plan['appearance_condition']).convert('RGB')],1024)
        texture=pipe.sample_tex_slat(appearance,pipe.models['tex_slat_flow_model_1024'],shape)
        mesh=pipe.decode_latent(shape,texture,1024)[0]
        export_native(mesh,(shape,texture,1024),directory)
    save(directory/'GENERATION.json',{'seed':a.seed,'mode':a.mode,'optimizer_updates_this_test':0,
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'condition_fix':'native feature LayerNorm for geometry conditions only',
        'SS_source':str(source),'SS_source_sha256':sha(source),'no_SS_resampling':True,
        'samplers_unchanged':True,'world_addressed_noise_matched':True,'reference_image_geometry_condition':False,
        'seconds':time.time()-start,'same_unrepaired_native_decoder':True,'real_map10_not_TRAIN':True})

if __name__=='__main__':main()

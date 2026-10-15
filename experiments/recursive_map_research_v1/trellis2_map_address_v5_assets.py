"""Complete native assets for the single address-validity correction."""
from trellis2_map_geometry_tokens_v3_generate import AddressedNoise
from trellis2_map_learning_generate_v1 import Trellis2ImageTo3DPipeline,rembg,export_native
from trellis2_map_address_validity_v5 import AddressValidMapConditioner
from trellis2_map_object_module_v1 import native_world_grid
from paths import JOB,read,save,sha
import argparse,numpy as np,torch,time
from PIL import Image
OUT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5';OLD=JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3'
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['correct','edit'],required=True);a=p.parse_args()
    torch.set_num_threads(4);rembg.UnusedBackgroundRemoval=lambda:None
    source=OUT/'address_valid'/f'seed{a.seed}'/'support'/a.mode/'SS_GENERATED.npz'
    folder=OUT/'separated'/f'seed{a.seed}'/'assets'/a.mode;folder.mkdir(parents=True,exist_ok=False)
    with np.load(source) as f:coords=torch.from_numpy(f['native_coords64'].copy()).cuda().int()
    if not len(coords):save(folder/'EMPTY_NATIVE_SUPPORT.json',{'native_asset':None,'failure':'EMPTY_SUPPORT','no_retry':True});return
    checkpoint=OLD/'separated'/f'seed{a.seed}'/'branch_000300.pt'
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'));pipeline.cuda()
    branch=AddressValidMapConditioner().cuda().eval().requires_grad_(False)
    branch.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['branch_state'])
    filename='map.npz' if a.mode=='correct' else 'explicit_remove_last_building_map.npz'
    with np.load(OLD/'real_input'/filename) as f:m={k:torch.from_numpy(f[k].copy()).unsqueeze(0).cuda() for k in f.files}
    with torch.no_grad():features=branch(branch.prepare(m,native_world_grid(device='cuda')),arm='separated').features
    fine={'cond':features,'neg_cond':torch.zeros_like(features)};del branch
    gen_seed=953201+a.seed-953101
    pipeline.shape_slat_sampler=AddressedNoise(pipeline.shape_slat_sampler,gen_seed,1)
    pipeline.tex_slat_sampler=AddressedNoise(pipeline.tex_slat_sampler,gen_seed,2)
    torch.manual_seed(gen_seed);start=time.time();plan=read(OLD/'PLAN.json')
    with torch.no_grad():
        shape=pipeline.sample_shape_slat(fine,pipeline.models['shape_slat_flow_model_1024'],coords)
        appearance=pipeline.get_cond([Image.open(plan['appearance_condition']).convert('RGB')],1024)
        texture=pipeline.sample_tex_slat(appearance,pipeline.models['tex_slat_flow_model_1024'],shape)
        mesh=pipeline.decode_latent(shape,texture,1024)[0];export_native(mesh,(shape,texture,1024),folder)
    save(folder/'GENERATION.json',{'seed':a.seed,'mode':a.mode,'generation_seed':gen_seed,'optimizer_updates_this_fix':0,
        'same_existing_learned_checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'single_fix':'known content confidence does not invalidate world query XYZ',
        'known_masks_not_changed_or_fabricated':True,'reference_for_texture_only':True,'no_normalization_change':True,
        'samplers_and_world_addressed_noise_unchanged':True,'actual_support_source_sha256':sha(source),
        'raw_G2B_and_v3_negative_outputs_preserved':True,'seconds':time.time()-start,'no_manual_geometry_repair':True})
if __name__=='__main__':main()

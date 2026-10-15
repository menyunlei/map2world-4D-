"""Fixed pure-noise curves and original/edit meshes; no optimizer or materials."""
from paths import JOB,read,save,sha
from trellis2_coverage_v12_prepare import OUT,ARMS,SEEDS
import argparse,time,gc,numpy as np

GROUPS=[('parent',0),(ARMS[0],1000),(ARMS[1],1000),
        (ARMS[0],100),(ARMS[1],100),(ARMS[0],300),(ARMS[1],300)]

def folder(seed,arm,update):return OUT/'evaluation'/f'seed{seed}'/arm/f'update{update:06d}'
def case(root,noise,mode,index):return root/f'noise{noise}'/mode/f'object{index:03d}'
def ratio(n,d):return float(n/max(1,d))
def iou(a,b):return ratio((a&b).sum(),(a|b).sum())

def evaluate(seed):
    import torch
    from scipy.ndimage import label
    from trellis2_coverage_v12_train import setup,train
    from trellis2_map_geometry_tokens_v3_generate import MapSSFlow
    from trellis2_map_object_module_v1 import native_world_grid
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2 import models
    from trellis2_map_learning_train_fix1 import base
    from trellis2_structure_learnability_v8_eval import tile
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    plan,native,branch,reader,_=setup(ARMS[1],seed)
    branch.eval().requires_grad_(False);native.eval().requires_grad_(False)
    wrapper=MapSSFlow(native,branch)
    decoder=models.from_pretrained(str(base.INSTALL/'weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')).cuda().eval().requires_grad_(False)
    cfg=plan['evaluation']['sampler'];sampler=FlowEulerGuidanceIntervalSampler(**cfg['args'])
    records=plan['evaluation_records'];edits=plan['edit_records']
    maps={mode:[train.load_record(r) for r in rs] for mode,rs in [('correct',records),('edit',edits)]}
    fps={mode:[m['footprints'][0,int(m['target_object'])].cpu().numpy()>.5 for m in ms] for mode,ms in maps.items()}
    targets=[]
    for r in records:
        assert sha(r['native_target'])==r['native_target_sha256']
        with np.load(r['native_target']) as z:targets.append(z['support64'][0,0].astype(bool))
    for arm,update in GROUPS:
        root=folder(seed,arm,update);root.mkdir(parents=True,exist_ok=False)
        ck=plan['parent_checkpoints'][str(seed)] if not update else read(OUT/'small'/arm/f'seed{seed}/train'/f'CHECKPOINT_{update:06d}.json')
        assert sha(ck['path'])==ck['sha256']
        state=torch.load(ck['path'],map_location='cpu',weights_only=False)
        branch.load_state_dict(state['branch_state'],strict=True);train.load_lora_state_dict(native,state['lora_state']);del state
        rows=[];edit_rows=[];mismatch=[]
        for gen in plan['generation_seeds']:
            noise=torch.randn((1,8,16,16,16),generator=torch.Generator().manual_seed(gen)).cuda()
            generated={};modes=['correct','edit'] if update in [0,1000] else ['correct']
            for mode in modes:
                for index,r in enumerate(records):
                    start=time.time();m=maps[mode][index];wrapper.mapbatch=m
                    with torch.no_grad():
                        features=branch(branch.prepare(m,native_world_grid(device='cuda'))).features
                        latent=sampler.sample(wrapper,noise.clone(),cond={'features':features,'mapped':True,'source':'trained_map_only'},
                          neg_cond={'features':torch.zeros_like(features),'mapped':False,'source':'native_zero_negative'},**cfg['params'],verbose=False).samples
                        support=(decoder(latent)>0)[0,0].cpu().numpy()
                    d=case(root,gen,mode,index);d.mkdir(parents=True,exist_ok=False)
                    with (d/'SS_GENERATED.npz').open('xb') as h:np.savez_compressed(h,latent=latent.float().cpu().numpy(),native_support64=support)
                    xy=support.any(2).T;fp=fps[mode][index];coords=np.argwhere(support)
                    extent=(np.ptp(coords,axis=0)+1).tolist() if len(coords) else [0,0,0]
                    components,n=label(support);counts=np.bincount(components.ravel())[1:]
                    row={'seed':seed,'arm':arm,'update':update,'index':index,'noise':gen,'mode':mode,
                      'family':r['family_id'],'object':r['scene_id']+':'+str(r['object_id']),
                      'group':'original_six' if index<6 else 'added_TRAIN' if index<24 else 'heldout',
                      'source_asset_id':r['source_asset_id'],'pure_noise':True,'pose':r['frame'],
                      'cells':int(support.sum()),'extent_cells':extent,'empty':not bool(support.any()),
                      'axes_below_three_cells':int(sum(x<3 for x in extent)),
                      'components_6connected':int(n),'largest_component_fraction':ratio(counts.max() if len(counts) else 0,support.sum()),
                      'map_projection_IoU':iou(xy,fp),'required_footprint_coverage':ratio((xy&fp).sum(),fp.sum()),
                      'outside_footprint_fraction':ratio((xy&~fp).sum(),xy.sum()),
                      'native_target_IoU_diagnostic':iou(support,targets[index]),
                      'native_target_used_by_generation':False,'seconds':time.time()-start}
                    save(d/'METRICS.json',row);rows.append(row);generated[mode,index]=support
                    tile(f'{arm} +{update} {mode} object{index} XY={row["map_projection_IoU"]:.3f}',support,targets[index],fp).save(d/'STRUCTURE.png')
                    print('PURE_NOISE',seed,arm,update,gen,mode,index,row['map_projection_IoU'],row['cells'],flush=True)
            for index,r in enumerate(records):
                # Same coordinate-indexed SS noise; no extra draw for mismatched map.
                other=(index+1)%len(records);a=generated['correct',index];b=generated['correct',other]
                mismatch.append({'index':index,'family':r['family_id'],'noise':gen,'other_index':other,
                  'correct_map_IoU':iou(a.any(2).T,fps['correct'][index]),
                  'mismatched_map_IoU':iou(b.any(2).T,fps['correct'][index]),
                  'support_change_fraction':ratio((a^b).sum(),(a|b).sum()),
                  'same_B_output_reinterpreted_at_A_pose':True,'not_an_attention_shuffle_test':True})
                if 'edit' not in modes:continue
                b=generated['edit',index];xy0=a.any(2).T;xy1=b.any(2).T
                fp0=fps['correct'][index];fp1=fps['edit'][index];removed=fp0&~fp1;protected=fp0&fp1
                edit_rows.append({'index':index,'family':r['family_id'],'noise':gen,
                  'group':'original_six' if index<6 else 'added_TRAIN' if index<24 else 'heldout',
                  'removed_pixels':int(removed.sum()),'edit_visible_at_actual_input':bool(removed.any()),
                  'removed_occupancy_before':ratio((xy0&removed).sum(),removed.sum()),
                  'removed_occupancy_after':ratio((xy1&removed).sum(),removed.sum()),
                  'retained_coverage_before':ratio((xy0&protected).sum(),protected.sum()),
                  'retained_coverage_after':ratio((xy1&protected).sum(),protected.sum()),
                  'protected_Q_change_fraction':ratio(((a^b)&protected.T[:,:,None]).sum(),((a|b)&protected.T[:,:,None]).sum()),
                  'Q_z_extents_before_after':[[int(np.where(x)[2].min()),int(np.where(x)[2].max())] if x.any() else None for x in [a,b]],
                  'object_pose_unchanged':True,'other_generated_objects_not_cached_as_preservation_evidence':True,
                  'full_neighbor_regeneration_not_measured':True})
        save(root/'RESULTS.json',{'seed':seed,'arm':arm,'additional_updates':update,'checkpoint':ck,
          'rows':rows,'edit_rows':edit_rows,'mismatched_maps':mismatch,'sampler':cfg,
          'no_materials':True,'no_Map10':True,'exact_target_IoU_is_separate_diagnostic':True,
          'edit_sampling_at_baseline_and_final_only':True,'geometry_extents_not_mesh_acceptance':True})
    save(OUT/'evaluation'/f'seed{seed}/STRUCTURE_COMPLETE.json',{'groups':GROUPS,'no_training':True})

def mesh(seed):
    import torch
    from trellis2_map_learning_train_fix1 import base
    from trellis2_map_object_contract_v6 import ObjectContractConditioner
    from trellis2_map_object_module_v1 import native_world_grid
    from trellis2_map_geometry_tokens_v3_generate import AddressedNoise
    from trellis2_map_edit_mesh_v9_run import export
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2 import models
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True;plan=read(OUT/'PLAN.json')
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    ck=plan['Shape_condition_checkpoints'][str(seed)];assert sha(ck['path'])==ck['sha256']
    fine.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['branch_state'],strict=True)
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    scfg=cfg['shape_slat_sampler'];pipe=Trellis2ImageTo3DPipeline(models=mods,
      shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**scfg['args']),plan['shape_address_seed'],1),
      shape_slat_sampler_params=scfg['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True)
    pipe.cuda();gen=plan['generation_seeds'][0]
    # Same fixed case order for every arm; no selection based on scores.
    for index in plan['mesh_indices']:
        for arm,update in GROUPS[:3]:
            for mode in ['correct','edit']:
                root=folder(seed,arm,update);d=case(root,gen,mode,index)
                r=plan['edit_records' if mode=='edit' else 'evaluation_records'][index]
                with np.load(d/'SS_GENERATED.npz') as z:support=z['native_support64'].copy()
                if not support.any():save(d/'EMPTY_MESH.json',{'no_retry':True,'native_Q_empty':True});continue
                assert sha(r['map_input'])==r['map_sha256']
                with np.load(r['map_input']) as z:m={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
                with torch.no_grad():
                    features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
                    coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda().contiguous()
                    torch.manual_seed(plan['shape_address_seed'])
                    shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
                    meshes,subs=pipe.decode_shape_slat(shape,1024)
                    if len(meshes[0].faces)==0:save(d/'EMPTY_MESH.json',{'no_retry':True,'decoder_empty_faces':True})
                    else:export(meshes[0],shape,d,{'seed':seed,'arm':arm,'updates':update,'index':index,'mode':mode,
                      'frame':r['frame'],'shape_condition_checkpoint':ck,'shape_sampler':scfg,'shape_noise':plan['shape_address_seed'],
                      'Shape_confidence_contract':'unchanged original v6 raw confidence','input_Q_sha256':sha(d/'SS_GENERATED.npz')})
                    del meshes,subs,shape
                gc.collect();torch.cuda.empty_cache()
    save(OUT/'evaluation'/f'seed{seed}/MESH_COMPLETE.json',{'fixed_indices':plan['mesh_indices'],'groups':GROUPS[:3]})

def views(seed):
    import torch
    from trellis2_map_edit_mesh_v9_views import load_mesh,footprint,camera_matrices,MeshRenderer
    from PIL import Image,ImageDraw
    torch.set_num_threads(4);plan=read(OUT/'PLAN.json');gen=plan['generation_seeds'][0]
    cameras=read(JOB/'map_object_world_v1/trellis2_map_edit_mesh_v9/PLAN.json')['cameras']
    renderer=MeshRenderer();opt=renderer.rendering_options
    opt.resolution=384;opt.ssaa=1;opt.near=.001;opt.far=10;opt.antialias=False;opt.chunk_size=8000000
    results=[]
    for index in plan['mesh_indices']:
        sheet=Image.new('RGB',(6*256,6*280),'#121922')
        for group,(arm,update) in enumerate(GROUPS[:3]):
            for mi,mode in enumerate(['correct','edit']):
                d=case(folder(seed,arm,update),gen,mode,index);rowid=group*2+mi
                ImageDraw.Draw(sheet).text((5,rowid*280+5),f'{arm} +{update} {mode} index{index}',fill='white')
                if not (d/'ASSET.json').exists():
                    ImageDraw.Draw(sheet).text((10,rowid*280+60),'EMPTY / NOT AVAILABLE -- not replaced',fill='red');continue
                mesh,v,f,meta=load_mesh(d);vd=d/'views';vd.mkdir(exist_ok=False)
                r=plan['edit_records' if mode=='edit' else 'evaluation_records'][index]
                with np.load(r['map_input']) as z:fp=z['footprints'][r['target_object']]>.5
                projected=footprint(v,f);overlay=np.zeros((64,64,3),np.uint8)+20
                overlay[projected]=[130,145,160];overlay[fp]=[220,60,55];overlay[projected&fp]=[230,205,90]
                image=Image.fromarray(overlay[::-1]).resize((384,384),Image.Resampling.NEAREST)
                image.save(vd/'map_overlay.png');sheet.paste(image.resize((256,256)),(5*256,rowid*280+24))
                selected=['front','lateral_left','lateral_right','top','oblique']
                for name,eye,look,up,fov in cameras:
                    K,w2c=camera_matrices(eye,look,up,fov)
                    with torch.no_grad():p=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['normal','depth','mask'])
                    mask=p['mask'].cpu().numpy();n=p['normal'].permute(1,2,0).cpu().numpy()*2-1
                    light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
                    shade=.28+.72*np.abs((n*light).sum(-1)).clip(0,1)
                    rgb=np.array([.76,.79,.82])*shade[...,None]*mask[...,None]+np.array([.075,.09,.12])*(1-mask[...,None])
                    image=Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8));image.save(vd/f'{name}.png')
                    if name in selected:sheet.paste(image.resize((256,256)),(selected.index(name)*256,rowid*280+24))
                row={'seed':seed,'arm':arm,'update':update,'index':index,'mode':mode,'family':r['family_id'],
                  'mesh_map_IoU':iou(projected,fp),'retained_coverage':ratio((projected&fp).sum(),fp.sum()),
                  'outside_fraction':ratio((projected&~fp).sum(),projected.sum()),'vertices':len(v),'faces':len(f),
                  'extent':np.ptp(v,axis=0).tolist(),'watertight':meta['watertight'],'asset_sha256':meta['sha256'],
                  'independent_exact_geometry_reload':True,'neutral_display_no_materials':True}
                results.append(row);save(vd/'RELOAD.json',row)
                del mesh;gc.collect();torch.cuda.empty_cache()
        sheet.save(OUT/'evaluation'/f'seed{seed}'/f'object{index:03d}_MATCHED_MESH.jpg',quality=92)
    save(OUT/'evaluation'/f'seed{seed}/RENDER_COMPLETE.json',{'rows':results,'cameras_reused_from_v9':True,'no_asset_modification':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--mode',choices=['structure','mesh','views'],required=True);a=p.parse_args()
    {'structure':evaluate,'mesh':mesh,'views':views}[a.mode](a.seed)

"""Zero-update same-seed sparse-support replay to locate first observed failure."""
from trellis2_map_learning_generate_v1 import *
from collections import Counter


def run():
    plan=read(OUT/'PLAN.json')
    rembg.UnusedBackgroundRemoval=lambda:None
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(JOB/'map_object_world_v1/trellis2_resume_v3/runtime'))
    pipeline.cuda()
    # The EXACT same image feature as normal inference, not an Oracle condition.
    image=Image.open(plan['appearance_condition']).convert('RGB')
    cond=pipeline.get_cond([image],512)
    native=pipeline.models['sparse_structure_flow_model']
    with np.load(OUT/'real_input/map.npz') as z:
        expected=np.any(z['footprints']>0,axis=0).reshape(32,2,32,2).any((1,3)).T
    records=[]
    for seed in [953101,953102]:
        ckpt=torch.load(read(OUT/f'seed{seed}/CHECKPOINT_000100.json')['path'],map_location='cpu',weights_only=False)
        branch=Trellis2MapObjectConditioner();branch.load_state_dict(ckpt['branch_state']);branch.eval().requires_grad_(False)
        wrapped=MapConditionedFlow(native,branch,None)
        pipeline.models['sparse_structure_flow_model']=wrapped
        masks={}
        for mode,file in [('baseline',None),('correct','map.npz'),('shuffled','shuffled_map.npz'),('edit','removed_last_building_map.npz')]:
            if file:
                with np.load(OUT/'real_input'/file) as z:
                    wrapped.mapbatch={k:torch.from_numpy(z[k].copy()).unsqueeze(0).cuda() for k in z.files}
            else:wrapped.mapbatch=None
            generation_seed=953201+seed-953101
            torch.manual_seed(generation_seed)
            with torch.no_grad():coords=pipeline.sample_sparse_structure(cond,32)
            array=coords.cpu().numpy()
            support=np.zeros((32,32,32),bool);support[array[:,1],array[:,2],array[:,3]]=True
            # This is a coarse support/proxy diagnostic, NOT final mesh occupancy.
            above_floor=support[:,:,5:].any(2)
            intersection=(above_floor&expected).sum();union=(above_floor|expected).sum()
            masks[mode]=support
            path=OUT/f'seed{seed}/SS_TRACE_{mode}.npz'
            assert not path.exists()
            np.savez_compressed(path,coords=array,support=support,above_floor_xy=above_floor,input_roof_proxy_xy=expected)
            records.append({'training_seed':seed,'mode':mode,'generation_seed':generation_seed,
                'SS_support_cells':int(support.sum()),'coarse_above_floor_XY_cells':int(above_floor.sum()),
                'input_proxy_XY_cells':int(expected.sum()),'coarse_roof_proxy_IoU_DIAGNOSTIC':float(intersection/max(1,union)),
                'not_measured_geographic_or_final3D_metric':True,'source':str(path)})
        for r in records[-4:]:
            other=masks[r['mode']];baseline=masks['baseline'];correct=masks['correct']
            r['SS_symmetric_difference_vs_baseline']=float((other^baseline).sum()/max(1,(other|baseline).sum()))
            r['SS_symmetric_difference_vs_correct']=float((other^correct).sum()/max(1,(other|correct).sum()))
        # Remove ONLY these diagnostic hook handles; never mutate weights.
        for h in wrapped.reader.hooks:h.remove()
        pipeline.models['sparse_structure_flow_model']=native
        del wrapped,branch;torch.cuda.empty_cache()
    save(OUT/'SPARSE_SUPPORT_ATTRIBUTION.json',{'same_weights_reference_sampler_and_seed':True,
        'new_model_updates':0,'records':records,
        'image_condition_contract':'reference building image is consumed by native SS and shape geometry as well as texture; not appearance-only',
        'prior_results_unmodified':True,'proxy_metric_diagnostic_only':True})
    print('SAME_SEED_SS_TRACE_COMPLETE',flush=True)


if __name__=='__main__':run()

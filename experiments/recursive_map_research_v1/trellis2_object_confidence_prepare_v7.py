from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import OUT as OLD,ARMS,SEEDS,write_npz
from trellis2_object_confidence_fix_v7 import correct_object_confidence
import numpy as np,shutil
OUT=JOB/'map_object_world_v1/trellis2_structure_path_diagnosis_v7/object_confidence_fix1'

if __name__=='__main__':
    assert not (OUT/'PLAN.json').exists()
    sources=read(OLD/'REAL_OBJECT_INPUTS.json');records=[];audit=[]
    OUT.mkdir(parents=True,exist_ok=True)
    for r in sources['records']:
        with np.load(r['map_input']) as z:m={k:z[k].copy() for k in z.files}
        fixed=correct_object_confidence(m)
        path=OUT/'real_input'/f"{r['mode']}_object{r['object_index']}.npz";write_npz(path,**fixed)
        records.append({**r,'map_input':str(path),'sha256':sha(path)})
        audit.append({'mode':r['mode'],'object_index':r['object_index'],
            'old_confidences':m['object_summary'][:,5].tolist(),
            'corrected_confidences':fixed['object_summary'][:,5].tolist(),
            'same_rasters_unknown_masks_pose_and_geometry':True,'original_input_sha256':sha(r['map_input'])})
    save(OUT/'REAL_OBJECT_INPUTS.json',{**sources,'records':records})
    # Copy pointers, never alter or retrain checkpoint data.
    for a in ARMS:
        for s in SEEDS:
            p=OLD/a/f'seed{s}'/'CHECKPOINT_000300.json';ck=read(p);assert sha(ck['path'])==ck['sha256']
            save(OUT/a/f'seed{s}'/'CHECKPOINT_000300.json',ck)
    save(OUT/'CONFIDENCE_EVIDENCE.json',{'proof':'old local_map summary[5] was footprint_known.mean across full crop; corrected to evidence mass over actual object footprint',
         'only_changed_column':'object_summary[:,5]','cases':audit,
         'annotation_and_UI_uncertainty_preserved':True,'no_known_mask_filling_or_output_edit':True})
    save(OUT/'PLAN.json',{'arms':ARMS,'seeds':SEEDS,'optimizer_updates':0,
         'single_fix':'object confidence denominator uses object support rather than unrelated background',
         'weights_unchanged':True,'original_v6_baseline_reused':True,
         'fixed_noise_pose_sampler_and_all_other_map_fields':True,
         'conditions_only':'all four map objects original/edit, both arms/two seeds; no seed search',
         'hard_GPU_seconds_each':300,'no_automatic_extension':True,
         'training_contract_limit':'v6 was trained with the erroneous crop-density summary. This is an input semantic correction, NOT newly trained adaptation; any failure remains a failure.'})
    print('SINGLE_CONFIDENCE_INPUT_FIX_REGISTERED',audit,flush=True)

from paths import JOB,read,save,sha
from trellis2_structure_path_owner_v7 import OUT
from trellis2_map_geometry_entry_v2_owner import GPUS
from map_object_composition_owner_v1 import bounded
if __name__=='__main__':
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_structure_path_math_fix1.py','trellis2_structure_path_diagnosis_v7.py']}
    save(OUT/'DECODER_MATH_DIAGNOSIS_CORRECTION.json',{'prior_failure_preserved':True,
        'only_missing_job_repeated':'object_grounded seed953102',
        'native_SS_latents_were_bit_identical':True,'reason':'match native v6 inference CuDNN flag; not geometry repair',
        'optimizer_updates':0,'source':freeze})
    fn=bounded();fn.__globals__['NEW']=OUT
    def verify():
        for p,h in freeze.items():assert sha(p)==h,p
    fn.__globals__['frozen']=verify
    fn(GPUS[3],'path_object_grounded',953102,'decoder_math_fix1','trellis2_structure_path_math_fix1.py',
       ['--arm','object_grounded','--seed','953102'],seconds=300)
    save(OUT/'TRACE_COMPLETE_FIX1.json',{'all_four_fixed_weight_diagnostics_complete':True,
        'corrected_missing_file':'decoder_math_fix1/object_grounded/seed953102/PATH_DIAGNOSIS.json',
        'no_training_or_model_changes':True})

"""Matched native Structure sampling; one change to free-space gradient allocation."""
from pathlib import Path
import sys
from trellis2_map_volume_v19 import OUT,OLD,SEED,JOB,read,save,sha

def register():
    d=read(OUT/'DIAGNOSIS.json');assert d['supported_issue']=='Sparse heightwise gradient allocation'
    assert all(r['saved_support_reproduced'] and r['old_direct_gradient_coverage']<.25 for r in d['rows'])
    p=read(OLD/'PLAN.json')
    p['scope']='One evidence-based gradient allocation correction, original/edit, same fixed native prior and noise'
    p['guidance']['negative']='Distributed softplus over height, detached per-column scale matching old negative logit-gradient L1'
    p['single_changed_factor']='Allocation of fixed per-column free-space gradient over Z'
    p['scalar_loss_comparison']='Surrogate scalar differs; judge binary support and saved mesh, not loss reduction'
    p['unchanged_settings']=['positive coverage term','prior term','LR','inner updates','trust RMS','active times','native sampler/CFG',
        'model weights','DINO reference','native Shape/texture','coordinate noise','decoder','camera poses','map masks']
    p['diagnosis_sha256']=sha(OUT/'DIAGNOSIS.json');p['control']=str(OLD)
    p['input_uncertainty']='Same approximate roof proxies and outside mask as v18; outside includes unannotated regions, so not surveyed free-space GT'
    save(OUT/'PLAN.json',p)
    for mode in ['correct','edit']:save(OUT/mode/'PLAN.json',read(OLD/mode/'PLAN.json'))
    names=['trellis2_map_volume_v19_guidance.py','trellis2_map_volume_v19_coarse.py','trellis2_map_volume_v19_generate.py',
        'trellis2_map_volume_v19_views.py','trellis2_mapguided_v18.py','trellis2_mapguided_v18_guidance.py',
        'trellis2_mapguided_v18_generate.py','trellis2_native_street_v17.py','trellis2_mapguided_v18_views.py',
        'trellis2_native_street_v17_views.py','trellis2_mapguided_v18_overlay_fix1.py']
    frozen={str(JOB/n):sha(JOB/n) for n in names}
    frozen[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    frozen[str(Path(p['runtime'])/'pipeline.json')]=sha(Path(p['runtime'])/'pipeline.json')
    frozen[p['input']]=p['input_sha256']
    frozen.update({v['path']:v['sha256'] for v in p['maps'].values()})
    save(OUT/'CANDIDATE_SOURCE.json',frozen)
    print('ONE_FIXED_CORRECTION_REGISTERED',flush=True)

if '--register' in sys.argv:register()
else:
    mode=sys.argv[sys.argv.index('--mode')+1];assert mode in ['correct','edit']
    source=Path(__file__).with_name('trellis2_mapguided_v18.py')
    code=source.read_text().split("if __name__=='__main__':")[0]
    code=code.replace('from trellis2_mapguided_v18_guidance import MapGuidedSampler',
        'from trellis2_map_volume_v19_guidance import VolumeSampler as MapGuidedSampler')
    scope={'__name__':'matched_v19_coarse','__file__':str(source)}
    exec(compile(code,str(source),'exec'),scope);scope['OUT']=OUT;scope['SEED']=SEED
    scope['coarse'](mode)

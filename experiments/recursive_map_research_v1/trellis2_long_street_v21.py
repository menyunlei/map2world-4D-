"""One larger native TRELLIS.2 street sample; retain v17 code and weights."""
from paths import JOB,read,save,sha
import trellis2_native_street_v17 as native
from pathlib import Path
import sys
OUT=JOB/'map_object_world_v1/trellis2_long_street_v21'
SEED=942217

def prepare():
    from PIL import Image
    import subprocess
    assert not OUT.exists()
    source=JOB/'map_object_world_v1/trellis2_long_street_v21_reference.png'
    old=read(JOB/'map_object_world_v1/trellis2_native_street_v17/PLAN.json')
    OUT.mkdir(parents=True)
    for s in ['logs','audit','gpu_cost','work']:(OUT/s).mkdir()
    image=Image.open(source).convert('RGBA');side=max(image.size)
    canvas=Image.new('RGBA',(side,side),(255,255,255,255));canvas.alpha_composite(image,((side-image.width)//2,(side-image.height)//2))
    canvas.convert('RGB').save(OUT/'input.png')
    plan={**old,'source_commit':subprocess.check_output(['git','-C',str(native.REPO),'rev-parse','HEAD'],text=True).strip(),
        'reference':str(source),'reference_sha256':sha(source),'input':str(OUT/'input.png'),
        'input_sha256':sha(OUT/'input.png'),'reference_kind':'One fixed AI-generated longer-street conditioning image; not map or evidence of a 3D result',
        'reference_prompt_file':str(JOB/'map_object_world_v1/trellis2_long_street_v21_reference.json'),
        'target':'approximately100m continuous street,8-12 distinct buildings, two sides; retain v17 quality',
        'sample_count':1,'seed':SEED,'no_tile_repetition':True,'no_separate_building_assembly':True,
        'nominal_scale':'one global scale mapping generated longest XY extent to100m, arbitrary display calibration, not surveyed metric recovery',
        'street_camera_rule':'full fixed v17 views plus lateral; then a logged road-center camera path on this saved asset only',
        'GPU_limit_seconds':3600,'one_generation_only':True,'no_training':True,'no_map_control':True,
        'quality_warning':'Increasing coverage within the same1024 volume may reduce per-building detail; assess actual saved mesh, not conditioning image'}
    save(OUT/'PLAN.json',plan)
    files=[Path(__file__),JOB/'trellis2_long_street_v21_owner.py',JOB/'trellis2_long_street_v21_views.py',
        JOB/'trellis2_native_street_v17.py',JOB/'trellis2_native_street_v17_views.py',source,OUT/'input.png',OUT/'PLAN.json',Path(old['runtime'])/'pipeline.json']
    save(OUT/'SOURCE_FREEZE.json',{str(f):sha(f) for f in files})
    print('V21_REGISTERED_ONE_NATIVE_STREET',flush=True)

def verify():
    for f,h in read(OUT/'SOURCE_FREEZE.json').items():assert sha(f)==h,('SOURCE_CHANGED',f)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        verify();native.OUT=OUT;native.SEED=SEED;native.generate()

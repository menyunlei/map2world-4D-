"""Fixed two-object, two-trained-seed inference diagnostic. No training."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import local_map,write_npz
import numpy as np
from PIL import Image,ImageDraw

OUT=JOB/'map_object_world_v1/trellis2_map_edit_mesh_v9'
V8=JOB/'map_object_world_v1/trellis2_structure_learnability_v8'
V6=JOB/'map_object_world_v1/trellis2_map_object_contract_v6'
SEEDS=[953101,953102]

def prepare():
    assert not (OUT/'PLAN.json').exists()
    prior=read(V8/'PLAN.json'); records=prior['selected_records'][:2]
    assert [r['object_id'] for r in records]==[1,2]
    assert records[0]['scene_id']==records[1]['scene_id']=='graph_0064_a0'
    global_r=next(r for r in read(JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3/PLAN.json')['records'] if r['scene_id']==records[0]['scene_id'])
    assert sha(global_r['map_input'])==global_r['map_sha256']
    with np.load(global_r['map_input']) as z: original={k:z[k].copy() for k in z.files}
    edited={k:v.copy() for k,v in original.items()}; j=records[0]['target_object']
    yy,xx=np.where(original['footprints'][j]>.5)
    removed=(original['footprints'][j]>.5)&(np.indices((64,64))[1] < xx.min()+.25*(xx.max()-xx.min()+1))
    edited['footprints'][j][removed]=0
    edited['open_space'][0][removed]=1;edited['open_space_known'][0][removed]=1
    assert removed.any() and np.array_equal(original['roads'],edited['roads'])
    maps=[]
    for mode,source in [('correct',original),('edit',edited)]:
        write_npz(OUT/'inputs'/f'{mode}_global.npz',**source)
        for i,r in enumerate(records):
            m,frame=local_map(source,r['target_object'],r['frame'])
            if mode=='correct':
                with np.load(r['map_input']) as z:
                    for k in z.files: assert np.array_equal(m[k],z[k]),('CANONICAL_CONTRACT_MISMATCH',i,k)
            p=OUT/'inputs'/f'{mode}_object{i}.npz';write_npz(p,**m)
            maps.append({'mode':mode,'index':i,'map_input':str(p),'sha256':sha(p),'frame':frame})
    cameras=[['front',[0,-1,-.12],[0,0,-.12],[0,0,1],50],
      ['lateral_left',[-.35,-1,-.12],[0,0,-.12],[0,0,1],50],
      ['lateral_right',[.35,-1,-.12],[0,0,-.12],[0,0,1],50],
      ['back',[0,1,-.12],[0,0,-.12],[0,0,1],50],
      ['left',[-1,0,-.12],[0,0,-.12],[0,0,1],50],
      ['right',[1,0,-.12],[0,0,-.12],[0,0,1],50],
      ['top',[0,0,1.1],[0,0,-.12],[0,1,0],50],
      ['oblique',[.8,-.8,.7],[0,0,-.12],[0,0,1],50]]
    checkpoints={str(s):read(V8/'small/structure_lora'/f'seed{s}/train/CHECKPOINT_001000.json') for s in SEEDS}
    for c in checkpoints.values(): assert sha(c['path'])==c['sha256']
    save(OUT/'PLAN.json',{'scope':'TRAIN mechanism diagnostic, not real-map/held-out world success',
      'selection':'first two v8 preregistered objects; not selected by output quality','records':records,'maps':maps,
      'training_seeds':SEEDS,'structure_generation_seed':970201,'shape_address_seed':980201,
      'SS_checkpoints':checkpoints,'Shape_condition_checkpoints':prior['parent_checkpoints'],
      'stage_contract':'v8 adapter+LoRA ONLY for SS; unchanged v6 jointly trained Shape adapter with its original raw-map confidence contract for fine geometry. v8 SS-only-updated head is NOT silently used as a trained Shape condition.',
      'edit':'remove west quarter of object1 footprint; retain object2 footprint and all original world frames; re-evaluate neighbor context without caching generated assets',
      'original_outputs':'reuse v8 noise970201 SS exactly; no unchanged inference resweep',
      'shape_chain':'unchanged official 1024 Shape Flow and shape decoder; no image conditions, no texture/PBR generation, no mesh repair/fill_holes',
      'diagnostics':['Oracle SS + SAME generated Shape, diagnostic only','encoded reference Shape + SAME decoder, diagnostic only, never delivered as generated prediction'],
      'oracle_not_normal_inference':True,'cameras':cameras,'GPU_occupancy_cap_seconds':7200,
      'hard_stop':'two seeds, two objects, correct/edit and registered diagnostics only; no training or expansion',
      'placement':'original input-map frames, not learned placement; no added ground/roads in mesh',
      'historical_v8_verdict_unchanged':True})
    freeze={str(JOB/n):sha(JOB/n) for n in ['trellis2_map_edit_mesh_v9_prepare.py','trellis2_map_edit_mesh_v9_run.py','trellis2_map_edit_mesh_v9_views.py','trellis2_map_edit_mesh_v9_owner.py']}
    freeze[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    save(OUT/'SOURCE_FREEZE.json',freeze)
    canvas=Image.new('RGB',(768,408),'#141c25')
    for n,src in enumerate([original,edited]):
        rgb=np.zeros((64,64,3),np.uint8)+30;rgb[src['roads'][0]>.5]=[95,95,95]
        for oi,fp in enumerate(src['footprints']):rgb[fp>.5]=[220,150,65] if oi==j else [70,155,215]
        canvas.paste(Image.fromarray(rgb[::-1]).resize((384,384),Image.Resampling.NEAREST),(384*n,24))
        ImageDraw.Draw(canvas).text((384*n+5,5),['Original map','West-quarter footprint edit; fixed pose'][n],fill='white')
    canvas.save(OUT/'INPUT_EDIT.png');print('REGISTERED_TWO_OBJECTS_TWO_SEEDS',flush=True)

if __name__=='__main__':prepare()

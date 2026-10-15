"""One fixed real-image development layout. No model/data fitting or new selection."""
from paths import JOB,read,save,sha
from trellis2_map_edit_mesh_v9_prepare import OUT as V9,V6,SEEDS
import shutil,numpy as np
OUT=JOB/'map_object_world_v1/trellis2_map10_block_v11'

def prepare():
    assert not (OUT/'PLAN.json').exists()
    old=read(V9/'PLAN.json');real=read(V6/'REAL_OBJECT_INPUTS.json')
    meta=read(V6/'real_input/MAP_INPUT.json')
    assert len(meta['objects'])==4 and real['edited_object']==3
    shutil.copytree(V6/'real_input',OUT/'inputs')
    maps=[];records=[]
    for i in range(4):
        for mode in ['correct','edit']:
            r=next(x for x in real['records'] if x['object_index']==i and x['mode']==mode)
            assert sha(r['map_input'])==r['sha256']
            p=OUT/'inputs'/f'{mode}_object{i}.npz';assert sha(p)==r['sha256']
            maps.append({'mode':mode,'index':i,'map_input':str(p),'sha256':sha(p),'frame':r['frame']})
            if mode=='correct':records.append({'object_index':i,'object_id':meta['objects'][i]['id'],'target_object':i,'frame':r['frame'],'label':meta['objects'][i]['label']})
    for ck in [*old['SS_checkpoints'].values(),*old['Shape_condition_checkpoints'].values()]:assert sha(ck['path'])==ck['sha256']
    # Cameras fixed from input-map coordinates, not generated bounds.
    g=-.4375;z=g+.045
    cameras=[['top',[0,0,1.2],[0,0,g],[0,1,0],45],
      ['oblique',[.95,-1.1,.55],[0,0,g+.10],[0,0,1],48],
      ['street_front',[-.12,-.60,z],[-.12,.24,z],[0,0,1],70],
      ['street_side',[.43,-.22,z],[-.25,-.05,z],[0,0,1],70],
      ['street_back',[.02,.50,z],[.02,-.20,z],[0,0,1],70]]
    lateral=[{'eye':[float(x),-.60,z],'look':[float(x),.24,z],'up':[0,0,1],'fov':70} for x in np.linspace(-.27,.13,48)]
    save(OUT/'PLAN.json',{'scope':'Fixed Map10 four-object development layout demonstration, not generalization proof',
      'records':records,'maps':maps,'training_seeds':SEEDS,'SS_checkpoints':old['SS_checkpoints'],
      'Shape_condition_checkpoints':old['Shape_condition_checkpoints'],
      'structure_generation_seed':old['structure_generation_seed'],'shape_address_seed':old['shape_address_seed'],
      'stage_contract':old['stage_contract'],'noise_contract':'Same v9 dense SS coordinate field and coordinate-indexed sparse Shape field; each object uses fixed canonical coordinates and original frame in both layouts. No output-dependent noise assignment.',
      'original_v9_evidence':'Positive local map-to-mesh response retained; v8 learnability failure and v10 unsuccessful anchor remain unchanged.',
      'no_v10_descriptor_anchor':True,'model_training':False,'materials':False,'mesh_repair':False,
      'input_metadata':meta,'edit':real['edit'],'edited_object_index':3,'original_object_pose_fixed':True,
      'global_maps':{'correct':str(OUT/'inputs/map.npz'),'edit':str(OUT/'inputs/footprint_edit.npz')},
      'cameras':cameras,'lateral':lateral,'fps':24,'reference_ground':g,
      'camera_height':'0.045 normalized world units above fixed reference ground; not a measured 1.6m camera',
      'background':'Model assets contain generated buildings only. Any separately rendered flat ground/road layer is input-map visualization, not AI-generated geometry.',
      'evaluation':'Horizontal map triangle-projection overlap and object counts separately from 3D extents, components, roof/height response and non-watertightness. No photorealism or unique hidden-roof gate.',
      'failure_policy':'Every object slot retained in exported scene and manifest, including explicit empty/error nodes. No substitute meshes or silently omitted objects.',
      'GPU_occupancy_cap_seconds':3600,'hard_stop':'2 seeds x 2 layouts x 4 objects, no repeat samples or follow-on training'})
    files=['trellis2_map10_block_v11_prepare.py','trellis2_map10_block_v11_run.py','trellis2_map10_block_v11_owner.py']
    save(OUT/'SOURCE_FREEZE.json',{**{str(JOB/f):sha(JOB/f) for f in files},str(OUT/'PLAN.json'):sha(OUT/'PLAN.json'),**{r['map_input']:r['sha256'] for r in maps}})
    print('FIXED_MAP10_FOUR_OBJECT_LAYOUT_READY',flush=True)

if __name__=='__main__':prepare()

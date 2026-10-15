"""One fixed Map10 street preview, existing weights, zero training."""
from paths import JOB,read,save,sha
from trellis2_map10_block_v11_prepare import OUT as V11, SEEDS
from trellis2_coverage_v12_prepare import OUT as V12
from trellis2_map_object_data_v6_fix1 import local_map,write_npz
from PIL import Image
import copy,shutil,numpy as np

OUT=JOB/'map_object_world_v1/trellis2_map10_street_v15'

def prepare():
    assert not OUT.exists()
    old=read(V11/'PLAN.json');p=copy.deepcopy(old)
    OUT.mkdir(parents=True);shutil.copytree(V11/'inputs',OUT/'inputs')
    with np.load(old['global_maps']['correct']) as z:m={k:z[k].copy() for k in z.files}
    # Visible asphalt corridor in the already selected910px crop, fixed before inference.
    road_px=[[410,0],[470,170],[560,420],[681,705],[785,910],
             [910,910],[851,735],[768,465],[674,200],[607,0]]
    xx,yy=np.meshgrid((np.arange(64)+.5)*910/64,910-(np.arange(64)+.5)*910/64)
    road=np.zeros((64,64),bool)
    for a,b in zip(road_px,road_px[1:]+road_px[:1]):
        if a[1]==b[1]:continue
        crossing=(a[1]>yy)!=(b[1]>yy)
        road^=crossing&(xx < (b[0]-a[0])*(yy-a[1])/(b[1]-a[1])+a[0])
    road&=~(m['footprints']>.5).any(0)
    m['roads'][0]=road;m['roads_known'][0][road]=1
    mp=OUT/'inputs/street_map.npz';write_npz(mp,**m)
    maps=[]
    for i,r in enumerate(old['records']):
        mm,_=local_map(m,i,r['frame']);path=OUT/'inputs'/f'street_object{i}.npz';write_npz(path,**mm)
        maps.append({'mode':'correct','index':i,'map_input':str(path),'sha256':sha(path),'frame':r['frame']})
    centers=np.array([[835,850],[794,745],[748,625],[698,505],[650,385],[603,270],[567,175]],float)
    segments=np.linalg.norm(np.diff(centers,axis=0),axis=1);arc=np.r_[0,np.cumsum(segments)]
    s=np.linspace(0,arc[-1],192);route=np.c_[np.interp(s,arc,centers[:,0]),np.interp(s,arc,centers[:,1])]
    g=old['reference_ground'];trajectory=[]
    for k,xy in enumerate(route):
        nx=route[min(k+8,len(route)-1)] if k<len(route)-1 else route[-1]+route[-1]-route[-2]
        eye=[xy[0]/910-.5,.5-xy[1]/910,g+.045]
        delta=np.array([(nx[0]-xy[0])/910,-(nx[1]-xy[1])/910]);delta/=max(1e-8,np.linalg.norm(delta))
        # Look along the street with a fixed small bias toward its generated west side.
        look=[eye[0]+.35*delta[0]-.07,eye[1]+.35*delta[1],eye[2]+.005]
        trajectory.append({'eye':eye,'look':look,'up':[0,0,1],'fov':72,'source_crop_xy':xy.tolist()})
    ss={str(s):read(V12/'small/six_object_rehearsal'/f'seed{s}/train/CHECKPOINT_001000.json') for s in SEEDS}
    appearance=read(JOB/'map_object_world_v1/trellis2_map_object_contract_v6/PLAN.json')['appearance_condition']
    p.update({'scope':'A bounded street approximation from10.jpg, not a recovered real street',
      'training_updates':0,'SS_checkpoints':ss,'maps':maps,'global_maps':{'correct':str(mp)},
      'appearance_condition':appearance,'appearance_sha256':sha(appearance),
      'weight_choice':'Registered v12 six-object continuation for BOTH seeds; failed v14 is not promoted, no output-based seed selection',
      'road_annotation_px':road_px,'trajectory':trajectory,'fps':24,'duration_seconds':8,
      'street_source_limit':'Four annotated buildings on west side only; east-side buildings and actual facades are not reconstructed',
      'appearance_source':'Existing fixed appearance reference; generated facade hypothesis, not verified Map10 facade',
      'background':'Map-derived flat road/ground visualization, labeled separately from model-generated buildings',
      'cameras':[['top',[0,0,1.2],[0,0,g],[0,1,0],45],
         ['oblique',[.95,-1.1,.55],[0,0,g+.10],[0,0,1],48],
         ['street_start',trajectory[0]['eye'],trajectory[0]['look'],[0,0,1],72],
         ['street_middle',trajectory[96]['eye'],trajectory[96]['look'],[0,0,1],72],
         ['street_end',trajectory[-1]['eye'],trajectory[-1]['look'],[0,0,1],72]],
      'GPU_occupancy_cap_seconds':3600,'hard_stop':'Two fixed seeds, one original layout each; no training, retries, alternate seeds or model changes'})
    save(OUT/'PLAN.json',p)
    frozen={str(OUT/'PLAN.json'):sha(OUT/'PLAN.json'),appearance:sha(appearance),str(mp):sha(mp)}
    frozen.update({r['map_input']:r['sha256'] for r in maps})
    for ck in [*ss.values(),*p['Shape_condition_checkpoints'].values()]:
        assert sha(ck['path'])==ck['sha256'];frozen[ck['path']]=ck['sha256']
    save(OUT/'INPUT_FREEZE.json',frozen)
    print('STREET_REGISTERED',SEEDS,4,192,flush=True)

if __name__=='__main__':prepare()

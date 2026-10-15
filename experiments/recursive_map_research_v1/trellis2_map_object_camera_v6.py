"""Fixed map-derived inspection cameras, before native fine assets exist."""
from trellis2_map_object_data_v6_fix1 import OUT
from paths import read,save,JOB
import numpy as np

if __name__=='__main__':
    poses=read(JOB/'map_object_world_v1/trellis2_map_learning_v1/trial_fix1/CAMERA_PROTOCOL.json')['cameras'][:6]
    with np.load(OUT/'real_input/map.npz') as z:
        roads=z['roads'][0];union=z['footprints'].max(0);known=z['roads_known'][0]
    points=(np.argwhere((roads*known>.2)&(union==0))[:,::-1]+.5)/64-.5
    inputs=read(OUT/'REAL_OBJECT_INPUTS.json')['records']
    for j in range(4):
        frame=next(r['frame'] for r in inputs if r['mode']=='correct' and r['object_index']==j)
        center=np.array(frame['center_xy']);dist=np.linalg.norm(points-center,axis=1)
        eligible=np.where(dist>frame['scale']*.25+.03)[0]
        assert len(eligible),'NO_MAP_DERIVED_OBSERVATION_POSITION'
        q=eligible[np.argmin(dist[eligible])];eye=points[q]
        look=[*center,-14/32+.07];base=[*eye,-14/32+.05]
        poses.append([f'street_object{j}',base,look,[0,0,1],70])
        if j==3:
            tangent=np.array([center[1]-eye[1],eye[0]-center[0]])
            tangent=tangent/np.linalg.norm(tangent)*.04
            poses.append(['street_lateral_left',[*(eye-tangent),base[2]],look,[0,0,1],70])
            poses.append(['street_lateral_right',[*(eye+tangent),base[2]],look,[0,0,1],70])
    save(OUT/'CAMERA_PROTOCOL.json',{'cameras':poses,'resolution':640,
        'source':'fixed map roads + original footprint frames; no generated output bounds',
        'identical_across_arms_seeds_and_edit':True,'ground_convention':-14/32,
        'road_ground_background':'optional 2D diagnostic only, NOT generated 3D geometry'})

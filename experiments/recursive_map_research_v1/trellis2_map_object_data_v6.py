"""Common TRAIN/inference object frame. No target-derived pose or resizing.

The map fixes XY anchor, isotropic scale and z=0 ground convention. A footprint
edit reuses the ORIGINAL frame. Source mesh is read only by target preparation.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from paths import JOB, read, save, sha

OUT = JOB / 'map_object_world_v1/trellis2_map_object_contract_v6'
OLD = JOB / 'map_object_world_v1/trellis2_map_geometry_tokens_v3'
ARMS = ['ordinary', 'object_grounded']
SEEDS = [953101, 953102]
GROUND = -.4


def frame_from_map(m, index):
    y,x = np.where(m['footprints'][index] * m['footprint_known'][index] > 0)
    assert len(x), 'Mapped target has no reliable footprint'
    center = np.array([(x.min()+x.max()+1)/128-.5,
                       (y.min()+y.max()+1)/128-.5], np.float32)
    # One fixed map-only scale rule, never fitted to the target or generated mesh.
    scale = max(x.max()-x.min()+1,y.max()-y.min()+1)/64/.5
    return {'center_xy':center.tolist(), 'scale':float(scale),
            'center_z':float(-GROUND*scale), 'rotation':0,
            'source':'original map footprint bounds; unknown height NOT observed',
            'ground_z':GROUND, 'isotropic_scale_rule':'max map XY extent / 0.5'}


def local_map(m, index, frame=None):
    frame = frame_from_map(m,index) if frame is None else frame
    center=np.asarray(frame['center_xy']); scale=frame['scale']
    axis=(np.arange(64)+.5)/64-.5
    gx,gy=np.meshgrid(axis*scale+center[0],axis*scale+center[1],indexing='xy')
    x=np.floor((gx+.5)*64).astype(int); y=np.floor((gy+.5)*64).astype(int)
    inside=(x>=0)&(x<64)&(y>=0)&(y<64)
    def resample(a):
        return a[:,y.clip(0,63),x.clip(0,63)]*inside[None]
    r={k:resample(m[k]).astype(np.float32) for k in
       ['footprints','footprint_known','roads','roads_known','open_space','open_space_known']}
    valid=m['object_valid'].copy()
    summaries=[]
    for j,fp in enumerate(r['footprints']):
        yy,xx=np.where(fp*r['footprint_known'][j]>0)
        valid[j] &= len(xx)>0
        summaries.append([(xx.mean()+.5)/64-.5,(yy.mean()+.5)/64-.5,
            (xx.max()-xx.min()+1)/64,(yy.max()-yy.min()+1)/64,
            float(fp.mean()),float(r['footprint_known'][j].mean()),0.,1.] if len(xx)
            else [0.,0.,0.,0.,0.,0.,0.,1.])
    assert valid[index]
    r['object_valid']=valid
    r['object_summary']=np.asarray(summaries,np.float32)
    r['target_object']=np.array(index,np.int64)
    return r,frame


def write_npz(path, **values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: np.savez_compressed(f,**values)


def prepare():
    import hashlib
    plan=read(OLD/'PLAN.json')
    assert not (OUT/'PLAN.json').exists()
    records=[]; geometries={}; audit=[]
    for r in plan['records']:
        assert r['original_split']=='train' and r['split'] in ['adapter_TRAIN','adapter_HELDOUT']
        assert sha(r['source_manifest'])==r['manifest_sha256']
        assert sha(r['target_mesh'])==r['mesh_sha256']
        meta=read(r['source_manifest'])
        with np.load(r['map_input']) as z: m={k:z[k].copy() for k in z.files}
        with np.load(r['target_mesh']) as z: mesh={k:z[k].copy() for k in z.files}
        assert set(o['object_id'] for o in meta['objects']) <= set(np.unique(mesh['face_object_ids']))
        for j,obj in enumerate(meta['objects']):
            mm,frame=local_map(m,j)
            f=mesh['faces'][mesh['face_object_ids']==obj['object_id']]
            assert len(f)>0
            used,inverse=np.unique(f,return_inverse=True)
            v=mesh['vertices_native'][used].astype(np.float32)
            # Re-express unchanged authored geometry; no new faces or corrections.
            v=(v-np.array([*frame['center_xy'],frame['center_z']],np.float32))/frame['scale']
            faces=inverse.reshape(-1,3).astype(np.int32)
            limit=float(np.abs(v).max())
            audit.append({'scene':r['scene_id'],'object':obj['object_id'],
                'faces':len(faces),'finite':bool(np.isfinite(v).all()),
                'native_bound':limit,'frame':frame,'source_asset':obj['asset_id'],
                'license':obj['source_license'],'source_asset_sha256':obj['asset_sha256']})
            assert np.isfinite(v).all() and limit<.4999, ('MAP_FRAME_OUTSIDE_NATIVE_CODEC',audit[-1])
            key=hashlib.sha256(v.tobytes()+faces.tobytes()).hexdigest()[:24]
            if key not in geometries:
                target=OUT/'source_geometry'/f'{key}.npz'
                write_npz(target,vertices=v,faces=faces)
                geometries[key]={'key':key,'source_geometry':str(target),'sha256':sha(target),
                    'source_mesh':r['target_mesh'],'source_mesh_sha256':r['mesh_sha256'],
                    'source_object_id':obj['object_id'],'faces':len(faces)}
            mp=OUT/'maps'/f"{r['scene_id']}_object{obj['object_id']}.npz"
            write_npz(mp,**mm)
            records.append({'scene_id':r['scene_id'],'family_id':r['family_id'],
                'split':r['split'],'original_split':'train','object_id':obj['object_id'],
                'target_object':j,'object_count':r['object_count'],'map_input':str(mp),
                'map_sha256':sha(mp),'geometry_key':key,'frame':frame,
                'source_manifest':r['source_manifest'],'manifest_sha256':r['manifest_sha256'],
                'source_asset_id':obj['asset_id'],'source_asset_sha256':obj['asset_sha256'],
                'target_source':'existing legitimate authored TRAIN mesh, original face_object_ids',
                'not_real_city_reconstruction':True})
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'TARGET_SOURCE_AUDIT.json',{'objects':audit,'valid_nonempty_object_targets':len(audit),
         'unique_native_geometries':len(geometries),'real_paired_TRAIN_verified':0,
         'no_NEW_target_meshes':True,'no_historical_DEV_or_sealed_access':True})
    save(OUT/'SOURCE_GEOMETRIES.json',{'records':list(geometries.values())})
    # Fix real evaluation inputs and edit before learning or sampling.
    import shutil
    shutil.copytree(OLD/'real_input',OUT/'real_input')
    with np.load(OUT/'real_input/map.npz') as z: real={k:z[k].copy() for k in z.files}
    edited={k:v.copy() for k,v in real.items()}
    target_index=len(real['footprints'])-1
    yy,xx=np.where(real['footprints'][target_index]>0)
    # One predeclared west-half narrowing, preserving object slot and original pose.
    remove=(np.indices((64,64))[1] < (xx.min()+xx.max()+1)/2) & (real['footprints'][target_index]>0)
    edited['footprints'][target_index][remove]=0
    edited['open_space'][0][remove]=1;edited['open_space_known'][0][remove]=1
    write_npz(OUT/'real_input/footprint_edit.npz',**edited)
    views=[]
    for mode,src in [('correct',real),('edit',edited)]:
        for j in range(len(real['footprints'])):
            frame=frame_from_map(real,j)
            mm,_=local_map(src,j,frame)
            path=OUT/'real_input'/f'{mode}_object{j}.npz'
            write_npz(path,**mm)
            views.append({'mode':mode,'object_index':j,'map_input':str(path),
                'sha256':sha(path),'frame':frame})
    save(OUT/'REAL_OBJECT_INPUTS.json',{'records':views,'edited_object':target_index,
        'pose_unchanged':True,'edit':'remove west half of selected footprint at original pose',
        'placement':'fixed map-only XY, scale and ground; NOT learned scene placement',
        'background':'none model-generated yet; any fixed ground/roads must be labeled separately',
        'normalized_map10_not_georeferenced':True})
    parents={str(s):read(OLD/'separated'/f'seed{s}'/'CHECKPOINT_000300.json') for s in SEEDS}
    for p in parents.values(): assert sha(p['path'])==p['sha256']
    save(OUT/'PLAN.json',{'status':'REGISTERED_OBJECT_CONTRACT_WITH_VALID_TARGET_SOURCES',
        'arms':ARMS,'seeds':SEEDS,'records':records,'parent_checkpoints':parents,
        'max_updates':300,'checkpoints':[20,100,300],'effective_batch':2,
        'learning_rate':1e-4,'weight_decay':.01,'betas':[.9,.95],'clip_grad':1.,
        'objective':'existing native SS + Shape flow MSE, equal stage weighting; no new loss',
        'geometry_image_condition':False,'appearance_image_after_geometry':True,
        'trainable':'exact existing 611584-parameter v3 map adapter only',
        'ordinary':'all-map ordinary routing in SAME per-object canonical crop and output organization',
        'object_grounded':'selected-object routing with existing cross-object MHA summaries',
        'shared_contract':'both train/infer pre-evidence query context; fixed map-only object frame; native separate object SS and shape',
        'real_paired_TRAIN_verified':0,'TRAIN_map10':False,
        'appearance_condition':plan['appearance_condition'],'appearance_sha256':plan['appearance_sha256'],
        'budget':'register job time limits after real microbatch throughput; no automatic extension',
        'material_gate':'inspect each generated object Structure first; no material on empty or grossly mismatched support',
        'no_automatic_budget_extension':True,'old_v5_preserved':True})
    print('OBJECT_TARGETS_REGISTERED',len(records),len(geometries),flush=True)


if __name__=='__main__': prepare()

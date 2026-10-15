"""Separate horizontal-layout and native-massing measurements; no new historical gate."""
from paths import read,save,sha
from trellis2_map10_block_v11_prepare import OUT,SEEDS
import numpy as np,cv2,trimesh,csv
from PIL import Image,ImageDraw
AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32);N=256

def projection(v,f):
    a=np.zeros((N,N),np.uint8);xy=np.round((v[:,:2]+.5)*N-.5).astype(np.int32)
    for t in xy[f]:cv2.fillConvexPoly(a,t,1)
    return a.astype(bool)

def main():
    p=read(OUT/'PLAN.json');rows=[];masks={};layouts=[]
    for seed in SEEDS:
        for mode in ['correct','edit']:
            d=OUT/f'seed{seed}'/mode;entry=read(d/'LAYOUT.json')
            sc=trimesh.load(d/'layout.glb',force='scene',process=False)
            with np.load(p['global_maps'][mode]) as z:gm={k:z[k].copy() for k in z.files}
            total=np.zeros((N,N),bool);expected_all=np.zeros((N,N),bool);color=np.zeros((N,N,3),np.uint8)+28
            crop=Image.open(OUT/'inputs/source_crop.png').convert('RGB');overlay=np.asarray(crop).copy()
            for i,r in enumerate(p['records']):
                expected=cv2.resize((gm['footprints'][i]>.5).astype(np.uint8),(N,N),interpolation=cv2.INTER_NEAREST).astype(bool);expected_all|=expected
                od=d/f'object{i}';status=entry['objects'][i]['status'];row={'training_seed':seed,'mode':mode,'object_index':i,'object_id':r['object_id'],'status':status}
                mask=np.zeros((N,N),bool)
                if status=='MESH_GENERATED':
                    meta=read(od/'ASSET.json');assert sha(od/'geometry.glb')==meta['sha256']
                    with np.load(od/'NATIVE_MESH.npz') as z:v=z['vertices'].copy();f=z['faces'].copy()
                    frame=r['frame'];w=v*frame['scale']+np.array([*frame['center_xy'],frame['center_z']])
                    T,gkey=sc.graph[r['object_id']];g=sc.geometry[gkey]
                    reloaded=trimesh.transform_points(g.vertices,T)@AXIS
                    assert np.allclose(reloaded,w,atol=1e-7) and np.array_equal(g.faces,f)
                    mask=projection(w,f);bounds=v.max(0)-v.min(0)
                    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);area=np.linalg.norm(normals,axis=1);nz=np.abs(normals[:,2])/np.maximum(area,1e-12)
                    with np.load(od/'SS_GENERATED.npz') as z:q=z['native_support64']
                    xy=np.argwhere(mask);ep=np.argwhere(expected)
                    row.update({'vertices':len(v),'faces':len(f),'map_projection_IoU':float((mask&expected).sum()/max(1,(mask|expected).sum())),
                      'input_footprint_coverage':float((mask&expected).sum()/max(1,expected.sum())),
                      'outside_footprint_fraction':float((mask&~expected).sum()/max(1,mask.sum())),
                      'centroid_error_normalized_world':float(np.linalg.norm(xy.mean(0)-ep.mean(0))/N) if len(xy) and len(ep) else None,
                      'native_extent_xyz':bounds.tolist(),'world_extent_xyz':(bounds*frame['scale']).tolist(),'world_z_min':float(w[:,2].min()),'world_z_max':float(w[:,2].max()),
                      'height_to_max_horizontal_extent':float(bounds[2]/max(bounds[0],bounds[1],1e-9)),
                      'has_3D_extent_over_two_native_decoder_cells':bool(np.all(bounds>2/1024)),
                      'vertical_face_area_fraction':float(area[nz<.3].sum()/max(area.sum(),1e-12)),
                      'near_horizontal_face_area_fraction':float(area[nz>.7].sum()/max(area.sum(),1e-12)),
                      'SS_occupied_height_levels':int(q.any((0,1)).sum()),'watertight':meta['watertight'],'winding_consistent':meta['winding_consistent'],
                      'world_export_exact':True})
                masks[(seed,mode,i)]=mask;total|=mask;rows.append(row)
                # Same affine screenshot coordinates as the frozen annotation; not geo-rectification.
                for binary,bgr in [(expected,[255,75,65]),(mask,[45,235,245])]:
                    contours,_=cv2.findContours(binary[::-1].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    contours=[np.round(c.astype(float)*(overlay.shape[0]/N)).astype(np.int32) for c in contours]
                    cv2.drawContours(overlay,contours,-1,bgr,2)
                yy,xx=np.where(expected)
                if len(xx):cv2.putText(overlay,str(i+1),(int(xx.mean()/N*overlay.shape[1]),int((1-yy.mean()/N)*overlay.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            road=cv2.resize(((gm['roads'][0]>.5)&(gm['roads_known'][0]>0)).astype(np.uint8),(N,N),interpolation=cv2.INTER_NEAREST).astype(bool)
            color[road]=[80,80,80];color[expected_all]=[210,65,60];color[total]=[50,205,230];color[expected_all&total]=[235,200,70]
            Image.fromarray(color[::-1]).resize((768,768),Image.Resampling.NEAREST).save(d/'top_map_overlay.png')
            out=Image.fromarray(overlay);draw=ImageDraw.Draw(out);draw.rectangle((0,0,910,29),fill='#111923');draw.text((6,7),'INPUT roof proxies RED | generated mesh projection CYAN | screenshot affine, NOT surveyed footprint',fill='white');out.save(d/'input_map_overlay.png')
            np.savez_compressed(d/'GLOBAL_PROJECTION.npz',generated=total,expected=expected_all,roads=road)
            layouts.append({'seed':seed,'mode':mode,'expected_objects':4,'generated_objects':entry['generated_count'],'combined_map_IoU':float((total&expected_all).sum()/max(1,(total|expected_all).sum())),
              'known_road_area_covered_fraction':float((total&road).sum()/max(1,road.sum())),'world_export_all_objects_checked':True})
    edits=[]
    with np.load(p['global_maps']['correct']) as z:before=z['footprints'][3]>.5
    with np.load(p['global_maps']['edit']) as z:after=z['footprints'][3]>.5
    removed=cv2.resize((before&~after).astype(np.uint8),(N,N),interpolation=cv2.INTER_NEAREST).astype(bool)
    for seed in SEEDS:
        for i in range(4):
            a=next(r for r in rows if r['training_seed']==seed and r['mode']=='correct' and r['object_index']==i);b=next(r for r in rows if r['training_seed']==seed and r['mode']=='edit' and r['object_index']==i)
            r={'seed':seed,'object_index':i,'is_edited_object':i==3,'before_status':a['status'],'after_status':b['status']}
            if a['status']==b['status']=='MESH_GENERATED':
                ds=[OUT/f'seed{seed}'/m/f'object{i}' for m in ['correct','edit']]
                with np.load(ds[0]/'NATIVE_MESH.npz') as x,np.load(ds[1]/'NATIVE_MESH.npz') as y:
                    r['mesh_exact']=bool(np.array_equal(x['vertices'],y['vertices']) and np.array_equal(x['faces'],y['faces']))
                with np.load(ds[0]/'SS_GENERATED.npz') as x,np.load(ds[1]/'SS_GENERATED.npz') as y:
                    q0=x['native_support64'];q1=y['native_support64'];r['SS_symmetric_difference_over_union']=float((q0^q1).sum()/max(1,(q0|q1).sum()))
                r.update({'height_before':a['world_extent_xyz'][2],'height_after':b['world_extent_xyz'][2],'height_relative_change':float((b['world_extent_xyz'][2]-a['world_extent_xyz'][2])/max(1e-9,a['world_extent_xyz'][2]))})
                if i==3:r.update({'removed_area_occupancy_before':float(masks[(seed,'correct',i)][removed].mean()),'removed_area_occupancy_after':float(masks[(seed,'edit',i)][removed].mean())})
            edits.append(r)
    save(OUT/'RESULTS.json',{'scope':p['scope'],'per_object':rows,'layout':layouts,'edits':edits,'v9_positive_local_map_to_mesh_evidence_preserved':True,'no_new_training':True,
      'horizontal_layout':'Triangle projection versus frozen screenshot-derived proxies; not hidden 3D target reconstruction',
      'massing':'Extents and surface orientation reported separately; nonzero extent is not proof of closed solid volume',
      'AI_visual_review':'pending actual image inspection','no_photorealism_or_exact_hidden_roof_gate':True,'no_new_historical_thresholds':True})
    for name,rs in [('PER_OBJECT.csv',rows),('LAYOUT.csv',layouts),('EDIT.csv',edits)]:
        with (OUT/name).open('x',newline='') as h:w=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
    print('MAP10_GEOMETRY_MEASURED',layouts,edits,flush=True)

if __name__=='__main__':main()

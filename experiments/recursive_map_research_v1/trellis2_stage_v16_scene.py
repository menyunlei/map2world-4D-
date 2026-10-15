"""Export fixed-world layouts, fresh-load matched street and look-back views."""
from trellis2_stage_v16 import OUT,V15,SEEDS
from paths import read,save,sha
from pathlib import Path
import argparse,numpy as np,trimesh

AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)

def run(seed):
    from trellis2_map_edit_mesh_v9_views import camera_matrices,MeshRenderer
    from trellis2.representations import Mesh
    import torch
    from PIL import Image,ImageDraw
    p=read(V15/'PLAN.json');rows=[];assets={}
    renderer=MeshRenderer();renderer.rendering_options.resolution=512;renderer.rendering_options.ssaa=1
    renderer.rendering_options.near=.001;renderer.rendering_options.far=10
    renderer.rendering_options.antialias=False;renderer.rendering_options.chunk_size=8000000
    midpoint=p['trajectory'][len(p['trajectory'])//2]
    lookback={'eye':midpoint['eye'],'look':(2*np.asarray(midpoint['eye'])-np.asarray(midpoint['look'])).tolist(),
              'up':midpoint['up'],'fov':midpoint['fov']}
    cameras=p['cameras']+[['look_back',lookback['eye'],lookback['look'],lookback['up'],lookback['fov']]]
    for arm in ['baseline','repair']:
        d=OUT/'streets'/f'seed{seed}'/arm;d.mkdir(parents=True,exist_ok=False)
        scene=trimesh.Scene();entries=[]
        for i,r in enumerate(p['records']):
            src=V15/f'seed{seed}/correct/object{i}' if arm=='baseline' else OUT/f'generated/seed{seed}/object{i}'
            entry={'object_index':i,'object_id':r['object_id'],'geometry_source':str(src/'geometry.glb'),
                   'pose_source':'Unchanged input map frame; not model predicted placement','frame':r['frame']}
            f=r['frame'];T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=np.array([*f['center_xy'],f['center_z']])@AXIS.T
            if (src/'ASSET.json').exists():
                meta=read(src/'ASSET.json');assert sha(src/'geometry.glb')==meta['sha256']
                sc=trimesh.load(src/'geometry.glb',force='scene',process=False);g=list(sc.geometry.values())[0]
                scene.add_geometry(g,geom_name=r['object_id'],node_name=r['object_id'],transform=T)
                entry.update(status='MESH_GENERATED',sha256=meta['sha256'],faces=len(g.faces),bounds_native=meta['native_bounds'])
            else:
                scene.graph.update(frame_to=r['object_id']+'__EMPTY_STRUCTURE',matrix=T,metadata={'no_substitute':True})
                entry.update(status='EMPTY_STRUCTURE',sha256=None)
            with np.load(src/'SS_GENERATED.npz') as z:q=z['native_support64'].astype(bool)
            mr=next(x for x in p['maps'] if x['mode']=='correct' and x['index']==i)
            with np.load(mr['map_input']) as z:fp=z['footprints'][r['target_object']]>.5
            xy=q.any(2).T
            from scipy.ndimage import label
            _,cc=label(q)
            entry.update(support_cells=int(q.sum()),support_components=int(cc),
                Structure_map_IoU=float((xy&fp).sum()/max(1,(xy|fp).sum())),
                footprint_coverage=float((xy&fp).sum()/max(1,fp.sum())),
                outside_footprint=float((xy&~fp).sum()/max(1,xy.sum())))
            entries.append(entry)
        scene.metadata={'source':'TRELLIS.2 generated meshes, no geometry repair; fixed map frames',
                        'background':'not included','no_map10_target':'no real 3D GT available'}
        asset=d/'street_buildings.glb';scene.export(asset)
        # Independent file parser load, all world transforms applied by parser.
        sc=trimesh.load(asset,force='scene',process=False);loaded=sc.to_mesh()
        v=(loaded.vertices@AXIS).astype(np.float32);f=loaded.faces.astype(np.int32)
        assert len(f)==sum(e.get('faces',0) for e in entries)
        # Diagnostic orthographic raster of ACTUAL generated triangles, no plane.
        import nvdiffrast.torch as dr
        clip=torch.ones((len(v),4),device='cuda',dtype=torch.float32)
        clip[:,:2]=torch.from_numpy(v[:,:2].copy()).cuda()*2
        clip[:,2]=-torch.from_numpy(v[:,2].copy()).cuda()/2
        glctx=dr.RasterizeCudaContext()
        with torch.no_grad():raster,_=dr.rasterize(glctx,clip[None],torch.from_numpy(f).cuda(),resolution=[256,256])
        projected=(raster[0,:,:,3]>0).cpu().numpy()
        with np.load(p['global_maps']['correct']) as z:expected=z['footprints'].max(0)>.5;road=z['roads'][0]>.5
        expected=np.asarray(Image.fromarray(expected).resize((256,256),Image.Resampling.NEAREST))
        road=np.asarray(Image.fromarray(road).resize((256,256),Image.Resampling.NEAREST))
        overlay=np.zeros((256,256,3),np.uint8)+20;overlay[road]=[80,80,80]
        overlay[projected]=[70,150,220];overlay[expected]=[210,65,55];overlay[projected&expected]=[230,205,85]
        Image.fromarray(overlay[::-1]).save(d/'ACTUAL_MESH_MAP_OVERLAY.png')
        projection_metrics={'actual_triangle_projection_IoU':float((projected&expected).sum()/max(1,(projected|expected).sum())),
            'mesh_projection_outside_map':float((projected&~expected).sum()/max(1,projected.sum())),
            'map_overlay_legend':'yellow overlap; red required footprint; blue generated triangles; gray map road',
            'view':'fixed orthographic XY in [-0.5,0.5]; Z clip fixed [-2,2]; no reference plane included',
            'world_bounds':[v.min(0).tolist(),v.max(0).tolist()]}
        del clip,raster,glctx
        # A reference plane is visualization only, excluded from delivered AI mesh.
        plane=np.array([[-.5,-.5,p['reference_ground']],[.5,-.5,p['reference_ground']],
                        [.5,.5,p['reference_ground']],[-.5,.5,p['reference_ground']]],np.float32)
        vv=np.concatenate([v,plane]);ff=np.concatenate([f,np.array([[0,1,2],[0,2,3]],np.int32)+len(v)])
        mesh=Mesh(torch.from_numpy(vv).cuda(),torch.from_numpy(ff).cuda())
        vd=d/'views';vd.mkdir()
        def render(eye,look,up,fov):
            K,w2c=camera_matrices(eye,look,up,fov)
            with torch.no_grad():out=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),
                torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['normal','mask','depth'])
            mask=out['mask'].cpu().numpy();n=out['normal'].permute(1,2,0).cpu().numpy()*2-1
            light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
            s=.3+.7*np.abs((n*light).sum(-1)).clip(0,1)
            rgb=np.array([.76,.79,.82])*s[...,None]*mask[...,None]+np.array([.11,.15,.2])*(1-mask[...,None])
            return Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)),out
        for name,eye,look,up,fov in cameras:
            image,diag=render(eye,look,up,fov);image.save(vd/f'{name}.png')
            if name in ['street_middle','top']:
                np.savez_compressed(vd/f'{name}_depth.npz',depth=diag['depth'].cpu().numpy(),mask=diag['mask'].cpu().numpy())
        import imageio.v2 as imageio
        with imageio.get_writer(str(d/'lateral_comparison.mp4'),fps=24,codec='libx264',quality=7) as writer:
            for delta in np.linspace(-.035,.035,48):
                eye=np.array(midpoint['eye']);look=np.array(midpoint['look']);eye[0]+=delta;look[0]+=delta
                image,_=render(eye.tolist(),look.tolist(),midpoint['up'],midpoint['fov'])
                writer.append_data(np.asarray(image))
        save(d/'LAYOUT.json',{'arm':arm,'seed':seed,'objects':entries,'asset':str(asset),'sha256':sha(asset),
            'generated_count':sum(e['status']=='MESH_GENERATED' for e in entries),
            'fixed_map_placement':True,'generated_geometry_not_fixed_ground':True,
            'actual_triangle_projection':projection_metrics,
            'no_generated_materials':True,'neutral_geometry_inspection':True,
            'same_cameras_as_v15':True,'additional_lookback_camera':lookback,'geographic_registration':False})
        assets[arm]=d
        rows.extend([{'arm':arm,'seed':seed,**e} for e in entries]);del mesh;torch.cuda.empty_cache()
        print('STREET_SAVED',seed,arm,len(f),flush=True)
    names=['top','oblique','street_start','street_middle','street_end','look_back']
    sheet=Image.new('RGB',(1536,2*540),'#121923');draw=ImageDraw.Draw(sheet)
    for i,arm in enumerate(['baseline','repair']):
        for j,name in enumerate(names):
            sheet.paste(Image.open(assets[arm]/'views'/f'{name}.png').resize((256,256)),(j*256,i*540+25))
            draw.text((j*256+4,i*540+5),f'{arm} {name}',fill='white')
        source=Image.open(V15/'inputs/source_crop.png').convert('RGB').resize((256,256))
        sheet.paste(source,(0,i*540+282))
        sheet.paste(Image.open(assets[arm]/'ACTUAL_MESH_MAP_OVERLAY.png'),(256,i*540+282))
        draw.text((520,i*540+300),'Map10 source | fixed map placement | gray ground is reference only',fill='white')
        draw.text((520,i*540+330),'No generated materials; no geometry repair; same Shape weights',fill='white')
    sheet.save(OUT/f'STREET_COMPARISON_{seed}.jpg',quality=94)
    save(OUT/f'SCENE_COMPLETE_{seed}.json',{'rows':rows,'no_new_model_work':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();run(a.seed)

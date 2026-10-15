"""Independent process reloads saved GLBs; fixed views, no asset changes."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_edit_mesh_v9_prepare import OUT,SEEDS
from trellis2_map_entry_views_v2 import camera_matrices
from trellis2.representations import Mesh
from trellis2.renderers import MeshRenderer
from paths import read,save,sha
import argparse,numpy as np,torch,trimesh,cv2,gc
from PIL import Image,ImageDraw

AXIS=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
MODES=['correct','edit','oracleQ_diagnostic','codec_reference_diagnostic']
def load_mesh(folder):
    meta=read(folder/'ASSET.json');assert sha(meta['mesh'])==meta['sha256']
    scene=trimesh.load(folder/'geometry.glb',force='scene',process=False)
    assert len(scene.geometry)==1
    g=list(scene.geometry.values())[0];v=(g.vertices@AXIS).astype(np.float32);f=g.faces.astype(np.int32)
    with np.load(folder/'NATIVE_MESH.npz') as z:assert np.array_equal(f,z['faces']) and np.allclose(v,z['vertices'],atol=1e-7)
    return Mesh(torch.from_numpy(v).cuda(),torch.from_numpy(f).cuda()),v,f,meta

def footprint(v,f):
    out=np.zeros((64,64),np.uint8);xy=np.round((v[:,:2]+.5)*64-.5).astype(np.int32)
    # Actual projected triangles, not Gaussian/mesh centers.
    for tri in xy[f]:cv2.fillConvexPoly(out,tri,1)
    return out.astype(bool)

def run(seed):
    torch.set_num_threads(4);plan=read(OUT/'PLAN.json');root=OUT/f'seed{seed}'
    renderer=MeshRenderer();renderer.rendering_options.resolution=512;renderer.rendering_options.ssaa=1
    renderer.rendering_options.near=.001;renderer.rendering_options.far=10
    renderer.rendering_options.antialias=False;renderer.rendering_options.chunk_size=8000000
    rows=[]
    def render(mesh,eye,look,up,fov):
        K,w2c=camera_matrices(eye,look,up,fov)
        with torch.no_grad():p=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['normal','depth','mask'])
        mask=p['mask'].cpu().numpy();n=p['normal'].permute(1,2,0).cpu().numpy()
        # Renderer normals are [0,1] camera-space normals. Fixed display lighting.
        normal=n*2-1;light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
        shade=.28+.72*np.abs((normal*light).sum(-1)).clip(0,1)
        col=np.array([.76,.79,.82])*shade[...,None]
        rgb=col*mask[...,None]+np.array([.075,.09,.12])*(1-mask[...,None])
        return Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)),p,K,w2c
    for mode in MODES:
        if mode=='codec_reference_diagnostic' and seed!=SEEDS[0]:continue
        for i,r in enumerate(plan['records']):
            d=root/mode/f'object{i}'
            if not (d/'ASSET.json').exists():continue
            mesh,v,f,meta=load_mesh(d);vd=d/'views';vd.mkdir(exist_ok=False)
            fp=footprint(v,f)
            mr=next(x for x in plan['maps'] if x['mode']==('edit' if mode=='edit' else 'correct') and x['index']==i)
            with np.load(mr['map_input']) as z:expected=z['footprints'][r['target_object']]>.5
            rgb=np.zeros((64,64,3),np.uint8)+20;rgb[fp]=[135,150,165];rgb[expected]=[215,65,55];rgb[fp&expected]=[235,205,95]
            Image.fromarray(rgb[::-1]).resize((512,512),Image.Resampling.NEAREST).save(vd/'mesh_map_overlay.png')
            for name,eye,look,up,fov in plan['cameras']:
                image,p,K,w2c=render(mesh,eye,look,up,fov);image.save(vd/f'{name}.png')
                if name in ['front','lateral_left','top']:
                    np.savez_compressed(vd/f'{name}_diagnostics.npz',depth=p['depth'].cpu().numpy(),mask=p['mask'].cpu().numpy(),K=K,w2c=w2c)
            row={'mode':mode,'object_index':i,'training_seed':seed,'vertices':len(v),'faces':len(f),
              'projected_mesh_map_IoU':float((fp&expected).sum()/max(1,(fp|expected).sum())),
              'required_footprint_coverage':float((fp&expected).sum()/max(1,expected.sum())),
              'outside_footprint_fraction':float((fp&~expected).sum()/max(1,fp.sum())),
              'source_asset_hash':meta['sha256'],'exact_geometry_preserved_on_reload':True,
              'neutral_not_realistic_material_render':True,'AI_visual_review_pending':True}
            np.savez_compressed(vd/'MESH_PROJECTION.npz',projection=fp,expected=expected)
            save(vd/'RELOAD.json',row);rows.append(row)
            if mode in ['correct','edit']:
                import imageio.v2 as imageio
                with imageio.get_writer(str(vd/'lateral_motion.mp4'),fps=24,codec='libx264',quality=7) as writer:
                    for x in np.linspace(-.35,.35,48):
                        image,_,_,_=render(mesh,[float(x),-1,-.12],[0,0,-.12],[0,0,1],50);writer.append_data(np.asarray(image))
            del mesh;torch.cuda.empty_cache();gc.collect()
            print('INDEPENDENT_MESH_RELOAD',seed,mode,i,row['projected_mesh_map_IoU'],flush=True)
    save(root/'MESH_RESULTS.json',{'rows':rows,'no_geometry_quality_pass_from_IoU_alone':True})
    for i in range(2):
        sheet=Image.new('RGB',(1536,4*310),'#111924')
        names=['front','lateral_left','lateral_right','top','mesh_map_overlay']
        for mi,mode in enumerate(MODES):
            source=(OUT/f'seed{SEEDS[0]}' if mode=='codec_reference_diagnostic' else root)/mode/f'object{i}/views'
            for ci,name in enumerate(names):
                p=source/f'{name}.png'
                if p.exists():sheet.paste(Image.open(p).convert('RGB').resize((300,280)),(ci*306,mi*310+27))
            ImageDraw.Draw(sheet).text((4,mi*310+5),f'{mode} | object{i} | seed{seed} | FRONT / SIDE-L / SIDE-R / TOP / MAP',fill='white')
        sheet.save(root/f'object{i}_MATCHED_MESH.jpg',quality=94)
    save(root/'RENDER_COMPLETE.json',{'independent_process_reload':True,'fixed_cameras':True,'no_new_geometry':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);run(p.parse_args().seed)

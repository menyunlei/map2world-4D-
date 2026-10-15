"""CPU-only saved-mesh fallback. Re-render ALL comparison arms identically."""
import sys,os,json,math,subprocess,concurrent.futures,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'map_object_world_v1/trellis2_edit_identity_v10'
V9=HERE/'map_object_world_v1/trellis2_map_edit_mesh_v9'
SEEDS=[953101,953102]

def worker():
    import bpy
    from mathutils import Vector,Matrix
    sys.path.insert(0,str(HERE))
    import trellis2_saved_mesh_render_v2 as old
    args=old.parse_args(sys.argv[sys.argv.index('--')+1:])
    plan=json.loads((OUT/'PLAN.json').read_text())
    cameras={n:dict(eye=e,look=l,up=u,fov=f) for n,e,l,u,f in plan['cameras']}
    names=['front','lateral_left','lateral_right','back','top','oblique']
    old.SIZE=512;old.SAMPLES=4;old.FRAME_COUNT=0;old.MODES=('neutral',);old.VIEW_NAMES=names
    old.camera_poses=lambda *unused:(cameras,[],2.)
    def pose(camera,p,_Vector):
        forward=(Vector(p['look'])-Vector(p['eye'])).normalized()
        right=forward.cross(Vector(p['up'])).normalized();up=right.cross(forward)
        camera.matrix_world=Matrix((right,up,-forward)).transposed().to_4x4()
        camera.location=Vector(p['eye'])
        camera.data.lens=36/(2*math.tan(math.radians(p['fov'])/2))
    old.set_pose=pose;old.contact_sheet=lambda *unused:None;old.encode_videos=lambda *unused:{}
    old.render(args)
    (args.output/'CPU_CAMERA_CONTRACT.json').write_text(json.dumps({'fixed_v9_camera_poses':cameras,'views':names,'CPU_only':True,'no_mesh_repair':True,'all_three_arms_same_CPU_renderer':True},indent=2))

def main():
    import numpy as np,cv2,trimesh
    from PIL import Image
    from paths import read,save,sha
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=read(OUT/'PLAN.json');jobs=[]
    # Same 64x64 triangle projection as the frozen native-view evaluator.
    for seed in SEEDS:
        for i in range(2):
            d=OUT/f'seed{seed}/edit/object{i}';meta=read(d/'ASSET.json')
            assert sha(d/'geometry.glb')==meta['sha256']
            sc=trimesh.load(d/'geometry.glb',force='scene',process=False);g=list(sc.geometry.values())[0]
            axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
            with np.load(d/'NATIVE_MESH.npz') as z:v=z['vertices'].copy();f=z['faces'].copy()
            assert np.allclose(g.vertices@axis,v,atol=1e-7) and np.array_equal(g.faces,f)
            fp=np.zeros((64,64),np.uint8);xy=np.round((v[:,:2]+.5)*64-.5).astype(np.int32)
            for tri in xy[f]:cv2.fillConvexPoly(fp,tri,1)
            fp=fp.astype(bool);r=p['records'][i]
            mr=next(x for x in p['maps'] if x['mode']=='edit' and x['index']==i)
            with np.load(mr['map_input']) as z:expected=z['footprints'][r['target_object']]>.5
            vd=d/'views';vd.mkdir(exist_ok=False)
            np.savez_compressed(vd/'MESH_PROJECTION.npz',projection=fp,expected=expected)
            rgb=np.zeros((64,64,3),np.uint8)+20;rgb[fp]=[135,150,165];rgb[expected]=[215,65,55];rgb[fp&expected]=[235,205,95]
            Image.fromarray(rgb[::-1]).resize((512,512),Image.Resampling.NEAREST).save(vd/'mesh_map_overlay.png')
            save(vd/'RELOAD_CPU.json',{'exact_geometry_preserved_on_reload':True,'no_GPU':True,'source_sha256':meta['sha256']})
            for label,root,mode in [('original',V9,'correct'),('old_edit',V9,'edit'),('new_edit',OUT,'edit')]:
                jobs.append((root/f'seed{seed}/{mode}/object{i}/geometry.glb',OUT/f'cpu_views/seed{seed}/{label}/object{i}'))
    blender=HERE.parents[1]/'experiments/mapworld_dit/real_target_gate_v1/tools/blender-4.5.9-linux-x64/blender'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':str(HERE.parents[1]/'experiments/mapworld_dit/real_target_gate_v1/tools/aptdeps/root/usr/lib/x86_64-linux-gnu')}
    def render(job):
        asset,d=job;d.mkdir(parents=True,exist_ok=True)
        with (d/'blender.log').open('x') as h:
            subprocess.run([str(blender),'-b','--factory-startup','--disable-autoexec','-t','4','--python-exit-code','1','--python',__file__,'--','--asset',str(asset),'--output',str(d),'--bounds','{"min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]}','--skip-video'],stdout=h,stderr=subprocess.STDOUT,env=env,check=True,timeout=240)
        print('CPU_RELOAD_RENDER_COMPLETE',d,flush=True)
    t=time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(render,jobs))
    # Preserve numerical scoring verbatim; only contact-sheet image locations change.
    import trellis2_edit_identity_v10_score as score
    original_open=score.Image.open
    def open_panel(path,*a,**kw):
        path=Path(path)
        if path.name!='mesh_map_overlay.png' and path.parent.name=='views':
            base=V9 if path.is_relative_to(V9) else OUT
            seed,mode,obj,_,name=path.relative_to(base).parts
            label='original' if mode=='correct' else 'old_edit' if base==V9 else 'new_edit'
            path=OUT/'cpu_views'/seed/label/obj/'neutral'/name
        return original_open(path,*a,**kw)
    score.Image.open=open_panel;score.main()
    save(OUT/'CPU_RENDER_COMPLETE.json',{'cpu_wall_seconds':time.time()-t,'new_GPU_seconds':0,'all_arms_rerendered':True,'geometry_modified':False,'no_video_regeneration':True})
    save(OUT/'COMPLETE.json',{'training_updates':0,'no_queued_jobs':True,'GPU_seconds':sum(read(x)['gpu_seconds'] for x in (OUT/'gpu_cost').glob('*.json')),'native_GPU_render_blocked_by_other_jobs':'CPU Blender matched fallback complete'})

if __name__=='__main__':
    if '--' in sys.argv:worker()
    elif '--detach' in sys.argv:
        with (OUT/'cpu_render.log').open('x') as h:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=h,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print('CPU_RENDER_OWNER',child.pid,flush=True)
    else:main()

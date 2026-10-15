"""Independent CPU Blender rendering of saved GLBs; fixed input-based cameras."""
import os,sys,json,math,subprocess,concurrent.futures,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'map_object_world_v1/trellis2_map10_block_v11'
SEEDS=[953101,953102]

def worker():
    import bpy,numpy as np
    from mathutils import Vector,Matrix
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--mode',required=True)
    args=a.parse_args(sys.argv[sys.argv.index('--')+1:]);p=json.loads((OUT/'PLAN.json').read_text());d=OUT/f'seed{args.seed}'/args.mode
    vd=d/'views';vd.mkdir(exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(d/'layout.glb'))
    generated=[o for o in bpy.context.scene.objects if o.type=='MESH']
    for o in generated:o['source']='AI generated TRELLIS.2 geometry; fixed map transform'
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=4
    s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.cycles.seed=0;s.cycles.max_bounces=4
    s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
    s.view_settings.view_transform='Standard';s.view_settings.look='None';s.render.film_transparent=False
    world=bpy.data.worlds.new('fixed_world');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.18,.18,.18,1);world.node_tree.nodes['Background'].inputs[1].default_value=.65;s.world=world
    material=bpy.data.materials.new('neutral_generated_geometry');material.use_nodes=True
    shader=material.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.59,.65,.72,1);shader.inputs['Roughness'].default_value=.8
    for o in generated:o.data.materials.clear();o.data.materials.append(material)
    for name,loc,power,size in [('key',[-2,-3,4],350,3),('fill',[3,-1,2],160,4),('rim',[1,3,4],180,3)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
        o=bpy.data.objects.new(name,data);s.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector([0,0,-.3])-o.location).to_track_quat('-Z','Y').to_euler()
    # Separate map-only flat reference, never exported into the generated asset.
    with np.load(p['global_maps'][args.mode]) as z:roads=z['roads'][0]>.5;known=z['roads_known'][0]>0
    rgba=np.ones((64,64,4),np.float32);rgba[...,:3]=[.20,.24,.22];rgba[roads&known,:3]=[.11,.12,.13]
    image=bpy.data.images.new('MAP_REFERENCE_ROADS_NOT_GENERATED',width=64,height=64);image.pixels.foreach_set(rgba.ravel())
    mat=bpy.data.materials.new('MAP_REFERENCE_NOT_MODEL_OUTPUT');mat.use_nodes=True
    nodes=mat.node_tree.nodes;tex=nodes.new('ShaderNodeTexImage');tex.image=image;tex.interpolation='Closest'
    shader=nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=1;mat.node_tree.links.new(tex.outputs['Color'],shader.inputs['Base Color'])
    bpy.ops.mesh.primitive_plane_add(size=1,location=(0,0,p['reference_ground']-.0002))
    plane=bpy.context.object;plane.name='MAP_REFERENCE_GROUND_ROAD_VISUALIZATION_ONLY';plane.data.materials.append(mat)
    camdata=bpy.data.cameras.new('fixed_camera');cam=bpy.data.objects.new('fixed_camera',camdata);s.collection.objects.link(cam);s.camera=cam
    camdata.clip_start=.0001;camdata.clip_end=20;camdata.sensor_fit='HORIZONTAL';camdata.sensor_width=36
    def pose(v):
        direction=(Vector(v['look'])-Vector(v['eye'])).normalized();right=direction.cross(Vector(v['up'])).normalized();up=right.cross(direction)
        cam.matrix_world=Matrix((right,up,-direction)).transposed().to_4x4();cam.location=Vector(v['eye']);camdata.lens=36/(2*math.tan(math.radians(v['fov'])/2))
    for name,eye,look,up,fov in p['cameras']:
        pose(dict(eye=eye,look=look,up=up,fov=fov));s.render.filepath=str(vd/f'{name}.png');bpy.ops.render.render(write_still=True)
    # One top view without the visualization plane makes source separation explicit.
    plane.hide_render=True;v=next(c for c in p['cameras'] if c[0]=='top');pose(dict(zip(['eye','look','up','fov'],v[1:])))
    s.render.filepath=str(vd/'top_generated_only.png');bpy.ops.render.render(write_still=True);plane.hide_render=False
    frames=vd/'lateral';frames.mkdir()
    for k,v in enumerate(p['lateral']):
        pose(v);s.render.filepath=str(frames/f'{k:03d}.png');bpy.ops.render.render(write_still=True)
    (vd/'RENDER.json').write_text(json.dumps({'asset':str(d/'layout.glb'),'CPU_only':True,'independent_saved_asset_reload':True,
      'generated_mesh_count':len(generated),'polygons':sum(len(o.data.polygons) for o in generated),'geometry_repair':False,
      'neutral_material_not_generated_appearance':True,'ground_roads':'separate flat input-map visualization, NOT generated asset',
      'cameras':p['cameras'],'lateral':p['lateral'],'source_contract':'Blender imports glTF Y-up to native Z-up; global map coordinates unchanged'},indent=2))
    print('SAVED_LAYOUT_RENDERED',args.seed,args.mode,flush=True)

def owner():
    import hashlib
    p=json.loads((OUT/'PLAN.json').read_text());start=time.time()
    blender=HERE.parents[1]/'experiments/mapworld_dit/real_target_gate_v1/tools/blender-4.5.9-linux-x64/blender'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':str(HERE.parents[1]/'experiments/mapworld_dit/real_target_gate_v1/tools/aptdeps/root/usr/lib/x86_64-linux-gnu')}
    def one(seed,mode):
        d=OUT/f'seed{seed}'/mode;layout=json.loads((d/'LAYOUT.json').read_text())
        assert hashlib.sha256((d/'layout.glb').read_bytes()).hexdigest()==layout['sha256']
        with (d/'blender.log').open('x') as h:
            subprocess.run([str(blender),'-b','--factory-startup','--disable-autoexec','-t','4','--python-exit-code','1','--python',__file__,'--','--seed',str(seed),'--mode',mode],stdout=h,stderr=subprocess.STDOUT,env=env,check=True,timeout=900)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        fs=[pool.submit(one,s,m) for s in SEEDS for m in ['correct','edit']]
        for f in fs:f.result()
    (OUT/'RENDER_COMPLETE.json').write_text(json.dumps({'cpu_wall_seconds':time.time()-start,'GPU_seconds':0,'all_four_layouts':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':
    if '--' in sys.argv:worker()
    elif '--detach' in sys.argv:
        with (OUT/'render_owner.log').open('x') as h:c=subprocess.Popen([sys.executable,'-B','-u',__file__],stdout=h,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print('CPU_LAYOUT_RENDER_OWNER',c.pid,flush=True)
    else:owner()

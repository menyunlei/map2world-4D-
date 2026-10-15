"""Fresh Blender asset load; fixed street cameras, GPU Cycles and synchronized map."""
from pathlib import Path
import os,sys,json,math,subprocess,argparse,time
HERE=Path(__file__).resolve().parent
OUT=HERE/'map_object_world_v1/trellis2_map10_street_v15'

def worker(seed):
    import bpy,numpy as np
    from mathutils import Vector,Matrix
    p=json.loads((OUT/'PLAN.json').read_text());d=OUT/f'seed{seed}/correct'
    vd=d/'street_views';vd.mkdir(exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    asset=d/'street_buildings.glb' if (d/'PAINT_COMPLETE.json').exists() else d/'layout.glb'
    bpy.ops.import_scene.gltf(filepath=str(asset))
    generated=[o for o in bpy.context.scene.objects if o.type=='MESH']
    layout=json.loads((d/'LAYOUT.json').read_text())
    assert len(generated)==layout['generated_count']
    for j,o in enumerate(sorted(generated,key=lambda x:x.name)):
        o['source']='AI generated geometry; fixed map frame; appearance present only when PAINT_COMPLETE exists'
        o.pass_index=j+1
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    prefs=bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type='CUDA';prefs.get_devices()
        devices=[x for x in prefs.devices if x.type=='CUDA']
        for device in prefs.devices:device.use=device.type=='CUDA'
    except Exception:devices=[]
    scene.cycles.device='GPU' if devices else 'CPU'
    scene.cycles.samples=8;scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_denoising=False;scene.cycles.seed=0;scene.cycles.max_bounces=4
    scene.render.resolution_x=640;scene.render.resolution_y=360;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='Standard'
    scene.view_settings.look='None';scene.render.film_transparent=False
    world=bpy.data.worlds.new('daylight');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(.35,.47,.65,1)
    world.node_tree.nodes['Background'].inputs[1].default_value=.45;scene.world=world
    sun=bpy.data.lights.new('fixed_sun','SUN');sun.energy=2;sun.angle=.12
    light=bpy.data.objects.new('fixed_sun',sun);scene.collection.objects.link(light)
    light.rotation_euler=(math.radians(28),math.radians(-18),math.radians(-25))
    with np.load(p['global_maps']['correct']) as z:road=z['roads'][0]>.5
    rgba=np.ones((64,64,4),np.float32);rgba[...,:3]=[.24,.29,.22];rgba[road,:3]=[.13,.14,.15]
    image=bpy.data.images.new('MAP_ROAD_REFERENCE',width=64,height=64)
    image.pixels.foreach_set(rgba.ravel());image.pack()
    material=bpy.data.materials.new('MAP_GROUND_ROAD_REFERENCE');material.use_nodes=True
    tex=material.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;tex.interpolation='Linear'
    shader=material.node_tree.nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.95
    material.node_tree.links.new(tex.outputs['Color'],shader.inputs['Base Color'])
    bpy.ops.mesh.primitive_plane_add(size=1,location=(0,0,p['reference_ground']-.0002))
    plane=bpy.context.object;plane.name='MAP_REFERENCE_ROAD_GROUND_NOT_AI_GENERATED'
    plane['source']='Map-derived background visualization';plane.data.materials.append(material);plane.pass_index=0
    camdata=bpy.data.cameras.new('route_camera');camera=bpy.data.objects.new('route_camera',camdata)
    scene.collection.objects.link(camera);scene.camera=camera
    camdata.clip_start=.0001;camdata.clip_end=20;camdata.sensor_width=36;camdata.sensor_fit='HORIZONTAL'
    def pose(v):
        direction=(Vector(v['look'])-Vector(v['eye'])).normalized()
        right=direction.cross(Vector(v['up'])).normalized();up=right.cross(direction)
        camera.matrix_world=Matrix((right,up,-direction)).transposed().to_4x4()
        camera.location=Vector(v['eye']);camdata.lens=36/(2*math.tan(math.radians(v['fov'])/2))
    scene.view_layers[0].use_pass_z=True;scene.view_layers[0].use_pass_object_index=True
    scene.use_nodes=True;nodes=scene.node_tree.nodes;nodes.clear();rl=nodes.new('CompositorNodeRLayers')
    composite=nodes.new('CompositorNodeComposite');scene.node_tree.links.new(rl.outputs['Image'],composite.inputs[0])
    diagnostic=[]
    for socket,label in [('Depth','depth'),('IndexOB','building_id')]:
        n=nodes.new('CompositorNodeOutputFile');n.base_path=str(vd/'diagnostics')
        n.format.file_format='OPEN_EXR';n.format.color_depth='32';n.format.color_mode='RGB'
        scene.node_tree.links.new(rl.outputs[socket],n.inputs[0]);diagnostic.append((n,label))
    for name,eye,look,up,fov in p['cameras']:
        pose(dict(eye=eye,look=look,up=up,fov=fov))
        for n,label in diagnostic:n.file_slots[0].path=name+'_'+label+'_'
        scene.render.filepath=str(vd/f'{name}.png');bpy.ops.render.render(write_still=True)
    for n,_ in diagnostic:n.mute=True
    frames=vd/'frames';frames.mkdir()
    for k,v in enumerate(p['trajectory']):
        pose(v);scene.render.filepath=str(frames/f'{k:04d}.png');bpy.ops.render.render(write_still=True)
    for o in generated:o['source']='TRELLIS.2 learned map-conditioned geometry; fixed map frame'
    bpy.ops.wm.save_as_mainfile(filepath=str(d/'street_world.blend'))
    (vd/'RENDER.json').write_text(json.dumps({'asset':str(asset),
        'independent_saved_asset_load':True,'generated_meshes':len(generated),
        'faces':sum(len(o.data.polygons) for o in generated),'device':scene.cycles.device,
        'CUDA_devices':[x.name for x in devices],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'frames':len(p['trajectory']),'fps':24,'camera_poses':p['trajectory'],
        'geographic_registration':False,'normalized_coordinates_not_measured_meters':True,
        'geometry_repair':False,'reference_plane_separate':True,
        'appearance_generated':(d/'PAINT_COMPLETE.json').exists(),'expected_buildings':4,
        'missing_buildings':[r['object_id'] for r in layout['objects'] if r['status']!='MESH_GENERATED'],
        'building_id_mapping':{str(o.pass_index):o.name for o in generated}},indent=2))

def owner(seed):
    from PIL import Image,ImageDraw
    import numpy as np,hashlib,shutil
    p=json.loads((OUT/'PLAN.json').read_text());d=OUT/f'seed{seed}/correct'
    painted=(d/'PAINT_COMPLETE.json').exists()
    meta=json.loads((d/('PAINT_COMPLETE.json' if painted else 'LAYOUT.json')).read_text())
    asset=d/('street_buildings.glb' if painted else 'layout.glb')
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==meta['sha256']
    root=HERE.parents[1]
    blender=root/'experiments/mapworld_dit/real_target_gate_v1/tools/blender-4.5.9-linux-x64/blender'
    env={**os.environ,'LD_LIBRARY_PATH':str(root/'experiments/mapworld_dit/real_target_gate_v1/tools/aptdeps/root/usr/lib/x86_64-linux-gnu')}
    with (d/'street_blender.log').open('x') as h:
        subprocess.run([str(blender),'-b','--factory-startup','--disable-autoexec','-t','4',
            '--python-exit-code','1','--python',__file__,'--','--seed',str(seed)],stdout=h,stderr=subprocess.STDOUT,env=env,check=True)
    vd=d/'street_views';mini=Image.open(OUT/'inputs/source_crop.png').convert('RGB').resize((156,156))
    route=[tuple((np.asarray(v['source_crop_xy'])*156/910).tolist()) for v in p['trajectory']]
    dest=vd/'frames_minimap';dest.mkdir(exist_ok=False)
    for k,v in enumerate(p['trajectory']):
        image=Image.open(vd/'frames'/f'{k:04d}.png').convert('RGB');mp=mini.copy();draw=ImageDraw.Draw(mp)
        draw.line(route,fill=(255,230,40),width=2);x,y=route[k]
        draw.ellipse((x-3,y-3,x+3,y+3),fill=(255,35,25))
        image.paste(mp,(476,8));draw=ImageDraw.Draw(image)
        draw.rectangle((0,340,640,360),fill='#111923')
        caption='Map10 approximate XY | AI buildings | map-derived road/ground' if painted else 'INCOMPLETE: building04 empty | neutral AI geometry | map reference road'
        draw.text((6,345),caption,fill='white')
        image.save(dest/f'{k:04d}.png')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    video=d/'street_8s_24fps.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-framerate','24','-i',str(dest/'%04d.png'),
        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    sheet=Image.new('RGB',(1280,1080),'#121922');draw=ImageDraw.Draw(sheet)
    for k,name in enumerate(['top','oblique','street_start','street_middle','street_end']):
        image=Image.open(vd/f'{name}.png');sheet.paste(image,((k%2)*640,(k//2)*360))
        draw.text(((k%2)*640+8,(k//2)*360+8),name,fill='white')
    sheet.save(d/'street_contact.jpg',quality=94)
    (d/'STREET_COMPLETE.json').write_text(json.dumps({'video':str(video),'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'frames':192,'fps':24,'duration_seconds':8,'unique_camera_positions':len({tuple(v['eye']) for v in p['trajectory']}),
        'asset':str(asset),'Blender_scene':str(d/'street_world.blend'),'appearance_generated':painted,
        'right_side_not_generated':True,'actual_asset_quality_review':'PENDING'},indent=2))
    print('STREET_VIDEO_COMPLETE',seed,video,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[953101,953102],required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    worker(args.seed) if '--' in sys.argv else owner(args.seed)

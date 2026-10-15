"""Fixed Blender reference images; diagnostic inputs, never model-generated claims."""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

DEST=Path(__file__).resolve().parent/'trellis2_native_quality_v1'
name=sys.argv[sys.argv.index('--')+1]
out=DEST/'inputs'/name
record=json.loads((out/'INPUT.json').read_text())
assert not (out/'view0.png').exists()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(out/'reference.glb'))
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=32
s.cycles.use_denoising=False;s.render.film_transparent=True
s.render.resolution_x=1024;s.render.resolution_y=1024;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA'
s.view_settings.view_transform='Standard';s.world.use_nodes=True
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.8
for location,power,size in [((3,-4,5),450,4),((-3,1,3),250,3)]:
    bpy.ops.object.light_add(type='AREA',location=location);light=bpy.context.object
    light.data.energy=power;light.data.shape='DISK';light.data.size=size
    light.rotation_euler=(Vector((0,0,0))-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add();cam=bpy.context.object;s.camera=cam;cam.data.clip_start=.005;cam.data.clip_end=10
for k,pose in enumerate(record['cameras']):
    cam.location=pose['eye'];cam.rotation_euler=(Vector(pose['look'])-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.angle=math.radians(pose['fov_degrees']);s.render.filepath=str(out/f'view{k}.png')
    bpy.ops.render.render(write_still=True)
print('REFERENCE_VIEWS_READY',name,flush=True)

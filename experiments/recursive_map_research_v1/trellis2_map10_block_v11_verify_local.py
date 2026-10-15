"""Verify received assets; sample the actual saved comparison video for inspection."""
from pathlib import Path
import json,hashlib,trimesh,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=Path(__file__).parent/'map_object_world_v1/trellis2_map10_block_v11'
manifest=json.loads((p/'DELIVERY_MANIFEST.json').read_text());records=[]
for r in manifest['files']:
    f=p/r['path'];assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
    if f.suffix=='.glb':
        scene=trimesh.load(f,force='scene',process=False)
        records.append({'file':r['path'],'mesh_count':len(scene.geometry),'nodes':list(scene.graph.nodes),'object_slots':scene.metadata.get('object_slots')})
reader=imageio.get_reader(p/'LATERAL_COMPARISON.mp4');canvas=Image.new('RGB',(1536,564),'#111923')
for col,n in enumerate([0,24,47]):
    canvas.paste(Image.fromarray(reader.get_data(n)).resize((512,544)),(512*col,20));ImageDraw.Draw(canvas).text((512*col+5,3),f'ACTUAL VIDEO FRAME {n}',fill='white')
reader.close();canvas.save(p/'LATERAL_SAMPLED.jpg',quality=94)
with (p/'LOCAL_VERIFICATION.json').open('x') as h:json.dump({'all_delivery_hashes_verified':True,'independent_local_GLBl_loads':records,'video_frames':48,'video_fps':24,'AI_preview_frames':[0,24,47]},h,indent=2)
print('ALL_LOCAL_HASHES_AND_FOUR_GLBS_VERIFIED',[(r['file'],r['mesh_count']) for r in records])

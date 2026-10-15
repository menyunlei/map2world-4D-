"""Fixed existing-weight before/after evidence and cost; no subsequent experiment."""
from pathlib import Path
from trellis2_map_volume_v19 import OUT,OLD,JOB,read,save,sha
from PIL import Image,ImageDraw
assert (OUT/'VIEWS_COMPLETE.json').exists()
save(OUT/'COMPLETE.json',{'one_correction_complete':True,'no_further_jobs':True})
source=Path(__file__).with_name('trellis2_mapguided_v18_finish.py')
code=source.read_text().replace('from trellis2_mapguided_v18 import OUT,V11,V17',
    'from trellis2_map_volume_v19 import OUT\nfrom trellis2_mapguided_v18 import V11,V17')
exec(compile(code,str(source),'exec'))
comparison=[]
for mode in ['correct','edit']:
    row={'mode':mode}
    for label,root in [('v18',OLD),('v19',OUT)]:
        s=read(root/mode/'SS_METRICS.json');m=read(root/mode/'MESH_MAP.json');a=read(root/mode/'ASSET.json')
        row[label]={'Structure_IoU':s['above_ground_map_IoU'],'Structure_outside_fraction':s['outside_fraction'],
            'Structure_extent':s['native_extent_cells'],'actual_mesh_IoU':m['above_floor_projection_IoU'],
            'actual_mesh_outside_fraction':m['outside_fraction'],'vertices':a['vertices'],'faces':a['faces'],
            'mesh_bounds':a['native_bounds'],'asset_sha256':a['sha256']}
    comparison.append(row)
save(OUT/'MATCHED_COMPARISON.json',{'rows':comparison,'diagnosis':read(OUT/'DIAGNOSIS.json'),
    'one_change':'Heightwise free-space gradient allocation at unchanged per-column logit-gradient L1',
    'scalar_losses_not_comparable':True,'training_updates':0,'same_noise_map_camera_shape_texture_contract':True,
    'old_assets_preserved_and_reused':True,'native_3D_assets_and_visual_inspection_required':True})
columns=['actual_mesh_map_overlay.png','views/appearance/oblique.png','views/appearance/street_front.png','views/appearance/street_side.png']
sheet=Image.new('RGB',(1280,1050),'#171d25');draw=ImageDraw.Draw(sheet)
for y,(label,root,mode) in enumerate([('v18 original',OLD,'correct'),('v19 original',OUT,'correct'),('v19 edit',OUT,'edit')]):
    for x,name in enumerate(columns):
        sheet.paste(Image.open(root/mode/name).convert('RGB').resize((320,320)),(320*x,y*350+30))
        draw.text((320*x+4,y*350+5),label+' / '+['map overlay','oblique','street','side'][x],fill='white')
sheet.save(OUT/'BEFORE_AFTER_EDIT.jpg',quality=94)
print('MATCHED_NATIVE_ASSET_COMPARISON',comparison,flush=True)

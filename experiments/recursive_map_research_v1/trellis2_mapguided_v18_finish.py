"""Small fixed original/edit evidence pack; no model work or new viewer."""
from trellis2_mapguided_v18 import OUT,V11,V17
from paths import read,save,sha,JOB
from PIL import Image,ImageDraw
import numpy as np,tarfile

assert (OUT/'COMPLETE.json').exists()
p=read(OUT/'PLAN.json');rows=[]
for mode in ['correct','edit']:
    a=read(OUT/mode/'ASSET.json');r=read(OUT/mode/'RELOAD.json')
    assert sha(a['asset'])==a['sha256'] and r['asset_sha256']==a['sha256']
    rows.append({'mode':mode,'asset':a,'Structure':read(OUT/mode/'SS_METRICS.json'),
        'actual_mesh_projection':read(OUT/mode/'MESH_MAP.json'),
        'native_load_and_sampler_record':str(OUT/mode/'LOAD.json')})
assert read(OUT/'correct/RELOAD.json')['views']==read(OUT/'edit/RELOAD.json')['views']
assert rows[0]['Structure']['SS_noise_sha256']==rows[1]['Structure']['SS_noise_sha256']
maps={};support={}
for mode in ['correct','edit']:
    with np.load(p['maps'][mode]['path']) as z:maps[mode]=z['footprints']>.5
    with np.load(OUT/mode/'SS.npz') as z:support[mode]=z['support64'].copy()
changed_input=(maps['correct']!=maps['edit']).any(0)
projections={k:v[:,:,p['guidance']['building_z_start']:].any(-1).T for k,v in support.items()}
changed_output=projections['correct']!=projections['edit']
edit={'changed_input_pixels':int(changed_input.sum()),'SS_projection_changed_pixels':int(changed_output.sum()),
    'projection_changes_outside_edited_pixels_fraction':float((changed_output&~changed_input).sum()/max(1,changed_output.sum())),
    'removed_footprint_region_clear_fraction':float((~projections['edit']&changed_input).sum()/max(1,changed_input.sum())),
    'not_a_local_edit_pass_by_itself':True,'same_XY_frame_no_object_placement':True}

columns=['actual_mesh_map_overlay.png','views/appearance/oblique.png',
    'views/appearance/street_front.png','views/appearance/street_side.png']
sheet=Image.new('RGB',(1280,700),'#171d25');d=ImageDraw.Draw(sheet)
for y,mode in enumerate(['correct','edit']):
    for x,name in enumerate(columns):
        im=Image.open(OUT/mode/name).convert('RGB').resize((320,320))
        sheet.paste(im,(x*320,y*350+30));d.text((x*320+4,y*350+5),f'{mode}: '+['map red / mesh blue','oblique','street front','street side'][x],fill='white')
sheet.save(OUT/'ORIGINAL_EDIT_COMPARISON.jpg',quality=94)
cost=sum(read(f)['gpu_seconds'] for f in (OUT/'gpu_cost').glob('*.json'))
assert cost<=p['GPU_cap_seconds']
save(OUT/'RESULTS.json',{'test':'One zero-training map-guided native-prior sample and fixed footprint edit',
    'scope':'Map10 development demonstration, not generalization or a trained map-module claim',
    'records':rows,'edit_diagnostic':edit,'GPU_seconds_including_load_idle':cost,'GPU_hours':cost/3600,
    'no_model_training':True,'no_geometry_repair':True,'historical_v17_preserved':True,
    'visual_acceptance':'Pending actual AI image inspection; not human acceptance',
    'map_not_real_metric_ground_truth':True,'roads':'No procedural road/ground added; ground and pathways from native generation',
    'appearance':'Official v17 stylized building reference, not real Map10 facades',
    'no_further_inference_training_or_tuning':True})
files=[f for f in OUT.rglob('*') if f.is_file() and (f.suffix in ['.json','.log','.jpg','.png','.mp4'] or f.name=='SS.npz')
       and 'frames' not in f.parts and 'work' not in f.parts]
with tarfile.open(OUT/'evidence.tar.gz','x:gz') as t:
    for f in files:t.add(f,arcname=f.relative_to(OUT))
save(OUT/'DELIVERY.json',{'assets':[{'path':r['asset']['asset'],'sha256':r['asset']['sha256']} for r in rows],
    'evidence_sha256':sha(OUT/'evidence.tar.gz'),'all_output_count':len(files),'GPU_seconds':cost,
    'stopped':True,'no_automatic_next_run':True})
print('FIXED_MAP_GUIDANCE_DELIVERY',cost,edit,flush=True)

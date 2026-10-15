"""Compact paired geometry evidence; never changes thresholds or model outputs."""
from trellis2_map_edit_mesh_v9_prepare import OUT,SEEDS,V8
from paths import read,save,sha
import numpy as np,csv
from PIL import Image,ImageDraw

def frac(a,mask):return float(a[mask].mean()) if mask.any() else None
def main():
    plan=read(OUT/'PLAN.json');pairs=[];assets=[]
    for seed in SEEDS:
        root=OUT/f'seed{seed}';assert (root/'RENDER_COMPLETE.json').exists()
        for i,r in enumerate(plan['records']):
            values={}
            for mode in ['correct','edit']:
                d=root/mode/f'object{i}'
                with np.load(d/'SS_GENERATED.npz') as z:ss=z['native_support64'].copy();latent=z['latent'].copy()
                with np.load(d/'views/MESH_PROJECTION.npz') as z:proj=z['projection'].copy();fp=z['expected'].copy()
                with np.load(d/'NATIVE_MESH.npz') as z:v=z['vertices'].copy();f=z['faces'].copy()
                values[mode]=(ss,latent,proj,fp,v,f)
            a,b=values['correct'],values['edit'];removed=a[3]&~b[3];retained=a[3]&b[3]
            row={'seed':seed,'object_index':i,'object_id':r['object_id'],'edited_object':i==0,
              'SS_projection_IoU_original':read(root/'correct'/f'object{i}/SS_METRICS.json')['map_projection_IoU'],
              'SS_projection_IoU_edit':read(root/'edit'/f'object{i}/SS_METRICS.json')['map_projection_IoU'],
              'SS_exactly_unchanged':bool(np.array_equal(a[0],b[0])),
              'SS_symmetric_change_over_union':float((a[0]^b[0]).sum()/max(1,(a[0]|b[0]).sum())),
              'SS_removed_area_occupancy_before':frac(a[0].any(2).T,removed),'SS_removed_area_occupancy_after':frac(b[0].any(2).T,removed),
              'mesh_removed_area_occupancy_before':frac(a[2],removed),'mesh_removed_area_occupancy_after':frac(b[2],removed),
              'mesh_retained_area_coverage_before':frac(a[2],retained),'mesh_retained_area_coverage_after':frac(b[2],retained),
              'mesh_projection_IoU_original':float((a[2]&a[3]).sum()/max(1,(a[2]|a[3]).sum())),
              'mesh_projection_IoU_edit':float((b[2]&b[3]).sum()/max(1,(b[2]|b[3]).sum())),
              'mesh_projection_change_over_union':float((a[2]^b[2]).sum()/max(1,(a[2]|b[2]).sum())),
              'mesh_vertices_faces_exactly_unchanged':bool(np.array_equal(a[4],b[4]) and np.array_equal(a[5],b[5])),
              'frame_unchanged':True}
            maps=[]
            for mode in ['correct','edit']:
                mr=next(x for x in plan['maps'] if x['mode']==mode and x['index']==i)
                with np.load(mr['map_input']) as z:maps.append({k:z[k].copy() for k in z.files})
            row['changed_input_fields']=[k for k in maps[0] if not np.array_equal(maps[0][k],maps[1][k])]
            row['own_footprint_unchanged']=bool(np.array_equal(maps[0]['footprints'][r['target_object']],maps[1]['footprints'][r['target_object']]))
            row['independent_world_evidence']=False
            pairs.append(row)
            # Regenerate only the contact sheet to include both seeds' completed reference rows.
            modes=['correct','edit','oracleQ_diagnostic','codec_reference_diagnostic']
            names=['front','lateral_left','lateral_right','top','mesh_map_overlay']
            sheet=Image.new('RGB',(1536,1240),'#111924')
            for mi,mode in enumerate(modes):
                src=(OUT/f'seed{SEEDS[0]}' if mode=='codec_reference_diagnostic' else root)/mode/f'object{i}/views'
                for ci,name in enumerate(names):sheet.paste(Image.open(src/f'{name}.png').convert('RGB').resize((300,280)),(ci*306,mi*310+27))
                ImageDraw.Draw(sheet).text((4,mi*310+5),f'{mode} | object{i} | seed{seed} | FRONT / SIDE-L / SIDE-R / TOP / MAP',fill='white')
            sheet.save(root/f'object{i}_FINAL_COMPARISON.jpg',quality=94)
        for p in root.glob('*/object*/ASSET.json'):assets.append(read(p))
    cost=[read(p) for p in (OUT/'gpu_cost').glob('*.json')]
    save(OUT/'MATCHED_RESULTS.json',{'pairs':pairs,'mesh_assets':assets,
      'GPU_seconds':sum(r['gpu_seconds'] for r in cost),'GPU_cost_includes_failed_load_idle':True,
      'old_v8_failure_preserved':True,'TRAIN_diagnostic_not_real_map_success':True,
      'geometry_projection_not_solid_or_appearance_acceptance':True,
      'oracle_cases_diagnostic_only':True,'no_more_work_queued':True})
    for name,rows in [('PAIR_METRICS.csv',pairs),('GPU_COST.csv',cost)]:
        with (OUT/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    freeze={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.glb','.npz','.json','.mp4','.py']}
    save(OUT/'DELIVERY_MANIFEST.json',freeze)
    print('MATCHED_EDIT_MESH_FINALIZED',len(assets),sum(r['gpu_seconds'] for r in cost),flush=True)

if __name__=='__main__':main()

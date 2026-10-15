"""Fixed matched edit checks. No claim of real-map or global generation success."""
from trellis2_edit_identity_v10_prepare import OUT,V9,SEEDS
from paths import read,save,sha
from PIL import Image,ImageDraw
import numpy as np,cv2,csv

def load(d):
    with np.load(d/'SS_GENERATED.npz') as z:q=z['native_support64'].copy()
    with np.load(d/'views/MESH_PROJECTION.npz') as z:proj=z['projection'].copy();fp=z['expected'].copy()
    a=read(d/'ASSET.json');height=a['native_bounds'][1][2]-a['native_bounds'][0][2]
    return q,proj,fp,height
def change(a,b,mask):
    mask=np.broadcast_to(mask.T[:,:,None],a.shape)
    return float(((a^b)&mask).sum()/max(1,((a|b)&mask).sum()))
def occ(a,m):return float(a[m].mean()) if m.any() else None
def same_mesh(a,b):
    with np.load(a/'NATIVE_MESH.npz') as x,np.load(b/'NATIVE_MESH.npz') as y:
        return bool(np.array_equal(x['vertices'],y['vertices']) and np.array_equal(x['faces'],y['faces']))

def main():
    rows=[];criteria=read(OUT/'PLAN.json')['fixed_diagnostic_decision']
    for seed in SEEDS:
        for i in range(2):
            dirs=[V9/f'seed{seed}/correct/object{i}',V9/f'seed{seed}/edit/object{i}',OUT/f'seed{seed}/edit/object{i}']
            vals=[load(d) for d in dirs];a,b,c=vals;removed=a[2]&~b[2]
            protect=(a[2]&b[2])&~cv2.dilate(removed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            iou=lambda v:float((v[1]&v[2]).sum()/max(1,(v[1]|v[2]).sum()))
            row={'seed':seed,'object_index':i,'edited_object':i==0,
              'SS_change_retained_old':change(a[0],b[0],protect),'SS_change_retained_new':change(a[0],c[0],protect),
              'height_original':a[3],'height_old_edit':b[3],'height_new_edit':c[3],
              'height_abs_change_old':abs(b[3]-a[3]),'height_abs_change_new':abs(c[3]-a[3]),
              'map_IoU_old':iou(b),'map_IoU_new':iou(c),
              'removed_projection_occupancy_old':occ(b[1],removed),'removed_projection_occupancy_new':occ(c[1],removed),
              'retained_coverage_old':occ(b[1],a[2]&b[2]),'retained_coverage_new':occ(c[1],a[2]&b[2]),
              'unmodified_object_SS_exact':bool(np.array_equal(a[0],c[0])) if i==1 else None,
              'mesh_vertices_faces_exact_vs_old_edit':same_mesh(dirs[1],dirs[2])}
            if i==0:
                row['paired_numeric_gate']=bool(row['map_IoU_new']>=row['map_IoU_old']-.02 and row['removed_projection_occupancy_new']<=.02
                  and row['retained_coverage_new']>=row['retained_coverage_old']-.02
                  and row['height_abs_change_new']<row['height_abs_change_old']
                  and row['SS_change_retained_new']<row['SS_change_retained_old'])
            rows.append(row)
            sheet=Image.new('RGB',(1536,930),'#111924')
            names=['front','lateral_left','lateral_right','top','mesh_map_overlay']
            for ri,(d,label) in enumerate(zip(dirs,['Original','Old edit (v9)','Descriptor-anchor edit (v10)'])):
                for ci,name in enumerate(names):sheet.paste(Image.open(d/'views'/f'{name}.png').convert('RGB').resize((300,280)),(ci*306,ri*310+27))
                ImageDraw.Draw(sheet).text((5,ri*310+5),f'{label} | object{i} | train seed {seed} | FRONT / SIDE-L / SIDE-R / TOP / MAP',fill='white')
            sheet.save(OUT/f'seed{seed}/object{i}_COMPARISON.jpg',quality=94)
    gate=all(r.get('paired_numeric_gate',True) for r in rows)
    costs=[read(p) for p in (OUT/'gpu_cost').glob('*.json')]
    save(OUT/'RESULTS.json',{'status':'NUMERIC_PASS_VISUAL_PENDING' if gate else 'STOP_SINGLE_INTERVENTION_NOT_JOINTLY_EFFECTIVE',
      'rows':rows,'GPU_seconds':sum(r['gpu_seconds'] for r in costs),'new_training':False,
      'no_more_candidates_or_jobs':True,'historical_gates_unchanged':True,
      'scope':'Two TRAIN objects, edit-context mechanism; not general map-to-world capability'})
    for name,records in [('PAIR_METRICS.csv',rows),('GPU_COST.csv',costs)]:
        with (OUT/name).open('x',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for r in records for k in r)));w.writeheader();w.writerows(records)
    print('IDENTITY_EDIT_DECISION',gate,rows,flush=True)

if __name__=='__main__':main()

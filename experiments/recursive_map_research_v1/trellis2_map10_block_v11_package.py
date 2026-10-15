"""Fixed comparison panels and 48-frame lateral comparison, not a new viewer."""
from paths import read,save,sha
from trellis2_map10_block_v11_prepare import OUT,SEEDS
from PIL import Image,ImageDraw
import numpy as np,imageio.v2 as imageio,csv

def main():
    assert (OUT/'RENDER_COMPLETE.json').exists() and (OUT/'RESULTS.json').exists()
    cols=[('input_map_overlay.png','INPUT + mesh outline'),('top_map_overlay.png','TOP projection vs input'),('views/oblique.png','OBLIQUE'),('views/street_front.png','STREET FRONT'),('views/street_side.png','STREET SIDE')]
    for seed in SEEDS:
        sheet=Image.new('RGB',(1920,860),'#101922');draw=ImageDraw.Draw(sheet)
        for row,mode in enumerate(['correct','edit']):
            d=OUT/f'seed{seed}'/mode
            draw.text((7,row*430+6),f'{seed} | {mode} | gray/blue = generated geometry; flat ground/road = MAP REFERENCE ONLY; NO GENERATED MATERIALS',fill='white')
            for col,(file,label) in enumerate(cols):
                sheet.paste(Image.open(d/file).convert('RGB').resize((380,380)),(col*384,row*430+43));draw.text((col*384+5,row*430+27),label,fill='white')
        sheet.save(OUT/f'seed{seed}/LAYOUT_COMPARISON.jpg',quality=95)
    with imageio.get_writer(str(OUT/'LATERAL_COMPARISON.mp4'),fps=24,codec='libx264',quality=8) as writer:
        for t in range(48):
            sheet=Image.new('RGB',(1024,1088),'#101922');draw=ImageDraw.Draw(sheet)
            for r,seed in enumerate(SEEDS):
                for c,mode in enumerate(['correct','edit']):
                    sheet.paste(Image.open(OUT/f'seed{seed}'/mode/'views/lateral'/f'{t:03d}.png').convert('RGB'),(512*c,544*r+32))
                    draw.text((512*c+7,544*r+4),f'{seed} {mode} | map-reference ground; generated buildings',fill='white')
            writer.append_data(np.asarray(sheet))
    costs=[read(x) for x in (OUT/'gpu_cost').glob('*.json')]
    with (OUT/'GPU_COST.csv').open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for r in costs for k in r)));w.writeheader();w.writerows(costs)
    files=['PLAN.json','RESULTS.json','PER_OBJECT.csv','LAYOUT.csv','EDIT.csv','LATERAL_COMPARISON.mp4','GPU_COST.csv',
      *[f'seed{s}/LAYOUT_COMPARISON.jpg' for s in SEEDS],*[f'seed{s}/{m}/layout.glb' for s in SEEDS for m in ['correct','edit']]]
    save(OUT/'DELIVERY_MANIFEST.json',{'files':[{'path':f,'bytes':(OUT/f).stat().st_size,'sha256':sha(OUT/f)} for f in files],
      'GPU_seconds':sum(r['gpu_seconds'] for r in costs),'stop':True,'no_more_jobs':True,'no_training':True,'all_fixed_results_retained':True})
    save(OUT/'COMPLETE.json',{'fixed_generation_and_visualization_complete':True,'AI_visual_review':'pending','no_follow_on_work':True,'GPU_seconds':sum(r['gpu_seconds'] for r in costs)})
    print('ALL_FIXED_OUTPUTS_PACKAGED_STOP',flush=True)

if __name__=='__main__':main()

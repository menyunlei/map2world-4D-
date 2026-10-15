"""CPU-only packaging and frozen-coordinate scoring; never edits generated geometry."""
from paths import save,read,sha
from trellis2_map_anchor_v20 import OUT
from PIL import Image,ImageDraw
import numpy as np,tarfile,json

def main():
    sheets={}
    for mode in ['correct','edit']:
        d=OUT/mode
        if not (d/'RELOAD.json').exists():continue
        with np.load(d/'MAP.npz') as z:fp=z['footprints'].any(0)
        mask=Image.fromarray((fp[::-1]*255).astype(np.uint8)).resize((512,512),Image.Resampling.NEAREST)
        import scipy.ndimage as ndi
        target=np.asarray(mask)>0;edge=target&~ndi.binary_erosion(target)
        im=np.asarray(Image.open(d/'ORTHO_UNLABELED.png').convert('RGB')).copy();im[edge]=[255,70,60]
        Image.fromarray(im).save(d/'MAP_OVERLAY.png')
        sheets[mode]=['MAP_OVERLAY.png','views/oblique.png','views/front.png','views/lateral_00.png','views/lateral_47.png']
    panel=Image.new('RGB',(5*300,2*326),'#14171f');draw=ImageDraw.Draw(panel)
    for row,(mode,files) in enumerate(sheets.items()):
        for col,f in enumerate(files):
            panel.paste(Image.open(OUT/mode/f).resize((300,300)),(col*300,row*326+26))
            draw.text((col*300+5,row*326+7),mode+' / '+f.replace('views/',''),fill='white')
    panel.save(OUT/'COMPARISON.jpg',quality=95)
    receipt={'experiment':'v20','new_training':False,'materials':False,
        'generation':{m:read(OUT/m/'STRUCTURE.json') for m in ['correct','edit'] if (OUT/m/'STRUCTURE.json').exists()},
        'asset_reload':{m:read(OUT/m/'RELOAD.json') for m in sheets},
        'gpu_seconds':sum(read(p)['gpu_seconds'] for p in (OUT/'gpu_cost').glob('*.json')),
        'building_metric_status':'await target-blind generated-roof semantic labels; all-geometry metrics are not building metrics',
        'human_review':'PENDING; AI visual assessments are not human ratings'}
    save(OUT/'EVIDENCE_SUMMARY.json',receipt)
    with tarfile.open(OUT/'evidence.tar.gz','w:gz') as pack:
        for f in sorted(OUT.rglob('*')):
            if not f.is_file() or any(x in f.parts for x in ['work','cache','extensions','tmp']):continue
            if f.name in ['evidence.tar.gz','RAW_MESH.npz','SHAPE_SLAT.pt','geometry.glb']:continue
            if f.suffix in ['.json','.png','.jpg','.mp4','.log','.npz']:pack.add(f,arcname=str(f.relative_to(OUT)))
    print(json.dumps({'evidence_sha256':sha(OUT/'evidence.tar.gz'),'bytes':(OUT/'evidence.tar.gz').stat().st_size,'gpu_seconds':receipt['gpu_seconds']}),flush=True)

if __name__=='__main__':main()

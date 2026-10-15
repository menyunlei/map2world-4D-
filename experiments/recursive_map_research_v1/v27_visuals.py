"""Fixed-frame map overlays on actual saved-mesh top renders. No asset edits."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def main():
    m=json.loads((OUT/'map10_scaffold.json').read_text(encoding='utf8'))
    tiles=[]
    ref=Image.open(OUT/'map10_reference.png').convert('RGB').resize((512,512))
    tiles.append(('Map10 input: oblique screenshot, not surveyed footprints',ref))
    for i in range(2):
        p=OUT/'attempts'/f'attempt-{i}'/'TOP_NATIVE.png'
        if not p.exists():continue
        im=Image.open(p).convert('RGB');draw=ImageDraw.Draw(im)
        def points(poly):return [((x/100+.5)*im.width,(z/100+.5)*im.height) for x,z in poly]
        for b in m['buildings']:
            pts=points(b['polygon']);draw.line(pts+[pts[0]],fill='#ffb769',width=4);draw.text(pts[0],b['id'],fill='white')
        for road in m['roads']:
            pts=points(road['polygon']);draw.line(pts+[pts[0]],fill='#3df3cd',width=3)
        im.save(p.with_name('TOP_MAP_OVERLAY.png'))
        tiles.append((f'Attempt {i}: saved mesh; orange roof proxies / cyan roads; NO output fit',im.resize((512,512))))
    sheet=Image.new('RGB',(512*len(tiles),560),'#17252f');d=ImageDraw.Draw(sheet)
    for i,(text,im) in enumerate(tiles):sheet.paste(im,(i*512,48));d.text((i*512+8,8),text,fill='white')
    sheet.save(OUT/'map_candidate_comparison.png')
    print('ACTUAL_MESH_MAP_OVERLAY',len(tiles)-1,'candidates')
if __name__=='__main__':main()

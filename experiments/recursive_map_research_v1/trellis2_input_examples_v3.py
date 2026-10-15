"""Input-only contact sheet; never selects a generated output."""
from trellis2_prepare_v1 import REPO, JOB
from PIL import Image, ImageDraw
import math

out = JOB / 'map_object_world_v1/trellis2_resume_v3'
files = sorted((REPO/'assets/example_image').glob('*'))
files = [p for p in files if p.suffix.lower() in ('.png','.webp','.jpg')]
width, height, columns = 180, 200, 8
contact = Image.new('RGB', (width*columns, height*math.ceil(len(files)/columns)), '#e8e8e8')
draw = ImageDraw.Draw(contact)
for i, path in enumerate(files):
    im = Image.open(path).convert('RGBA')
    im.thumbnail((width-8,height-30))
    x,y=(i%columns)*width,(i//columns)*height
    contact.paste(im, (x+(width-im.width)//2,y), im)
    draw.text((x+4,y+height-27),str(i)+' '+path.stem[:13],fill='black')
contact.save(out/'official_inputs_contact.jpg',quality=90)
print('INPUT_CONTACT',len(files),flush=True)

"""CPU contact sheets of unchanged independently reloaded native assets."""
from trellis2_stage_v16 import OUT,SEEDS
from paths import read,save
from PIL import Image,ImageDraw

for seed in SEEDS:
    canvas=Image.new('RGB',(1536,3*280),'#121923');draw=ImageDraw.Draw(canvas)
    for row,stage in enumerate(['predicted','oracleQ','codec']):
        for i in range(2):
            src=OUT/'diagnosis'/stage/f'seed{953101 if stage=="codec" else seed}'/f'object{i}'
            for col,name in enumerate(['front','lateral','top']):
                canvas.paste(Image.open(src/f'{name}.png').resize((256,256)),((i*3+col)*256,row*280+24))
        draw.text((5,row*280+3),f'{stage}: TRAIN objects 0 and 1 | front / lateral / top | seed {seed}',fill='white')
    canvas.save(OUT/f'TRAIN_MATCHED_{seed}.jpg',quality=94)
    canvas=Image.new('RGB',(1536,4*280),'#121923');draw=ImageDraw.Draw(canvas)
    for row in range(4):
        src=OUT/'diagnosis/Map10'/f'seed{seed}'/f'object{row}'
        draw.text((5,row*280+3),f'Map10 object {row} | native support XY/XZ/YZ then mesh front/lateral/top | no GT',fill='white')
        canvas.paste(Image.open(src/'SUPPORT.png').resize((768,256)),(0,row*280+24))
        for col,name in enumerate(['front','lateral','top']):
            if (src/f'{name}.png').exists():canvas.paste(Image.open(src/f'{name}.png').resize((256,256)),((col+3)*256,row*280+24))
        if not (src/'front.png').exists():draw.text((800,row*280+80),'EMPTY STRUCTURE: no mesh or substitute',fill='red')
    canvas.save(OUT/f'MAP10_STAGE_{seed}.jpg',quality=94)
print('CONTACTS_READY',flush=True)

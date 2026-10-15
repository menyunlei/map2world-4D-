"""Reuse fixed fresh-process views and native chunked diagnostic projection."""
from pathlib import Path
from trellis2_map_volume_v19 import OUT
import sys
source=Path(__file__).with_name('trellis2_mapguided_v18_views.py')
code=source.read_text().split('# Orthographic projection')[0]
code=code.replace('from trellis2_mapguided_v18 import OUT','from trellis2_map_volume_v19 import OUT')
scope={'__name__':'v19_saved_views','__file__':str(source)}
exec(compile(code,str(source),'exec'),scope)
import gc,torch
del scope;gc.collect();torch.cuda.empty_cache()
source=Path(__file__).with_name('trellis2_mapguided_v18_overlay_fix1.py')
code=source.read_text().split('def owner():')[0]
code=code.replace('from trellis2_mapguided_v18 import OUT,SEED','from trellis2_map_volume_v19 import OUT,SEED')
scope={'__name__':'v19_saved_overlay','__file__':str(source)}
exec(compile(code,str(source),'exec'),scope)
scope['worker'](sys.argv[sys.argv.index('--mode')+1])

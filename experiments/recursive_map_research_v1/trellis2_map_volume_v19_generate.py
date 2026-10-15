"""Identical native Shape/texture/export path; consume the new fixed Structure."""
from pathlib import Path
source=Path(__file__).with_name('trellis2_mapguided_v18_generate.py')
code=source.read_text().replace('from trellis2_mapguided_v18 import OUT,SEED','from trellis2_map_volume_v19 import OUT,SEED')
exec(compile(code,str(source),'exec'))

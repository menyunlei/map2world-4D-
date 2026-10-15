"""Use native MeshWithVoxel renderer's named base_color output; same saved asset."""
from pathlib import Path
source=Path(__file__).with_name('trellis2_native_street_v17_views.py')
code=source.read_text()
code=code.replace("attr=p['attr'].permute(1,2,0).cpu().numpy()", "attr=p['base_color'].permute(1,2,0).cpu().numpy()")
code=code.replace("rgb=attr[...,data['layout']['base_color']]", "rgb=attr")
code=code.replace("vd=OUT/'views'", "vd=OUT/'views_keys_fix1'")
exec(compile(code,str(source),'exec'))

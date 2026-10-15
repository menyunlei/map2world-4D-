"""Use actual named material outputs from official MeshRenderer."""
from pathlib import Path
code=Path(__file__).with_name('trellis2_raw_views_chunk_fix2.py').read_text()
code=code.replace("gbuffer['attr'][:3]","gbuffer['base_color']")
code=code.replace("gbuffer['attr'][5]","gbuffer['alpha']")
code=code.replace("'native_views_chunked'","'native_views_final'")
code=code.replace('NATIVE_RELOAD_COMPLETE_CHUNKED.json','NATIVE_RELOAD_COMPLETE_FINAL.json')
code=code.replace("source=source.replace('NATIVE_RELOAD_COMPLETE.json'", "source=source.replace('NATIVE_RELOAD_COMPLETE.json'")
code=code.replace("exec(compile(source,'native_asset_chunked_readonly_render','exec'))",
    "source=source.replace('if tensor.ndim==2:tensor=tensor[None].expand(3, 1, 1)', 'if tensor.ndim==2:tensor=tensor[None].expand(3, 1, 1)')\n"
    "source=source.replace('if tensor.ndim==2:tensor=tensor[None].expand(3,-1,-1)',"
    " 'if tensor.ndim==2:tensor=tensor[None].expand(3,-1,-1)\\n            if tensor.ndim==3 and tensor.shape[0]==1:tensor=tensor.expand(3,-1,-1)')\n"
    "exec(compile(source,'native_asset_correct_material_output_keys','exec'))")
exec(compile(code,'native_mesh_material_keys_fix','exec'))

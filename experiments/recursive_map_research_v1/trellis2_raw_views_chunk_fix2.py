"""Official chunked mesh rendering, native base color; all original triangles retained."""
from pathlib import Path
source=Path(__file__).with_name('trellis2_raw_asset_views_v3.py').read_text()
source=source.replace('from trellis2.renderers import EnvMap','from trellis2.renderers import MeshRenderer')
source=source.replace("image=cv2.cvtColor(cv2.imread(str(REPO/'assets/hdri/studio.exr'),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)", '')
source=source.replace("envmap=EnvMap(torch.tensor(image,dtype=torch.float32,device='cuda'))",'')
source=source.replace("target=root/'native_views'","target=root/'native_views_chunked'")
source=source.replace("renderer=render_utils.get_renderer(mesh,resolution=768,near=.005,far=10,ssaa=1)",
    "renderer=MeshRenderer(); renderer.rendering_options.resolution=768; renderer.rendering_options.near=.005; "
    "renderer.rendering_options.far=10; renderer.rendering_options.ssaa=1; renderer.rendering_options.chunk_size=8000000; "
    "renderer.rendering_options.antialias=False")
source=source.replace("with torch.no_grad():result=renderer.render(mesh,extr,intr,envmap=envmap)",
    "with torch.no_grad():\n"
    "            gbuffer=renderer.render(mesh,extr,intr,return_types=['attr','mask','normal'])\n"
    "            result={'shaded':gbuffer['attr'][:3]+0.18*(1-gbuffer['mask'])[None], "
    "'base_color':gbuffer['attr'][:3], 'normal':gbuffer['normal'], 'alpha':gbuffer['attr'][5]}")
source=source.replace("'shader':'official PbrMeshRenderer'","'shader':'official MeshRenderer chunk_size8M, opaque base-color inspection; no PBR lighting or transparent depth peeling'")
source=source.replace("'HDRI':str(REPO/'assets/hdri/studio.exr')","'lighting':'unlit native base color'")
source=source.replace('NATIVE_RELOAD_COMPLETE.json','NATIVE_RELOAD_COMPLETE_CHUNKED.json')
exec(compile(source,'native_asset_chunked_readonly_render','exec'))

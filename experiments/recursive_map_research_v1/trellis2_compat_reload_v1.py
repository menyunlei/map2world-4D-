"""Fresh Blender load of compatible base-color GLB; two fixed format-check views."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import trellis2_saved_mesh_render_v2 as render
render.FRAME_COUNT=2
render.VIEW_NAMES=('front','oblique','lateral-left','lateral-right')
render.MODES=('textured',)
render.encode_videos=lambda args: []
render.contact_sheet=lambda bpy,output: None
if __name__=='__main__':
    raise SystemExit(render.main())

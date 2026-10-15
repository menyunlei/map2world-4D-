"""Saved-asset stills with explicit module path for Blender's Python launcher."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import trellis2_saved_mesh_render_v2 as render
render.FRAME_COUNT=2
render.encode_videos=lambda args: []
if __name__=='__main__':
    raise SystemExit(render.main())

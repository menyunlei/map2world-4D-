"""Quick first delivery: same saved-asset renderer, stills before any video."""
import trellis2_saved_mesh_render_v2 as render

render.FRAME_COUNT = 2
render.encode_videos = lambda args: []
if __name__ == '__main__':
    raise SystemExit(render.main())

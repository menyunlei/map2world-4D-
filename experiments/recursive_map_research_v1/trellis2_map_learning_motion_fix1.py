"""Same fixed lateral renders; use the installed FFmpeg binary explicitly."""
from trellis2_map_learning_views_v1 import *
import subprocess
import imageio_ffmpeg


def run():
    planned = {n: (e, l, u, f) for n, e, l, u, f in cameras()}
    left = np.asarray(planned['street_lateral_left'][0])
    right = np.asarray(planned['street_lateral_right'][0])
    look = torch.tensor(planned['street_lateral_left'][1], dtype=torch.float32, device='cuda')
    up = torch.tensor([0., 0., 1.], device='cuda')
    fov = torch.tensor(math.radians(60), device='cuda')
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for seed in [953101, 953102]:
        for mode in ['correct', 'edit']:
            root = OUT / f'seed{seed}/assets' / mode
            d = torch.load(root / 'RAW_NATIVE.pt', map_location='cpu', weights_only=False)
            spatial = (d['coords'].max(0).values + 1).tolist()
            mesh = MeshWithVoxel(d['vertices'].cuda(), d['faces'].cuda(), [-.5, -.5, -.5], d['voxel_size'],
                d['coords'].cuda(), d['attrs'].cuda(), torch.Size([1, d['attrs'].shape[1], *spatial]), d['layout'])
            renderer = MeshRenderer()
            renderer.rendering_options.resolution = 512
            renderer.rendering_options.near = .001
            renderer.rendering_options.far = 10
            renderer.rendering_options.ssaa = 1
            renderer.rendering_options.chunk_size = 8000000
            renderer.rendering_options.antialias = False
            frames = root / 'lateral_frames'
            frames.mkdir(exist_ok=False)
            for i, t in enumerate(np.linspace(0, 1, 48)):
                eye = torch.tensor((1-t)*left+t*right, dtype=torch.float32, device='cuda')
                ext = utils3d.torch.extrinsics_look_at(eye, look, up)
                with torch.no_grad():
                    g = renderer.render(mesh, ext, intr, return_types=['attr', 'mask'])
                write_img(g['base_color'] + .18*(1-g['mask'])[None], frames / f'{i:04d}.png')
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-framerate', '24',
                '-i', str(frames / '%04d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(root / 'lateral_24fps.mp4')], check=True)
            save(root / 'MOTION.json', {'frames': 48, 'fps': 24, 'seconds': 2,
                'actual_camera_translation': True, 'start_eye': left.tolist(), 'end_eye': right.tolist(),
                'source': 'same saved native asset, no model resampling or frame duplication',
                'negative_empty_street_views_preserved': True, 'no_world_success_claim': True})
            print('NATIVE_MOTION_SAVED', seed, mode, flush=True)
            del mesh, d
            torch.cuda.empty_cache()


if __name__ == '__main__':
    run()

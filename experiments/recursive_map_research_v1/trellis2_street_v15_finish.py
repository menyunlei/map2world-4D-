"""CPU-only preservation of completed renders; no model or renderer invocation."""
from pathlib import Path
import json, hashlib, shutil, subprocess, tarfile
from PIL import Image, ImageDraw
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'map_object_world_v1/trellis2_map10_street_v15'
RELAY = HERE / 'map_object_world_v1/trellis2_coverage_v12/diagnosis_v1/street_v15_delivery'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + '\n')


plan = json.loads((OUT / 'PLAN.json').read_text())
outcomes = []
for seed in [953101, 953102]:
    d = OUT / f'seed{seed}/correct'
    vd = d / 'street_views'
    layout = json.loads((d / 'LAYOUT.json').read_text())
    frames = sorted((vd / 'frames').glob('*.png'))
    assert [f.stem for f in frames] == [f'{i:04d}' for i in range(len(frames))]
    assert 0 < len(frames) <= len(plan['trajectory'])
    painted = (d / 'PAINT_COMPLETE.json').exists()
    if not (d / 'STREET_COMPLETE.json').exists():
        mini = Image.open(OUT / 'inputs/source_crop.png').convert('RGB').resize((156, 156))
        route = [tuple((np.asarray(v['source_crop_xy']) * 156 / 910).tolist())
                 for v in plan['trajectory']]
        dest = vd / 'frames_minimap'
        dest.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            dst = dest / frame.name
            if dst.exists():
                continue
            image = Image.open(frame).convert('RGB')
            mp = mini.copy()
            draw = ImageDraw.Draw(mp)
            draw.line(route, fill=(255, 230, 40), width=2)
            x, y = route[i]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 35, 25))
            image.paste(mp, (476, 8))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 340, 640, 360), fill='#111923')
            draw.text((6, 345), 'FAILED preview | AI geometry | map road/ground | approximate XY', fill='white')
            image.save(dst)
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        video = d / ('street_8s_24fps.mp4' if len(frames) == 192 else 'street_partial_24fps.mp4')
        assert not video.exists(), video
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-framerate', '24',
                        '-i', str(dest / '%04d.png'), '-frames:v', str(len(frames)),
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(video)], check=True)
        sheet = Image.new('RGB', (1280, 1080), '#121922')
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(['top', 'oblique', 'street_start', 'street_middle', 'street_end']):
            sheet.paste(Image.open(vd / f'{name}.png'), ((i % 2)*640, (i // 2)*360))
            draw.text(((i % 2)*640+8, (i // 2)*360+8), name, fill='white')
        sheet.save(d / 'street_contact.jpg', quality=94)
        write(d / 'STREET_COMPLETE_CPU.json', {
            'video': str(video), 'sha256': digest(video), 'frames': len(frames), 'fps': 24,
            'duration_seconds': len(frames)/24, 'cpu_packaging_only': True,
            'gpu_job_timed_out_after_completed_blender_render': True,
            'no_model_or_renderer_rerun': True, 'quality': 'FAIL'})
    outcomes.append({'seed': seed, 'expected_objects': 4,
                     'generated_objects': layout['generated_count'],
                     'empty_structure_objects': [r['object_id'] for r in layout['objects']
                                                 if r['status'] != 'MESH_GENERATED'],
                     'appearance_generated': painted, 'rendered_frames': len(frames),
                     'street_quality': 'FAIL',
                     'agent_visual_observation':
                     'Vehicle-like / fragmented geometry and floating sheets; no usable street'
                     if painted else 'Floating platforms and broken volume; large building absent'})

receipts = [json.loads(p.read_text()) for p in sorted((OUT / 'gpu_cost').glob('*.json'))]
cost = sum(r['gpu_seconds'] for r in receipts)
write(OUT / 'RESULTS.json', {
    'verdict': 'STREET_GENERATION_FAIL', 'scope': 'Map10 development street preview, not generalization',
    'seeds': outcomes, 'training_updates': 0, 'gpu_seconds': cost, 'gpu_hours': cost/3600,
    'review': 'Agent inspection, not human acceptance', 'no_more_gpu_jobs': True,
    'map_conditioned_geometry': 'TRELLIS.2; existing learned Structure adapter/LoRA and Shape map adapter',
    'geometry_unchanged_by_paint': True, 'full_native_PBR_retained_on_server': True,
    'failure_first_observed': 'Structure empty output and incorrect support; Shape meshes visibly degenerate before paint',
    'causal_attribution': 'No claim that one unique parameter group is the cause',
    'not_a_wrong_reference_image': 'Fixed official building image, not a vehicle image',
    'fixed_map_placement': True, 'road_ground_source': 'Map-derived flat visualization, not AI generated',
    'coordinates': 'Approximate normalized XY from oblique screenshot; not geographic registration',
    'right_side_buildings': 'Not selected or generated', 'no_geometry_repair_or_substitution': True,
    'next_requirement': 'Reliable real-footprint Structure/Shape generation before further material work',
    'automatic_training_or_seed_search': False})
write(OUT / 'SESSION_STATE.json', {'status': 'COMPLETE_FAILED_STOP', 'results': 'RESULTS.json',
                                  'native_assets_preserved': True, 'historical_results_preserved': True,
                                  'gpu_jobs_queued': [], 'cpu_packaging_complete': True})

selected = []
for p in sorted(OUT.rglob('*')):
    if not p.is_file():
        continue
    rel = p.relative_to(OUT)
    if any(part in ['frames', 'frames_minimap', 'work', 'appearance'] for part in rel.parts):
        continue
    if p.suffix.lower() in ['.json', '.log', '.csv', '.png', '.jpg', '.mp4', '.exr']:
        selected.append(p)
manifest = {str(p.relative_to(OUT)).replace('\\', '/'): digest(p) for p in selected}
write(OUT / 'EVIDENCE_MANIFEST.json', manifest)
archive = RELAY / 'street_v15_evidence.tar.gz'
assert not archive.exists(), archive
with tarfile.open(archive, 'w:gz') as tar:
    for p in selected + [OUT / 'EVIDENCE_MANIFEST.json']:
        tar.add(p, arcname=str(p.relative_to(OUT)), recursive=False)
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'sha256': digest(archive), 'gpu_seconds': cost, 'outcomes': outcomes}), flush=True)

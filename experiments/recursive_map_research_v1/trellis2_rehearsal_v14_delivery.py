"""Package fixed outputs and matched render sheets; no model or GPU operations."""
from pathlib import Path
import hashlib, json, tarfile, argparse
from PIL import Image, ImageDraw

LOCAL_JOB = Path(__file__).resolve().parent
V12 = LOCAL_JOB/'map_object_world_v1/trellis2_coverage_v12'
OUT = LOCAL_JOB/'map_object_world_v1/trellis2_rehearsal_v14'
SEEDS = [953101, 953102]

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    d = hashlib.sha256()
    with p.open('rb') as h:
        for chunk in iter(lambda: h.read(4*1024*1024), b''): d.update(chunk)
    return d.hexdigest()

def pack():
    complete = read(OUT/'COMPLETE.json')
    assert complete['errors'] == [] and complete['no_more_jobs']
    plan = read(OUT/'PLAN.json')
    for seed in SEEDS:
        for index in plan['mesh_indices']:
            old = V12/f'evaluation/seed{seed}/object{index:03d}_MATCHED_MESH.jpg'
            new = OUT/f'evaluation/seed{seed}/object{index:03d}_MATCHED_MESH.jpg'
            with Image.open(old) as a, Image.open(new) as b:
                assert a.size == (1536, 1680) and b.size == (1536, 560)
                sheet = Image.new('RGB', (1536, 1120), '#121922')
                sheet.paste(a.crop((0, 1120, 1536, 1680)), (0, 0))
                sheet.paste(b, (0, 560))
                draw = ImageDraw.Draw(sheet)
                for row, label in enumerate(['UNIFORM24 +1000 ORIGINAL', 'UNIFORM24 +1000 FOOTPRINT EDIT',
                                             'OLD/NEW REPLAY +1000 ORIGINAL', 'OLD/NEW REPLAY +1000 FOOTPRINT EDIT']):
                    draw.rectangle((0, row*280, 1536, row*280+23), fill='#121922')
                    draw.text((5, row*280+5), f'{seed} object{index:03d}: {label}', fill='white')
                p = OUT/f'comparison/seed{seed}/object{index:03d}.jpg'
                assert not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(p, quality=94)
    manifests = {}; groups = {'evidence': [], 'assets': []}
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'work' in p.relative_to(OUT).parts: continue
        rel = p.relative_to(OUT)
        if p.name == 'geometry.glb': groups['assets'].append(p)
        elif p.suffix in ['.json', '.jsonl', '.jpg', '.png', '.log']:
            groups['evidence'].append(p)
        elif p.suffix == '.npz' and p.name == 'SS_GENERATED.npz':
            groups['evidence'].append(p)
        elif p.name == 'checkpoint_001000.pt': groups['evidence'].append(p)
    for name, files in groups.items():
        manifests[name] = [{'path': str(p.relative_to(OUT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files]
    mp = OUT/'DELIVERY_MANIFEST.json'
    assert not mp.exists(); mp.write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    groups['evidence'].append(mp)
    relay = V12/'diagnosis_v1/replay_v14_delivery'; relay.mkdir(parents=True, exist_ok=False)
    receipts = []
    for name, files in groups.items():
        p = relay/f'v14_{name}.tar.gz'
        with tarfile.open(p, 'x:gz', compresslevel=1) as archive:
            for f in files: archive.add(f, arcname=str(f.relative_to(OUT)), recursive=False)
        receipts.append({'name': name, 'bundle': str(p), 'bytes': p.stat().st_size,
                         'sha256': sha(p), 'files': len(files)})
    print(json.dumps({'bundles': receipts, 'actual_saved_meshes': len(groups['assets']),
                      'comparisons': len(plan['mesh_indices'])*2}), flush=True)

def unpack(bundle):
    assert bundle.is_file()
    with tarfile.open(bundle, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            assert member.isfile(), member.name
            target = (OUT/member.name).resolve()
            assert target.is_relative_to(OUT.resolve()), member.name
            if target.exists():
                src = archive.extractfile(member)
                d = hashlib.sha256()
                for chunk in iter(lambda: src.read(4*1024*1024), b''): d.update(chunk)
                assert sha(target) == d.hexdigest(), ('LOCAL_CONFLICT', str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archive.extract(member, path=OUT, filter='data')
    manifest = read(OUT/'DELIVERY_MANIFEST.json')
    checked = []
    for group in manifest.values():
        for record in group:
            path = OUT/record['path']
            if path.exists():
                assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
                checked.append(record['path'])
    print(json.dumps({'extracted_members': len(members), 'verified_present_files': len(checked),
                      'output': str(OUT)}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--unpack', type=Path); a = p.parse_args()
    if a.unpack: unpack(a.unpack)
    else: pack()

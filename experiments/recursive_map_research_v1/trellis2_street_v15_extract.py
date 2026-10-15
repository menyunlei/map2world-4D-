"""Verified local extraction of the CPU evidence pack; preserves existing assets."""
from pathlib import Path, PurePosixPath
import hashlib, json, tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'map_object_world_v1/trellis2_coverage_v12/diagnosis_v1/street_v15_delivery/street_v15_evidence.tar.gz'
DEST = ROOT / 'map_object_world_v1/trellis2_map10_street_v15'

with tarfile.open(ARCHIVE, 'r:gz') as tar:
    manifest = json.load(tar.extractfile('EVIDENCE_MANIFEST.json'))
    for m in tar.getmembers():
        name = PurePosixPath(m.name)
        assert m.isfile() and not name.is_absolute() and '..' not in name.parts, m.name
        data = tar.extractfile(m).read()
        if m.name != 'EVIDENCE_MANIFEST.json':
            assert hashlib.sha256(data).hexdigest() == manifest[m.name], m.name
        path = DEST.joinpath(*name.parts)
        assert path.resolve().is_relative_to(DEST.resolve())
        if path.exists():
            assert path.read_bytes() == data, ('EXISTING_DIFFERENT', path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
for name, expected in manifest.items():
    assert hashlib.sha256((DEST / name).read_bytes()).hexdigest() == expected, name
print(json.dumps({'verified_files': len(manifest), 'destination': str(DEST),
                  'results': json.loads((DEST / 'RESULTS.json').read_text())}, indent=2))

"""Verify and unpack this experiment's evidence without replacing any file."""
from pathlib import Path,PurePosixPath
import tarfile,json,hashlib
root=Path(__file__).resolve().parent/'map_object_world_v1/trellis2_stage_attribution_v16'
archive=root/'evidence_and_checkpoints.tar.gz'
with tarfile.open(archive,'r:gz') as tar:
    manifest=json.load(tar.extractfile('EVIDENCE_MANIFEST.json'))
    for member in tar.getmembers():
        name=PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts,member.name
        data=tar.extractfile(member).read()
        if member.name in manifest:assert hashlib.sha256(data).hexdigest()==manifest[member.name],member.name
        path=root.joinpath(*name.parts);assert path.resolve().is_relative_to(root.resolve())
        if path.exists():assert path.read_bytes()==data,('EXISTING_DIFFERENT',path)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
assets=json.loads((root/'ASSET_MANIFEST.json').read_text())
verified=[]
for name,h in assets.items():
    path=root/name
    if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name;verified.append(name)
print(json.dumps({'verified_evidence_files':len(manifest),'verified_downloaded_GLBS':verified,
    'checkpoints_present':len(list((root/'repair').glob('*/checkpoint_*.pt'))),
    'verdict':json.loads((root/'RESULTS.json').read_text())['verdict']},indent=2))

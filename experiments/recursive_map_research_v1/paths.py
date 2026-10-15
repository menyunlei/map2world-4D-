"""Task-scoped paths; no old experiment is writable through this module."""
from pathlib import Path
import hashlib, json

ROOT = Path('/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master')
JOB = ROOT / 'experiments/recursive_map_research_v1'
DATA = ROOT / 'experiments/dreamx_courtyard_data_repair_v2'
C4 = ROOT / 'experiments/c4_supervision_optimization_v1'
O3 = ROOT / 'experiments/o1_paired_stabilization_v1'
HIER = ROOT / 'experiments/courtyard_binding_search_v1'
NATIVE = ROOT / 'experiments/map_native_scene_dit_v1'
D2 = ROOT / 'experiments/dreamx_native_scene_a800_v1'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(JOB.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


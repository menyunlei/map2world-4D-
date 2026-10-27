"""Versioned source overlay; leaves the verified baseline checkout unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import sys

BASELINE_HASHES = {
    'models/gancraft.py': 'adf91dc4ea4d17d464cbbfead3620b2017ffcc89e57210776bcd7498bfaba2cc',
    'scripts/inference.py': 'af84529471e99a58c1ff1519f3c59fdf75ce5093e452bed26cf676e3dc7fad00',
}

def load(repo):
    repo = Path(repo).resolve()
    root = Path(__file__).resolve().parent
    for name, expected in BASELINE_HASHES.items():
        actual = hashlib.sha256((repo/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Baseline changed; review before using overlay: '+name)
    sys.path.insert(0, str(repo))
    import models
    import scripts
    for name, parent in [('models.gancraft', models), ('scripts.inference', scripts)]:
        file = root/'overlay'/Path(*name.split('.')).with_suffix('.py')
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        setattr(parent, name.rsplit('.', 1)[1], mod)
        spec.loader.exec_module(mod)
    return sys.modules['scripts.inference']

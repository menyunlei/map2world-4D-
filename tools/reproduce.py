"""Source audit and offline runtime-path configuration. Never launches a GPU job."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'experiments/recursive_map_research_v1'
LOCK = ROOT / 'docs/reproduction-lock.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root=ROOT):
    """Syntax/local imports of tracked files; does NOT prove external data availability."""
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '*.py'], text=True).splitlines()
    tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines())
    missing, errors = [], []
    for name in names:
        p = root / name
        try:
            tree = ast.parse(p.read_text(encoding='utf-8-sig'), filename=name)
        except SyntaxError as e:
            errors.append({'source': name, 'line': e.lineno})
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split('.')[0]]
            for module in modules:
                dep = p.parent / (module + '.py')
                if dep.is_file() and dep.relative_to(root).as_posix() not in tracked:
                    missing.append({'source': name, 'missing': dep.relative_to(root).as_posix()})
    return {'python_files': len(names), 'syntax_errors': errors, 'missing_local_imports': missing,
            'scope': 'Tracked syntax and available sibling modules only; not dynamic execution, weights or datasets'}


def configure(template, weights, dino, output):
    """Preserve native normalization arrays, rewrite only local paths/runtime settings."""
    document = json.loads(template.read_text())
    args = document['args']
    expected = {
        'sparse_structure_decoder': ('TRELLIS-image-large', 'ss_dec_conv3d_16l8_fp16'),
        'sparse_structure_flow_model': ('TRELLIS.2-4B', 'ss_flow_img_dit_1_3B_64_bf16'),
        'shape_slat_decoder': ('TRELLIS.2-4B', 'shape_dec_next_dc_f16c32_fp16'),
        'shape_slat_flow_model_512': ('TRELLIS.2-4B', 'slat_flow_img2shape_dit_1_3B_512_bf16'),
        'shape_slat_flow_model_1024': ('TRELLIS.2-4B', 'slat_flow_img2shape_dit_1_3B_1024_bf16'),
        'tex_slat_decoder': ('TRELLIS.2-4B', 'tex_dec_next_dc_f16c32_fp16'),
        'tex_slat_flow_model_512': ('TRELLIS.2-4B', 'slat_flow_imgshape2tex_dit_1_3B_512_bf16'),
        'tex_slat_flow_model_1024': ('TRELLIS.2-4B', 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'),
    }
    models, missing = {}, []
    for name, (folder, stem) in expected.items():
        prefix = weights.resolve() / folder / 'ckpts' / stem
        models[name] = str(prefix)
        for suffix in ['.json', '.safetensors']:
            if not Path(str(prefix) + suffix).is_file():
                missing.append(str(prefix) + suffix)
    for name in ['config.json', 'model.safetensors']:
        if not (dino / name).is_file():
            missing.append(str(dino / name))
    if missing:
        raise ValueError('Missing required local files: ' + json.dumps(missing))
    if digest(dino / 'model.safetensors') != json.loads(LOCK.read_text())['dinov3_model_sha256']:
        raise ValueError('DINO weight hash differs from the verified original; not exact reproduction')
    # Native normalization is part of the frozen pretrained interface.
    for name in ['shape_slat_normalization', 'tex_slat_normalization']:
        if len(args[name]['mean']) != 32 or len(args[name]['std']) != 32:
            raise ValueError('Unexpected native normalization: ' + name)
    args['models'] = models
    args['image_cond_model'] = {'name': 'DinoV3FeatureExtractor', 'args': {'model_name': str(dino.resolve())}}
    args['rembg_model'] = {'name': 'UnusedBackgroundRemoval', 'args': {}}
    args['low_vram'] = True
    args['default_pipeline_type'] = '1024_cascade'
    args.update(json.loads(LOCK.read_text())['samplers'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing runtime or original historical configuration.
    with output.open('x') as f:
        json.dump(document, f, indent=2)
    return {'pipeline': str(output.resolve()), 'sha256': digest(output), 'gpu_work': False,
            'note': 'UnusedBackgroundRemoval requires the existing native entry shim; not a standalone official pipeline file'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    cfg = sub.add_parser('configure')
    for name in ['template', 'weights', 'dino', 'output']:
        cfg.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'audit':
        result = audit()
        print(json.dumps(result, indent=2))
        return int(bool(result['syntax_errors'] or result['missing_local_imports']))
    print(json.dumps(configure(a.template, a.weights, a.dino, a.output), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

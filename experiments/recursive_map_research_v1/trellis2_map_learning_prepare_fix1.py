"""Pre-optimizer correction: select 3--5-object input families, not all old blocks.

Original failed registration and its partial files remain untouched.
"""
import inspect
from pathlib import Path
import trellis2_map_learning_prepare_v1 as base
from paths import JOB, read, save, sha
import numpy as np

OUT = base.OUT / 'trial_fix1'


def eligible_families(catalogue):
    result = []
    seen = set()
    for sid in sorted(catalogue['original_order']):
        r = catalogue['records'][sid]
        if r['family_id'] in seen:
            continue
        seen.add(r['family_id'])
        meta = read(Path(r['map_path']).parent/'MANIFEST.json')
        if 3 <= len(meta['objects']) <= 5:
            result.append(r['family_id'])
        if len(result) == 40:
            break
    assert len(result) == 40, 'Insufficient existing independent 3--5-building TRAIN families'
    return result


def masks(meta):
    value = base.masks(meta)
    value.pop('confidence')
    value['footprint_known'] = np.ones_like(value['footprints'])
    value['roads_known'] = np.ones_like(value['roads'])
    value['open_space_known'] = np.ones_like(value['open_space'])
    return value


if __name__ == '__main__':
    source = inspect.getsource(base.prepare).replace(
        "families = sorted({catalogue['records'][sid]['family_id'] for sid in catalogue['original_order']})",
        'families = eligible_families(catalogue)')
    scope = {**base.__dict__, 'OUT':OUT, 'eligible_families':eligible_families, 'masks':masks}
    exec(compile(source, __file__, 'exec'), scope)
    scope['prepare']()
    save(OUT/'PREPARATION_FIX.json',{'pre_optimizer':True,'original_failure':'unfiltered TRAIN includes blocks with >5 buildings',
        'single_correction':'take first40 eligible TRAIN families by registered object count before any generated outputs',
        'old_script_sha256':sha(JOB/'trellis2_map_learning_prepare_v1.py'),
        'fix_script_sha256':sha(JOB/'trellis2_map_learning_prepare_fix1.py'),
        'previous_partial_registration_preserved':True,'no_optimizer_or_GPU_work_before_fix':True})

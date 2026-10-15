"""Reuse v11 inference; registered newer SS weights, one original layout only."""
from pathlib import Path
from trellis2_street_v15_prepare import OUT
import argparse

def run(seed):
    path=Path(__file__).with_name('trellis2_map10_block_v11_run.py')
    code=path.read_text()
    assert code.count("for mode in ['correct','edit']:")==2
    code=code.replace("for mode in ['correct','edit']:","for mode in ['correct']:")
    code=code.replace("'expected_objects':8","'expected_objects':4")
    code=code.replace("'source':'v9_map_only'","'source':'v15_existing_map_only'")
    namespace={'__name__':'street_v15_reuse'};exec(compile(code,str(path),'exec'),namespace)
    namespace['OUT']=OUT;namespace['run'](seed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[953101,953102],required=True)
    run(p.parse_args().seed)

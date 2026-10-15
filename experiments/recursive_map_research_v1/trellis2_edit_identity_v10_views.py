"""Reuse the exact v9 independent renderer in a separate process, new output root."""
from trellis2_edit_identity_v10_prepare import OUT,SEEDS
import trellis2_map_edit_mesh_v9_views as previous
import argparse

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=SEEDS,required=True)
    previous.OUT=OUT;previous.run(a.parse_args().seed)

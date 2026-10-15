"""Use v12 generation/export/rendering verbatim for the new replay checkpoint."""
import argparse
from trellis2_rehearsal_v14_train import OUT, ARM, SEEDS, configure
import trellis2_coverage_v12_eval as prior

def run(mode, seed):
    configure()
    prior.OUT = OUT
    prior.GROUPS = [(ARM, 1000), (ARM, 100), (ARM, 300)] if mode == 'structure' else [(ARM, 1000)]
    {'structure': prior.evaluate, 'mesh': prior.mesh, 'views': prior.views}[mode](seed)
    if mode == 'views':
        # Only two populated rows in the reused six-row sheet; retain their exact pixels.
        from PIL import Image
        from paths import read
        for i in read(OUT/'PLAN.json')['mesh_indices']:
            path = OUT/'evaluation'/f'seed{seed}'/f'object{i:03d}_MATCHED_MESH.jpg'
            with Image.open(path) as image:
                image.crop((0, 0, image.width, 560)).save(path, quality=92)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['structure', 'mesh', 'views'], required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = parser.parse_args()
    run(a.mode, a.seed)

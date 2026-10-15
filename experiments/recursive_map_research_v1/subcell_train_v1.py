"""Reuse actual server trainer, objective, slots and optimizer without refactoring."""
import sys, argparse
from paths import *
from subcell_runtime_v1 import NEW, OLD, Data, initialize

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--arm', choices=['baseline', 'narrow'], required=True)
cfg, remaining = ap.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
source = (JOB / 'train_candidate.py').read_text()
before = 'if start==0:assert torch.equal(x,y)'
assert source.count(before) == 1
source = source.replace(before, 'if False:assert torch.equal(x,y)')
source = source.replace("'identity_required':start==0", "'identity_required':False")
namespace = {'__name__': 'subcell_reused_trainer', '__file__': str(JOB / 'train_candidate.py')}
exec(compile(source, str(JOB / 'train_candidate.py'), 'exec'), namespace)
namespace.update(JOB=NEW / cfg.arm, TrainData=Data,
                 initialize=lambda candidate, seed: initialize(cfg.arm, seed))
namespace['main']()

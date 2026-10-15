"""Reuse v12 training unchanged, replacing only its TRAIN record order."""
import random
from types import SimpleNamespace
from paths import JOB
import trellis2_coverage_v12_train as prior

OUT = JOB / 'map_object_world_v1/trellis2_rehearsal_v14'
ARM = 'old_new_replay'
SEEDS = [953101, 953102]

def replay_order(seed=971800):
    order = list(range(24))
    random.Random(seed).shuffle(order)
    old = [i for i in order if i < 6]
    new = [i for i in order if i >= 6]
    return [x for j, k in enumerate(new) for x in (old[j % 6], k)]

class ReplayShuffle:
    def __init__(self, seed):
        self.seed = seed

    def shuffle(self, order):
        assert order == list(range(24)), 'Only the registered 24 TRAIN objects are supported'
        order[:] = replay_order(self.seed)

def configure():
    prior.train.OUT = OUT
    prior.train.ARMS = [ARM]
    prior.train.SEEDS = SEEDS
    prior.train.random = SimpleNamespace(Random=ReplayShuffle)
    # v12's setup and native-flow step remain installed, including its slot noise offset.
    assert prior.train.setup is prior.setup and prior.train.step is prior.step

if __name__ == '__main__':
    configure()
    prior.train.main()

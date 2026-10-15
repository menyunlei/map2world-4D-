import trellis2_map_ss_known_coverage_v4 as probe
from trellis2_official_ss_noise_fix1 import bind
probe.OUT=probe.OUT/'cpu_noise_fix1'
if __name__=='__main__':
    bind(probe.torch);probe.main()

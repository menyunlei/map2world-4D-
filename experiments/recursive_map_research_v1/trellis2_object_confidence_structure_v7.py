"""Reuse EXACT v6 sampling with one proven object-confidence input correction."""
import trellis2_map_object_structure_v6 as probe
from trellis2_object_confidence_prepare_v7 import OUT
probe.OUT=OUT
if __name__=='__main__':probe.main()

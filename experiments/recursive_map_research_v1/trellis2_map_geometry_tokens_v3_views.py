"""Reuse fixed independent native views, with the valid known-open edit input."""
import trellis2_map_entry_views_v2 as view
from trellis2_map_geometry_tokens_v3_prepare import OUT

view.EXPERIMENT=OUT
view.MODE_MAPS={**view.MODE_MAPS,'edit':'explicit_remove_last_building_map.npz'}

if __name__=='__main__':view.main()

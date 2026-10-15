"""Point the unchanged renderer at the corrected-noise artifact root."""
from paths import JOB
import trellis2_map_entry_views_v2 as view
view.EXPERIMENT=JOB/'map_object_world_v1/trellis2_map_address_validity_v5/cpu_noise_fix1'
view.MODE_MAPS={**view.MODE_MAPS,'edit':'explicit_remove_last_building_map.npz'}
if __name__=='__main__':view.main()

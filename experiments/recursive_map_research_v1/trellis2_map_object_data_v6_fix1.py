"""Use registered scene ground -14/32, not an invented world z=0."""
import trellis2_map_object_data_v6 as data
from trellis2_map_object_data_v6 import OUT, OLD, ARMS, SEEDS, local_map, write_npz

original_frame=data.frame_from_map
def frame_from_map(m,index):
    frame=original_frame(m,index)
    frame['center_z'] -= 14/32
    frame['scene_ground_z']=-14/32
    return frame
data.frame_from_map=frame_from_map

if __name__=='__main__': data.prepare()

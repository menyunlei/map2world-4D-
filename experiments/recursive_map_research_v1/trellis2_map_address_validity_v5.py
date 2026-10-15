"""Content confidence is not query-coordinate validity.

Reuse every map value/known channel/object calculation. Only keep the token
MLP output at valid world queries rather than blanking its entire output when
local content is unobserved. No unknown map value is made known, no target
height/mesh is accessed, and all parameters remain those of the trained v3.
"""
from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import MapObjectContext

class AddressValidMapConditioner(Trellis2MapGeometryTokenConditioner):
    def prepare(self,mapbatch,grid):
        captured=[]
        hook=self.token_mlp.register_forward_hook(lambda _m,_args,output:captured.append(output))
        try:original=super().prepare(mapbatch,grid)
        finally:hook.remove()
        assert len(captured)==1
        return MapObjectContext(captured[0],original.object_tokens,original.routing_weights)

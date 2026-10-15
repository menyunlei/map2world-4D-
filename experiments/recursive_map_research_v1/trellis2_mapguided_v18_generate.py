"""Reuse native generation/export, with already-sampled map-guided Structure."""
from pathlib import Path
import sys
from trellis2_mapguided_v18 import OUT,SEED

mode=sys.argv[sys.argv.index('--mode')+1]
assert mode in ['correct','edit']
source=Path(__file__).with_name('trellis2_native_street_v17.py')
code=source.read_text().split("if __name__=='__main__':")[0]
code=code.replace("'no_map_control':True", "'map_control_under_test':True")
needle="    with torch.no_grad():\n        meshes,latents=pipe.run"
assert needle in code
injection='''    from trellis2_mapguided_v18_guidance import AddressedSparseNoise
    with np.load(OUT/'SS.npz') as z: coords=torch.from_numpy(z['coords32'].copy()).int()
    assert len(coords)>0,'EMPTY_STRUCTURE_NO_ASSET_SUBSTITUTION'
    def fixed_structure(self,cond,resolution,*args,**kwargs):
        assert resolution==32,('UNEXPECTED_STRUCTURE_RESOLUTION',resolution)
        return coords.to(self.device)
    pipe.sample_sparse_structure=types.MethodType(fixed_structure,pipe)
    pipe.shape_slat_sampler=AddressedSparseNoise(pipe.shape_slat_sampler,SEED+100000)
    pipe.tex_slat_sampler=AddressedSparseNoise(pipe.tex_slat_sampler,SEED+200000)
    with torch.no_grad():
        meshes,latents=pipe.run'''
code=code.replace(needle,injection)
scope={'__name__':'v18_native_reuse','__file__':str(source)}
exec(compile(code,str(source),'exec'),scope)
scope['OUT']=OUT/mode;scope['SEED']=SEED
scope['generate']()
from paths import read,save,sha
plan=read(OUT/'PLAN.json')
save(OUT/mode/'MAP_GENERATION.json',{'mode':mode,'SS_sha256':sha(OUT/mode/'SS.npz'),
    'model_weight_updates':0,'native_Shape_and_texture_unchanged':True,
    'geometry_image_condition_retained_not_appearance_only':True,
    'map_guidance':'Predicted-clean latent gradient during Structure sampling, no post-output mask or mesh repair',
    'object_placement':'None; whole scene generated in one common native XY frame',
    'coordinate_indexed_sparse_noise':True,'seed':SEED,
    'world_z_translation_for_display':plan['world_from_native_z_translation'],
    'reference_image_is_not_Map10_appearance':True,'map_annotation_is_approximate_roof_proxy':True})

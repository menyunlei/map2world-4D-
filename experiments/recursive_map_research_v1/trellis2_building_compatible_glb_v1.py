"""Blender-compatible base-color preview; native full PBR fields remain in RAW_NATIVE.pt."""
from trellis2_asset_trial_v3 import OUT, DEST, REPO, SEEDS, save, sha
import os
import sys
os.environ['CUDA_HOME']=str(DEST/'environment')
os.environ['PATH']=str(DEST/'environment/bin')+':'+os.environ['PATH']
sys.path.insert(0,str(REPO))
import torch
import numpy as np
import trimesh
from trellis2.representations import MeshWithVoxel

for seed in SEEDS:
    folder=OUT/f'seed{seed}'/'official_building'
    data=torch.load(folder/'RAW_NATIVE.pt',map_location='cpu',weights_only=False)
    spatial=(data['coords'].max(0).values+1).tolist()
    mesh=MeshWithVoxel(data['vertices'].cuda(),data['faces'].cuda(),[-.5,-.5,-.5],data['voxel_size'],
        data['coords'].cuda(),data['attrs'].cuda(),torch.Size([1,data['attrs'].shape[1],*spatial]),data['layout'])
    colors=[];roughness_sum=0.;metallic_sum=0.
    for start in range(0,len(mesh.vertices),250000):
        with torch.no_grad():a=mesh.query_attrs(mesh.vertices[start:start+250000])
        rgba=torch.cat([a[:,data['layout']['base_color']],a[:,data['layout']['alpha']]],dim=1)
        colors.append((rgba.clamp(0,1)*255).byte().cpu().numpy())
        roughness_sum+=float(a[:,data['layout']['roughness']].sum())
        metallic_sum+=float(a[:,data['layout']['metallic']].sum())
    vertices=data['vertices'].float().numpy().copy()
    converted=vertices@np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32).T
    faces=data['faces'].numpy()
    geometry=trimesh.Trimesh(vertices=converted,faces=faces,vertex_colors=np.concatenate(colors),process=False)
    scene=trimesh.Scene(geometry)
    path=folder/'building_basecolor_preview.glb'
    assert not path.exists()
    scene.export(path)
    # Load the independently saved GLB through a fresh parser; no transformations/repairs.
    check=trimesh.load(path,force='scene',process=False)
    exported=list(check.geometry.values())[0]
    assert len(exported.vertices)==len(vertices) and len(exported.faces)==len(faces)
    assert np.max(np.abs(exported.vertices-converted))<1e-6
    save(folder/'COMPATIBLE_GLB.json',{'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
        'vertices':len(vertices),'faces':len(faces),'all_original_geometry_preserved':True,
        'appearance':'native model base color and alpha sampled at unchanged vertices, uint8; GLB vertex-color display',
        'material_limit':'This preview does not encode spatial roughness/metallic fields. Full native PBR is retained in RAW_NATIVE.pt.',
        'native_roughness_vertex_mean':roughness_sum/len(vertices),'native_metallic_vertex_mean':metallic_sum/len(vertices),
        'fresh_parser_verified':True,'training_updates':0})
    print('COMPATIBLE_BUILDING_GLB_READY',seed,path.stat().st_size,flush=True)
    del mesh,data,geometry,scene,check,exported;torch.cuda.empty_cache()

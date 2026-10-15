"""Prepare fixed authorized reference assets and input cameras, before any inference."""
from trellis2_prepare_v1 import *
import numpy as np
import trimesh
from PIL import Image


def asset(data, face_mask, center, span):
    scene=trimesh.Scene()
    faces=data['faces'][face_mask]
    colors=data['vertex_colors']
    # The registered reference has flat authored vertex colors, not photographic textures.
    face_colors=colors[faces]
    assert np.allclose(face_colors,face_colors[:,0:1,:],atol=1e-6), 'Do not silently average texture/colors'
    keys=np.rint(face_colors[:,0]*255).astype(np.uint8)
    verts=(data['vertices_world_m']-center)/span
    for index,color in enumerate(np.unique(keys,axis=0)):
        subset=faces[(keys==color).all(1)]
        used,inverse=np.unique(subset.reshape(-1),return_inverse=True)
        mat=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGB',(1,1),tuple(color)),
            baseColorFactor=[255,255,255,255],roughnessFactor=.85,metallicFactor=0,alphaMode='OPAQUE',doubleSided=True)
        mesh=trimesh.Trimesh(vertices=verts[used],faces=inverse.reshape(-1,3),process=False,
            visual=trimesh.visual.TextureVisuals(uv=np.zeros((len(used),2)),material=mat))
        scene.add_geometry(mesh,node_name=f'authored_color_{index}')
    return scene


def run():
    plan=read(DEST/'PLAN.json')
    for path,digest in plan['reference_hashes'].items():
        assert sha(path)==digest
    d=np.load(Path(plan['reference_scene'])/'target_mesh.npz')
    records=[]
    for name,mask in [('reference_building_1',d['face_object_ids']==1),
                      ('reference_complete_four_building_block',np.ones(len(d['faces']),bool))]:
        indices=np.unique(d['faces'][mask]);points=d['vertices_world_m'][indices]
        lo,hi=points.min(0),points.max(0)
        center=(lo+hi)/2;span=float((hi-lo).max()/0.99999)
        out=DEST/'inputs'/name;out.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(out/'reference_native.npz',vertices=(d['vertices_world_m']-center)/span,
            faces=d['faces'][mask],vertex_colors=d['vertex_colors'],face_object_ids=d['face_object_ids'][mask])
        scene=asset(d,mask,center,span)
        # Library voxelization consumes Z-up; Blender/GLB import consumes Y-up.
        transform=np.eye(4);transform[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        scene.apply_transform(transform)
        scene.export(out/'reference.glb')
        cameras=[]
        for k in range(8):
            angle=2*np.pi*k/8
            cameras.append({'eye':[1.5*np.sin(angle),-1.5*np.cos(angle),1.05],
                            'look':[0,0,0], 'fov_degrees':45,'resolution':[1024,1024]})
        record={'id':name,'source':plan['reference_scene'],'reference_sha256':sha(out/'reference_native.npz'),
            'world_from_native':{'center_world_m':center.tolist(),'meters_per_native_unit':span,
                                  'normalization_from_INPUT_only':True},
            'object_ids':sorted(set(int(x) for x in d['face_object_ids'][mask] if x>0)),
            'original_faces':int(mask.sum()),'cameras':cameras,'image_to_3D_input_view':0,
            'reference_appearance':'authored flat colors, no true urban photographic material'}
        save(out/'INPUT.json',record);records.append(record)
    save(DEST/'INPUT_MANIFEST.json',{'units':records,'fixed_before_generation':True,
         'no_new_training_data':True,'not_urban_reconstruction':True})
    print('FIXED_REFERENCE_INPUTS_READY',flush=True)


if __name__=='__main__':
    run()

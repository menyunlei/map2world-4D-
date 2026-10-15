"""Fresh process: reload the entire GLB; reuse native MeshRenderer, no models."""
from trellis2_map_learning_train_fix1 import base
from trellis2_map_object_data_v6_fix1 import OUT,ARMS,SEEDS
from trellis2_map_entry_views_v2 import camera_matrices
from paths import read,save,sha
from trellis2.representations import Mesh
from trellis2.renderers import MeshRenderer
import numpy as np,torch,trimesh,argparse
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    torch.set_num_threads(4);axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
    protocol=read(OUT/'CAMERA_PROTOCOL.json')
    for mode in ['correct','edit']:
        root=OUT/a.arm/f'seed{a.seed}'/'assets'/mode;record=read(root/'BLOCK_ASSET.json')
        assert sha(record['asset'])==record['sha256']
        scene=trimesh.load(record['asset'],force='scene',process=False)
        vertices=[];faces=[];attrs=[];offset=0
        for node in scene.graph.nodes_geometry:
            T,key=scene.graph[node];g=scene.geometry[key]
            v=trimesh.transformations.transform_points(g.vertices,T)@axis
            vertices.append(v);faces.append(g.faces+offset);offset+=len(v)
            idx=int(node.rsplit('object',1)[-1])
            attrs.append(np.c_[g.visual.vertex_colors.astype(np.float32)/255,np.full(len(v),idx+1)])
        assert sum(len(f) for f in faces)==sum(r['faces'] for r in record['objects'])
        assert offset==sum(r['vertices'] for r in record['objects'])
        mesh=Mesh(torch.tensor(np.concatenate(vertices),dtype=torch.float32,device='cuda'),
            torch.tensor(np.concatenate(faces),dtype=torch.int32,device='cuda'),
            torch.tensor(np.concatenate(attrs),dtype=torch.float32,device='cuda'))
        renderer=MeshRenderer();renderer.rendering_options.resolution=640;renderer.rendering_options.ssaa=1
        renderer.rendering_options.near=.001;renderer.rendering_options.far=10
        renderer.rendering_options.chunk_size=8000000;renderer.rendering_options.antialias=False
        d=root/'views';d.mkdir(exist_ok=False);tiles=[];viewrecords=[]
        for name,eye,look,up,fov in protocol['cameras']:
            K,w2c=camera_matrices(eye,look,up,fov)
            with torch.no_grad():g=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),
                torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['attr','normal','depth','mask'])
            mask=g['mask'].cpu().numpy();col=g['attr'][:3].permute(1,2,0).cpu().numpy()
            neutral=g['normal'].permute(1,2,0).cpu().numpy()
            rgb=col*mask[...,None]+np.array([.16,.19,.23])*(1-mask[...,None])
            neu=neutral*mask[...,None]+.16*(1-mask[...,None])
            im=Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8));im.save(d/f'{name}.png')
            Image.fromarray((neu.clip(0,1)*255).astype(np.uint8)).save(d/f'{name}_neutral.png')
            np.savez_compressed(d/f'{name}_diagnostics.npz',depth=g['depth'].cpu().numpy(),
                object_id=g['attr'][4].cpu().numpy(),mask=mask,K=K,w2c=w2c)
            tile=Image.new('RGB',(640,664),'black');tile.paste(im,(0,24));ImageDraw.Draw(tile).text((4,4),name,fill='white');tiles.append(tile)
            viewrecords.append({'camera':name,'model_visible_fraction':float(mask.mean()),'source':'independently reloaded complete generated GLB','added_geometry':False})
        sheet=Image.new('RGB',(2560,1992),'black')
        for i,t in enumerate(tiles):sheet.paste(t,((i%4)*640,(i//4)*664))
        sheet.resize((1920,1494)).save(d/'contact.jpg',quality=92)
        save(d/'RELOAD.json',{'asset_sha256':record['sha256'],'vertices':offset,
            'faces':sum(len(f) for f in faces),'original_geometry_preserved':True,
            'renderer':'unchanged native TRELLIS.2 MeshRenderer on whole saved GLB vertex colors',
            'world_axis':'single exact glTF->native rotation','fixed_transform_source':'map frames',
            'background':'flat display color only; no procedural road/buildings in scene',
            'views':viewrecords,'visual_review':'AI_PENDING; not human acceptance'})
        print('INDEPENDENT_FULL_BLOCK_RENDERED',a.arm,a.seed,mode,flush=True)
        del mesh,scene;torch.cuda.empty_cache()

if __name__=='__main__':main()

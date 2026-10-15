"""New-process native-asset views, fixed shared world cameras; no generator call."""
from trellis2_map_learning_train_fix1 import base
from paths import JOB, read, save, sha
from trellis2_map_learning_prepare_fix1 import OUT
import torch, numpy as np, math, argparse
import utils3d
from PIL import Image, ImageDraw
from trellis2.representations import MeshWithVoxel
from trellis2.renderers import MeshRenderer


def cameras():
    # Camera planning is derived ONLY from registered input road pixels, not outputs.
    with np.load(OUT/'real_input/map.npz') as z:
        road=z['roads'][0]*z['roads_known'][0]
    rows,cols=np.where(road>.2)
    assert len(rows)>0
    order=np.argsort(rows,kind='stable');i=order[len(order)//2]
    xy=np.array([(cols[i]+.5)/64-.5,(rows[i]+.5)/64-.5])
    floor=-.4375
    position=np.array([xy[0],xy[1],floor+.025])
    direction=np.array([0.,.12,0.])
    return [
        ('top',[0,0,1.5],[0,0,-.35],[0,1,0],35),
        ('oblique',[1,-1,.45],[0,0,-.3],[0,0,1],45),
        ('front',[0,-1.35,-.2],[0,0,-.3],[0,0,1],45),
        ('back',[0,1.35,-.2],[0,0,-.3],[0,0,1],45),
        ('left',[-1.35,0,-.2],[0,0,-.3],[0,0,1],45),
        ('right',[1.35,0,-.2],[0,0,-.3],[0,0,1],45),
        ('street_front',position.tolist(),(position+direction).tolist(),[0,0,1],60),
        ('street_lateral_left',(position+[-.025,0,0]).tolist(),(position+direction).tolist(),[0,0,1],60),
        ('street_lateral_right',(position+[.025,0,0]).tolist(),(position+direction).tolist(),[0,0,1],60)]


def write_img(value,path):
    if value.ndim==2:value=value[None]
    if value.shape[0]==1:value=value.expand(3,-1,-1)
    arr=np.clip(value.detach().float().cpu().numpy().transpose(1,2,0)*255,0,255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[953101,953102],required=True);a=ap.parse_args()
    poses=cameras()
    if not (OUT/'CAMERA_PROTOCOL.json').exists():
        save(OUT/'CAMERA_PROTOCOL.json',{'cameras':poses,'fixed_before_asset_inspection':True,
            'frame':'native unit cube; floor-.4375 is synthetic canonical assumption, NOT measured Map10 elevation',
            'lateral_translation':.05,'map_annotations_not_geo_calibration':True,
            'no_output_based_camera_selection':True})
    else:assert read(OUT/'CAMERA_PROTOCOL.json')['cameras']==[[n,e,l,u,f] for n,e,l,u,f in poses]
    for mode in ['baseline','correct','shuffled','edit']:
        root=OUT/f'seed{a.seed}/assets'/mode
        raw=root/'RAW_NATIVE.pt';d=torch.load(raw,map_location='cpu',weights_only=False)
        spatial=(d['coords'].max(0).values+1).tolist()
        mesh=MeshWithVoxel(d['vertices'].cuda(),d['faces'].cuda(),[-.5,-.5,-.5],d['voxel_size'],
            d['coords'].cuda(),d['attrs'].cuda(),torch.Size([1,d['attrs'].shape[1],*spatial]),d['layout'])
        folder=root/'views';folder.mkdir(exist_ok=False)
        renderer=MeshRenderer()
        renderer.rendering_options.resolution=640
        renderer.rendering_options.near=.001
        renderer.rendering_options.far=10
        renderer.rendering_options.ssaa=1
        renderer.rendering_options.chunk_size=8000000
        renderer.rendering_options.antialias=False
        for name,eye,look,up,fov in poses:
            eye=torch.tensor(eye,dtype=torch.float32,device='cuda')
            look=torch.tensor(look,dtype=torch.float32,device='cuda')
            up=torch.tensor(up,dtype=torch.float32,device='cuda')
            extr=utils3d.torch.extrinsics_look_at(eye,look,up)
            f=torch.tensor(math.radians(fov),device='cuda')
            intr=utils3d.torch.intrinsics_from_fov_xy(f,f)
            with torch.no_grad():gb=renderer.render(mesh,extr,intr,return_types=['attr','mask','normal','depth'])
            write_img(gb['base_color']+.18*(1-gb['mask'])[None],folder/f'{name}.png')
            normal=(gb['normal']*.6+.25)*gb['mask'][None]+.18*(1-gb['mask'])[None]
            write_img(normal,folder/f'{name}_geometry.png')
            np.savez_compressed(folder/f'{name}_diagnostic.npz',depth=gb['depth'].detach().float().cpu().numpy(),
                visible_mask=gb['mask'].detach().cpu().numpy(),w2c=extr.cpu().numpy(),K=intr.cpu().numpy())
            print('SAVED_NATIVE_VIEW',a.seed,mode,name,flush=True)
        contact=Image.new('RGB',(3*320,3*345),'#202020');draw=ImageDraw.Draw(contact)
        for j,(name,*_) in enumerate(poses):
            im=Image.open(folder/f'{name}.png');im.thumbnail((320,320))
            x,y=(j%3)*320,(j//3)*345;contact.paste(im,(x,y));draw.text((x+5,y+322),name,fill='white')
        contact.save(folder/'contact.jpg',quality=92)
        save(folder/'RELOAD.json',{'raw_asset_sha256':sha(raw),'fresh_process':True,'model_sampling':False,
            'geometry_or_material_modified':False,'native_faces':len(mesh.faces),'native_vertices':len(mesh.vertices),
            'shader':'native MeshRenderer, chunked all triangles, unlit original base-color and geometry normals',
            'PBR_lighting_not_claimed':True,'camera_protocol_sha256':sha(OUT/'CAMERA_PROTOCOL.json'),
            'no_map_control_success_claim_from_export':True})
        del mesh,d;torch.cuda.empty_cache()
    save(OUT/f'seed{a.seed}/ASSET_RELOAD_COMPLETE.json',{'modes':4,'views_each':9,'fresh_saved_native_assets':True})


if __name__=='__main__':main()

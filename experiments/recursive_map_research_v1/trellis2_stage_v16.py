"""Bounded stage attribution using exact existing paired native outputs."""
from paths import JOB, read, save, sha
from pathlib import Path
import argparse, numpy as np, time, json

OUT=JOB/'map_object_world_v1/trellis2_stage_attribution_v16'
V15=JOB/'map_object_world_v1/trellis2_map10_street_v15'
V12=JOB/'map_object_world_v1/trellis2_coverage_v12'
V9=JOB/'map_object_world_v1/trellis2_map_edit_mesh_v9'
SEEDS=[953101,953102]

def prepare():
    p=read(V15/'PLAN.json'); old=read(V9/'PLAN.json'); coverage=read(V12/'PLAN.json')
    assert not OUT.exists()
    units=[]; frozen={}
    for seed in SEEDS:
        c=p['Shape_condition_checkpoints'][str(seed)]
        assert c['sha256']==old['Shape_condition_checkpoints'][str(seed)]['sha256']
        assert p['shape_address_seed']==old['shape_address_seed']==coverage['shape_address_seed']
        assert p['structure_generation_seed']==coverage['generation_seeds'][0]
        for i,r in enumerate(old['records']):
            rr=coverage['evaluation_records'][i]
            assert r['native_target_sha256']==rr['native_target_sha256']
            assert r['map_sha256']==rr['map_sha256']
            for stage,src in [
                ('predicted',V12/f'evaluation/seed{seed}/six_object_rehearsal/update001000/noise970201/correct/object{i:03d}'),
                ('oracleQ',V9/f'seed{seed}/oracleQ_diagnostic/object{i}'),
                ('codec',V9/f'seed953101/codec_reference_diagnostic/object{i}')]:
                if stage=='codec' and seed!=SEEDS[0]:continue
                assert (src/'ASSET.json').exists(),src
                if stage=='predicted':
                    assert read(src/'ASSET.json')['shape_condition_checkpoint']['sha256']==c['sha256']
                units.append({'stage':stage,'seed':seed,'index':i,'record':r,'source':str(src),
                              'reuse':True,'target_diagnostic_only':stage!='predicted'})
    for seed in SEEDS:
        for i,r in enumerate(p['records']):
            units.append({'stage':'Map10','seed':seed,'index':i,'record':r,
                'source':str(V15/f'seed{seed}/correct/object{i}'),'reuse':True,'no_GT':True})
    for u in units:
        d=Path(u['source'])
        for name in ['ASSET.json','geometry.glb','NATIVE_MESH.npz','SS_GENERATED.npz','SHAPE_SLAT.npz']:
            if (d/name).exists():frozen[str(d/name)]=sha(d/name)
        if u['stage']!='Map10':
            r=u['record'];assert r['split']=='adapter_TRAIN'
            for key in ['native_target','map_input']:frozen[r[key]]=sha(r[key])
    for group in ['SS_checkpoints','Shape_condition_checkpoints']:
        for c in p[group].values():frozen[c['path']]=c['sha256']
    save(OUT/'PLAN.json',{'scope':'Existing paired TRAIN stage attribution; Map10 has no GT',
        'GPU_cap_seconds':28800,'training_max_updates':100,'one_repair_only':True,
        'no_texture_or_geometry_repair':True,'units':units,'seeds':SEEDS,
        'GPUs':coverage['GPUs'][:2]+[
            {'index':2,'uuid':'GPU-006fc57d-f17f-7ea8-7ff0-a11a85b16c71'},
            {'index':3,'uuid':'GPU-07f3805a-2bfa-4594-0495-1b254ab0f80d'}],
        'stage_GPU':{'predicted':0,'oracleQ':1,'codec':2,'Map10':3},
        'cameras':[['front',[0,-1,-.12],[0,0,-.12],[0,0,1],50],
                   ['lateral',[.35,-1,-.12],[0,0,-.12],[0,0,1],50],
                   ['top',[0,0,1.1],[0,0,-.12],[0,1,0],50],
                   ['oblique',[.8,-.8,.7],[0,0,-.12],[0,0,1],50]]})
    frozen[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    frozen[str(JOB/'trellis2_stage_v16.py')]=sha(JOB/'trellis2_stage_v16.py')
    frozen[str(JOB/'trellis2_stage_v16_owner.py')]=sha(JOB/'trellis2_stage_v16_owner.py')
    save(OUT/'SOURCE_FREEZE.json',frozen)
    print('STAGE_REGISTERED',len(units),len(frozen),flush=True)

def worker(stage):
    from trellis2_map_edit_mesh_v9_views import load_mesh,footprint,camera_matrices,MeshRenderer
    import torch
    from scipy.ndimage import label
    from PIL import Image,ImageDraw
    torch.set_num_threads(4);p=read(OUT/'PLAN.json')
    renderer=MeshRenderer();renderer.rendering_options.resolution=384;renderer.rendering_options.ssaa=1
    renderer.rendering_options.near=.001;renderer.rendering_options.far=10
    renderer.rendering_options.antialias=False;renderer.rendering_options.chunk_size=8000000
    def render(mesh,eye,look,up,fov):
        K,w2c=camera_matrices(eye,look,up,fov)
        with torch.no_grad():r=renderer.render(mesh,torch.tensor(w2c,dtype=torch.float32,device='cuda'),
            torch.tensor(K,dtype=torch.float32,device='cuda'),return_types=['normal','mask'])
        mask=r['mask'].cpu().numpy();n=r['normal'].permute(1,2,0).cpu().numpy()*2-1
        light=np.array([-.35,.5,.8]);light/=np.linalg.norm(light)
        s=.28+.72*np.abs((n*light).sum(-1)).clip(0,1)
        rgb=np.array([.76,.79,.82])*s[...,None]*mask[...,None]+np.array([.075,.09,.12])*(1-mask[...,None])
        return Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8))
    rows=[]
    for u in [u for u in p['units'] if u['stage']==stage]:
        d=OUT/'diagnosis'/stage/f"seed{u['seed']}"/f"object{u['index']}";d.mkdir(parents=True,exist_ok=False)
        src=Path(u['source']);row={**u,'source_geometry_sha256':None};support=None
        if stage in ['oracleQ','codec']:
            with np.load(u['record']['native_target']) as z:support=z['support64'][0,0].astype(bool)
        else:
            with np.load(src/'SS_GENERATED.npz') as z:support=z['native_support64'].astype(bool)
        xyz=np.argwhere(support);_,n=label(support)
        row.update({'support_cells':len(xyz),'support_components':int(n),
            'support_native_bounds':[((xyz.min(0)+.5)/64-.5).tolist(),((xyz.max(0)+.5)/64-.5).tolist()] if len(xyz) else None,
            'empty_at_Structure':not bool(len(xyz))})
        im=Image.new('RGB',(1152,400),'#131922');draw=ImageDraw.Draw(im)
        for j,proj in enumerate([support.any(2).T,support.any(1).T,support.any(0).T]):
            rgb=np.zeros((64,64,3),np.uint8)+20;rgb[proj]=[170,185,200]
            im.paste(Image.fromarray(rgb[::-1]).resize((384,384),Image.Resampling.NEAREST),(j*384,16))
        draw.text((4,1),f"{stage} seed{u['seed']} object{u['index']} native Structure XY / XZ / YZ",fill='white')
        im.save(d/'SUPPORT.png')
        if (src/'ASSET.json').exists():
            mesh,v,f,meta=load_mesh(src)
            row.update({'source_geometry_sha256':meta['sha256'],'vertices':len(v),'faces':len(f),
                'mesh_native_bounds':[v.min(0).tolist(),v.max(0).tolist()],
                'mesh_below_support_min_z_fraction':float((v[:,2] < (xyz[:,2].min()+.5)/64-.5-1/64).mean()) if len(xyz) else None,
                'independent_asset_reload':True,'no_geometry_changed':True})
            for name,eye,look,up,fov in p['cameras']:render(mesh,eye,look,up,fov).save(d/f'{name}.png')
            del mesh;torch.cuda.empty_cache()
        else:row['no_mesh_reason']='EMPTY_STRUCTURE; no substitute'
        save(d/'METRICS.json',row);rows.append(row)
        print('STAGE_CASE',stage,u['seed'],u['index'],row['support_cells'],row.get('faces'),flush=True)
    save(OUT/f'{stage}_COMPLETE.json',{'rows':rows,'all_exact_outputs_reused':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--stage',choices=['predicted','oracleQ','codec','Map10']);a=p.parse_args()
    prepare() if a.prepare else worker(a.stage)

"""Official per-scene GS optimization with an honest partial-view data adapter."""
import map_object_runtime_overlay_v1
from pathlib import Path
import sys,os,json,ast,shutil,time,hashlib,subprocess,runpy,argparse
JOB=Path(__file__).resolve().parent
DEST=JOB/'map_object_world_v1'
PHASE=DEST/'world_composition_v1'
WORLD=DEST/'vendor/HY-World-2.0/hyworld2/worldgen'
sys.path[:0]=[str(WORLD/'third_party/gsplat_maskgaussian'),str(WORLD),str(PHASE/'dependencies')]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,indent=2)+'\n')
def extract(path,name,scope):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]
def register():
    PHASE.mkdir(exist_ok=True)
    for f in ('dependencies','extensions','logs','audit','gpu_cost','evidence'): (PHASE/f).mkdir(exist_ok=True)
    code=[WORLD/'world_gs_trainer.py',WORLD/'gen_gs_data.py',WORLD/'gs/opencv.py',WORLD/'gs/utils.py',JOB/'map_object_composition_v1.py',JOB/'map_object_composition_entry_v1.py']
    code+=sorted((WORLD/'third_party/gsplat_maskgaussian/gsplat').rglob('*.py'))
    save(PHASE/'SOURCE_FREEZE.json',{str(p):sha(p) for p in code})
    save(PHASE/'PLAN.json',{'source_seeds':[832001,832002],'fitting_rng_seed':42,
        'task_kind':'per_scene_3DGS_fitting','map_model_training':False,
        'max_steps_per_scene':8000,'wall_limit_seconds_per_scene':7200,
        'official_recipe':'one GPU per scene; official README recommends max_steps=8000',
        'native_densification_periods':{'refine_start_iter':800,'refine_stop_iter':4000,'refine_every':533,'refine_scale2d_stop_iter':4000,'reset_every':99990},
        'input_adapter':'existing 12 uniformly selected RGB views, WorldMirror cameras/depth/points plus official MoGe2 normals; reuse gen_gs_data.save_io',
        'not_full_panorama_generation_bank':True,'no_fabricated_sky_or_back_views':True,
        'trainer':'unchanged official world_gs_trainer.py; custom gsplat_maskgaussian',
        'no_normalize':True,'coordinates':'same original predicted relative world frame, not map registered',
        'precision':'Gaussian optimizer/render FP32; MoGe2 normal inference native FP32',
        'evaluation':'same saved aerial/9 views/288-frame 12s camera path, fresh process from optimized checkpoint',
        'fixed_before_results':True,'no_seed_or_parameter_search':True})
    for seed in (832001,832002):
        for kind,suffix in [('audit','LAUNCH.json'),('logs','log')]:
            name=f'subcell_city_gate_{seed}_reconstruct'
            src=DEST/kind/(name+('_'+suffix if kind=='audit' else '.'+suffix))
            if src.exists():
                out=PHASE/'evidence'/(name+('.json' if kind=='audit' else '_original_log.md'))
                shutil.copy2(src,out)
    print('COMPOSITION_REGISTERED',flush=True)
def prepare(seed):
    import torch,numpy as np
    from PIL import Image
    from plyfile import PlyData,PlyElement
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import prepare_images_to_tensor
    from moge.model.v2 import MoGeModel
    from gs.utils import load_16bit_png_depth
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    root=PHASE/f'seed{seed}';root.mkdir(exist_ok=False)
    data=root/'gs_data';data.mkdir()
    for f in ('images','depths','normals','temp_depths'): (data/f).mkdir()
    source=DEST/'city_asset_gate_v1'/f'seed{seed}'
    raw=torch.load(source/'NATIVE_RAW_PREDICTIONS.pt',map_location='cpu',weights_only=True)
    h,w=raw['depth'].shape[2:4]
    paths=sorted((source/'inputs').glob('*.png'))
    images=prepare_images_to_tensor([str(p) for p in paths],target_size=w,resize_strategy='crop')
    assert list(images.shape[-2:])==[h,w]
    fn=extract(WORLD/'gen_gs_data.py','save_io',{'Image':Image,'np':np,'shutil':shutil})
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()},root/'NORMAL_RNG_BEFORE.pt')
    moge=MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval();moge.requires_grad_(False)
    cameras={};normal_stats=[];all_colors=[]
    for i,path in enumerate(paths):
        x=images[0,i] if images.ndim==5 else images[i]
        assert x.shape==(3,h,w)
        rgb=(x.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
        pred=moge.infer(x.cuda()[None])
        normals=pred['normal'][0].float().cpu().numpy()
        assert normals.shape==(h,w,3) and np.isfinite(normals).all()
        z=raw['depth'][0,i,:,:,0].numpy();assert np.isfinite(z).all()
        tmp=data/'temp_depths'/f'{i:03d}.png'
        Image.fromarray(z.astype(np.float16).view(np.uint16)).save(tmp)
        name,w2c,K=fn(f'video_{i:03d}',Image.fromarray(rgb),str(tmp),normals,
            np.linalg.inv(raw['camera_poses'][0,i].numpy()),raw['camera_intrs'][0,i].numpy(),str(data))
        check=load_16bit_png_depth(str(data/'depths'/f'{name}.png'))
        assert np.array_equal(check,z.astype(np.float16).astype(np.float32))
        cameras[name]={'extrinsic':w2c,'intrinsic':K}
        normal_stats.append({'image':name,'valid_fraction':float((np.linalg.norm(normals,axis=-1)>.5).mean())})
        all_colors.append(rgb.reshape(-1,3))
        print('OFFICIAL_SAVE_IO',seed,name,'depth_fp16_bits',flush=True)
    points=raw['pts3d'][0].reshape(-1,3).numpy()
    colors=np.concatenate(all_colors);assert len(points)==len(colors) and np.isfinite(points).all()
    vertices=np.empty(len(points),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for j,k in enumerate(('x','y','z')):vertices[k]=points[:,j]
    for j,k in enumerate(('red','green','blue')):vertices[k]=colors[:,j]
    PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(str(data/'points.ply'))
    save(data/'cameras.json',cameras)
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()},root/'NORMAL_RNG_AFTER.pt')
    save(root/'DATA_MANIFEST.json',{'seed':seed,'source_raw_sha256':sha(source/'NATIVE_RAW_PREDICTIONS.pt'),
        'source_input_manifest':json.loads((source/'INPUT_MANIFEST.json').read_text()),
        'adapter':'official save_io function reused verbatim; no gen_gs_data CLI due to absent generation_bank/panorama',
        'cameras':'WorldMirror c2w inverted to official extrinsic w2c; no frame transform',
        'depth':'WorldMirror z-depth FP16 bit-pattern PNG, official loader roundtrip checked',
        'normals':'official pretrained MoGe2 normal model, camera-frame native output',
        'normal_stats':normal_stats,'initial_points':len(points),'sky_points':'absent, not fabricated',
        'files':{str(p.relative_to(data)):sha(p) for p in data.rglob('*') if p.is_file()}})
    print('COMPOSITION_DATA_READY',seed,len(cameras),len(points),flush=True)
def fit(seed,steps,preflight):
    import torch,numpy as np
    root=PHASE/f'seed{seed}'
    output=root/('preflight' if preflight else 'optimized')
    assert not output.exists(),str(output)
    output.mkdir()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()},output/'RNG_BEFORE.pt')
    args=['default','--data_dir',str(root/'gs_data'),'--result_dir',str(output),
        '--max_steps',str(steps),'--save_steps',str(steps),'--eval_steps',str(steps),'--ply_steps',str(steps),
        '--save_ply','--disable_video','--disable_viewer','--no-normalize',
        '--use_scale_regularization','--antialiased','--depth_loss','--normal_loss',
        '--use_mask_gaussian','--mask_export_stochastic','--no-mask-export-anchor-protection','--use_anchor_protection',
        '--strategy.refine-start-iter','800','--strategy.refine-stop-iter','4000','--strategy.refine-every','533',
        '--strategy.refine-scale2d-stop-iter','4000','--strategy.reset-every','99990',
        '--strategy.grow-grad2d','0.0001','--strategy.prune-scale3d','0.1']
    # Sky-depth and mesh/SPZ export require absent panorama/sky evidence or optional conversion tools.
    # Their omission is recorded, not silently portrayed as the complete panoramic pipeline.
    save(output/'EXECUTED_COMMAND.json',{'entry':str(WORLD/'world_gs_trainer.py'),'argv':args,
        'task_kind':'per_scene_3DGS_fitting','map_model_training':False,
        'omitted_flags':['sky_depth_from_pcd (no sky_points)','export_mesh (native Gaussian deliverable)','convert_to_spz (optional compression)'],
        'optimizer_seed':42,'precision':'FP32','torch':torch.__version__})
    print('EXECUTING_OFFICIAL_WORLD_GS_TRAINER',json.dumps(args),flush=True)
    sys.argv=[str(WORLD/'world_gs_trainer.py'),*args]
    os.chdir(WORLD)
    runpy.run_path(str(WORLD/'world_gs_trainer.py'),run_name='__main__')
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()},output/'RNG_AFTER.pt')
    save(output/'COMPLETED.json',{'steps':steps,'map_model_training':False,'task_kind':'per_scene_3DGS_fitting','seed':seed})
if __name__=='__main__':
    sys.path.insert(0,str(DEST/'vendor/HY-World-2.0'))
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','prepare','fit','install']);p.add_argument('--seed',type=int);p.add_argument('--steps',type=int,default=8000);p.add_argument('--preflight',action='store_true');a,extra=p.parse_known_args()
    if a.mode=='install':
        os.environ.pop('HF_HUB_OFFLINE',None)
        sys.argv=['pip','install','--target',str(PHASE/'dependencies'),'--no-deps',*extra]
        runpy.run_module('pip',run_name='__main__')
    elif a.mode=='register':register()
    elif a.mode=='prepare':prepare(a.seed)
    else:fit(a.seed,a.steps,a.preflight)

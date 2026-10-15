"""Resume saved SS; only repair native sparse-coordinate memory contiguity."""
from trellis2_map_edit_mesh_v9_run import *

def run_fixed(seed):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    plan=read(OUT/'PLAN.json');root=OUT/f'seed{seed}'
    assert (root/'PIPELINE_CONTRACT.json').exists() and not list(root.glob('*/object*/ASSET.json'))
    cfg=read(JOB/'map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json')['args']
    fine=ObjectContractConditioner(reading_mode='object_grounded').cuda().eval().requires_grad_(False)
    parent=plan['Shape_condition_checkpoints'][str(seed)];assert sha(parent['path'])==parent['sha256']
    fine.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['branch_state'])
    mods={k:models.from_pretrained(cfg['models'][k]).eval().requires_grad_(False) for k in ['shape_slat_flow_model_1024','shape_slat_decoder']}
    scfg=cfg['shape_slat_sampler']
    pipe=Trellis2ImageTo3DPipeline(models=mods,shape_slat_sampler=AddressedNoise(FlowEulerGuidanceIntervalSampler(**scfg['args']),plan['shape_address_seed'],1),
      shape_slat_sampler_params=scfg['params'],shape_slat_normalization=cfg['shape_slat_normalization'],low_vram=True)
    pipe.cuda()
    save(root/'CONTIGUITY_FIX.json',{'coordinate_values_unchanged':True,'sole_fix':'contiguous storage for coords/features required by flex_gemm',
      'SS_reused_not_regenerated':True,'old_logs_preserved':True,'sampling_and_weights_unchanged':True})
    for mode in ['correct','edit','oracleQ_diagnostic','codec_reference_diagnostic']:
        for i,r in enumerate(plan['records']):
            if mode=='codec_reference_diagnostic' and seed!=SEEDS[0]:continue
            d=root/mode/f'object{i}';d.mkdir(parents=True,exist_ok=True);assert not (d/'ASSET.json').exists()
            tick=time.time()
            with torch.no_grad():
                if mode=='codec_reference_diagnostic':
                    with np.load(r['native_target']) as z:
                        shape=SparseTensor(feats=torch.from_numpy(z['feats'].copy()).float().cuda().contiguous(),coords=torch.from_numpy(z['coords'].copy()).int().cuda().contiguous())
                else:
                    if mode=='oracleQ_diagnostic':
                        with np.load(r['native_target']) as z:support=z['support64'][0,0].astype(bool)
                    else:
                        with np.load(d/'SS_GENERATED.npz') as z:support=z['native_support64']
                    if not support.any():save(d/'EMPTY.json',{'no_repair_or_retry':True});continue
                    coords=torch.from_numpy(np.c_[np.zeros(int(support.sum()),np.int32),np.argwhere(support)]).int().cuda().contiguous()
                    assert coords.is_contiguous()
                    m=maps(plan,'edit' if mode=='edit' else 'correct',i,False)
                    features=fine(fine.prepare(m,native_world_grid(device='cuda'))).features
                    torch.manual_seed(plan['shape_address_seed'])
                    shape=pipe.sample_shape_slat({'cond':features,'neg_cond':torch.zeros_like(features)},mods['shape_slat_flow_model_1024'],coords)
                meshes,subs=pipe.decode_shape_slat(shape,1024)
                export(meshes[0],shape,d,{'mode':mode,'object_index':i,'training_seed':seed,'frame':r['frame'],
                  'diagnostic_only':mode.endswith('diagnostic'),'reference_not_in_normal_inference':True,'seconds':time.time()-tick})
                del meshes,subs,shape
            gc.collect();torch.cuda.empty_cache()
    for mode in ['correct','edit']:
        scene=trimesh.Scene()
        for i,r in enumerate(plan['records']):
            p=root/mode/f'object{i}/geometry.glb'
            if not p.exists():continue
            obj=trimesh.load(p,force='scene',process=False);g=list(obj.geometry.values())[0]
            f=r['frame'];T=np.eye(4);T[:3,:3]*=f['scale'];T[:3,3]=np.array([*f['center_xy'],f['center_z']])@AXIS.T
            scene.add_geometry(g,geom_name=f'object{i}',node_name=f'object{i}',transform=T)
        scene.export(root/mode/'two_object_world.glb')
    save(root/'COMPLETE.json',{'training':False,'two_objects':True,'textures':False,'history_preserved':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);run_fixed(p.parse_args().seed)

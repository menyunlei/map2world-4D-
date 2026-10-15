"""Same one native64 test: contiguous sparse-coordinate ABI repair only."""
from pathlib import Path
p=Path(__file__).with_name('v28_native64.py')
s=p.read_text().split("\nif __name__=='__main__':")[0]
s=s.replace("D=OUT/'native64'", "D=OUT/'native64_contiguous'")
s=s.replace("coords=torch.tensor(expected,device=self.device,dtype=torch.int32)","coords=torch.tensor(expected,device=self.device,dtype=torch.int32).contiguous()")
s=s.replace("idx=2 if which=='generate' else 3;uuid=GPUS[idx][1]", """owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    free=[(i,u) for i,u in GPUS if u not in owners]
    assert free,'No idle authorized GPU'
    idx,uuid=free[0]""")
s=s.replace("OUT/f'ONE_{which.upper()}.json'", "OUT/f'ONE_{which.upper()}_RECOVERY.json'")
s=s.replace("OUT/f'LAUNCH_{which.upper()}.json'", "OUT/f'LAUNCH_{which.upper()}_RECOVERY.json'")
s=s.replace("OUT/f'GPU_COST_{which.upper()}.json'", "OUT/f'GPU_COST_{which.upper()}_RECOVERY.json'")
s=s.replace("OUT/f'{which}.log'", "OUT/f'{which}_recovery.log'")
s=s.replace("limit=480 if which=='generate' else 90", "limit=480-read(OUT/'GPU_COST_GENERATE.json')['gpu_seconds'] if which=='generate' else 90")
needle='        pipe.sample_sparse_structure=types.MethodType(frozen_structure,pipe)'
s=s.replace(needle,needle+"""
        for method,filename in [('sample_shape_slat','SHAPE_LATENT.pt'),('sample_tex_slat','TEXTURE_LATENT.pt')]:
            fn=getattr(pipe,method)
            def capture(*args,_fn=fn,_filename=filename,**kwargs):
                value=_fn(*args,**kwargs)
                torch.save({'coords':value.coords.detach().cpu().contiguous(),'feats':value.feats.detach().cpu().contiguous()},D/_filename)
                return value
            setattr(pipe,method,capture)
""")
exec(compile(s,str(p),'exec'),globals())
if __name__=='__main__':
    if '--prepare' in sys.argv:
        assert not D.exists();D.mkdir()
        import shutil
        shutil.copyfile(OUT/'native64/input.png',D/'input.png')
        plan=read(OUT/'native64/PLAN.json');plan['input']=str(D/'input.png');save(D/'PLAN.json',plan)
        save(OUT/'ABI_FIX.json',dict(change='coords.contiguous(): identical numerical values and row order',
             original_script_sha256=sha(p),wrapper_sha256=sha(__file__),old_failure_retained=True,
             remaining_generation_GPU_seconds=480-read(OUT/'GPU_COST_GENERATE.json')['gpu_seconds'],
             new_model_candidate=False,seed=942271,training_updates=0))
        print('V28_ABI_ONLY_RECOVERY_PREPARED',flush=True)
    elif '--work-generate' in sys.argv:generate()
    elif '--work-verify' in sys.argv:verify()
    elif '--verify' in sys.argv:owner('verify')
    elif '--detach' in sys.argv:
        with (OUT/'generation_owner_recovery.log').open('x') as f:child=subprocess.Popen([sys.executable,'-B','-u',__file__],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid)),flush=True)
    else:owner('generate')

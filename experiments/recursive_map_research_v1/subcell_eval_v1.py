"""Matched native inference, plus fixed within-cell correspondence intervention."""
from paths import *
from subcell_runtime_v1 import initialize, NEW
import sys,argparse,copy
p=argparse.ArgumentParser(add_help=False);p.add_argument('--arm',required=True)
p.add_argument('--mode',choices=['structure','native','transmittance'],required=True)
p.add_argument('--shuffled',action='store_true');a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
arm=a.arm+'_shuffled' if a.shuffled else a.arm
if a.shuffled:
    assert a.arm=='narrow'
    opts=argparse.ArgumentParser();opts.add_argument('--candidate');opts.add_argument('--seed',type=int);opts.add_argument('--update',type=int)
    args=opts.parse_args(rest)
    dest=NEW/arm/'runs/g2_b'/f'seed{args.seed}/CHECKPOINT_{args.update:06d}.json'
    if not dest.exists():save(dest,read(NEW/a.arm/'runs/g2_b'/f'seed{args.seed}/CHECKPOINT_{args.update:06d}.json'))
    dest=NEW/arm/'packets'/f'g2_b_{args.update}.json'
    if not dest.exists():
        packet=copy.deepcopy(read(NEW/a.arm/'packets'/f'g2_b_{args.update}.json'))
        packet['rows']=[r for r in packet['rows'] if r['family'].startswith('courtyard_new_') and r['seed_index']==0]
        save(dest,packet)
if a.mode=='structure':
    import infer_structure as engine
    engine.initialize=lambda candidate,seed:initialize(a.arm,seed,'shuffled' if a.shuffled else None)
elif a.mode=='native':import full_chain as engine
else:
    path=JOB/'narrow_transmittance_v1.py';ns={'__name__':'subcell_rays','__file__':str(path)}
    exec(compile(path.read_text(),str(path),'exec'),ns);ns['NEW']=NEW
    opts=argparse.ArgumentParser();opts.add_argument('--candidate');opts.add_argument('--seed');opts.add_argument('--update')
    args=opts.parse_args(rest);sys.argv=[sys.argv[0],'--arm',arm,'--seed',args.seed,'--update',args.update]
    ns['main']();raise SystemExit(0)
engine.JOB=NEW/arm;engine.main()

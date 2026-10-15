"""Linux Landlock: candidate reads only TRAIN/code/weights, writes its own run.

Not a hostile-root security boundary. Enforced below Python, inherited across
exec. TCP connect/bind denied. Candidate code cannot open the verifier or DEV
targets, or rewrite historical outputs through another Python/file API.
"""
import ctypes,errno,json,os,sys,runpy
from pathlib import Path

READ=(1<<0)|(1<<2)|(1<<3)
WRITE=(1<<1)|(1<<4)|(1<<5)|(1<<7)|(1<<8)|(1<<9)|(1<<10)|(1<<12)|(1<<13)|(1<<14)
HANDLED=(1<<15)-1

class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64),('handled_access_net',ctypes.c_uint64)]
class Rule(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int)]

def restrict(read_paths,write_root,list_paths=()):
    libc=ctypes.CDLL(None,use_errno=True)
    abi=libc.syscall(444,0,0,1)
    if abi<4:raise RuntimeError(f'Landlock ABI{abi} insufficient; do not run unsandboxed')
    rs=Ruleset(HANDLED,3)
    fd=libc.syscall(444,ctypes.byref(rs),ctypes.sizeof(rs),0)
    if fd<0:raise OSError(ctypes.get_errno(),'create Landlock ruleset')
    def allow(path,access):
        p=Path(path).resolve()
        if not p.exists():return
        if p.is_file():access &= (1<<0)|(1<<1)|(1<<2)|(1<<14)
        handle=os.open(str(p),os.O_PATH|os.O_CLOEXEC)
        try:
            r=Rule(access,handle)
            if libc.syscall(445,fd,1,ctypes.byref(r),0):raise OSError(ctypes.get_errno(),f'Landlock allow {p}')
        finally:os.close(handle)
    try:
        for p in sorted(set(read_paths)):allow(p,READ)
        for p in list_paths:allow(p,1<<3)
        # CUDA device access is not permission to create/delete device nodes.
        allow('/dev',READ|(1<<1))
        # CUDA/Numba create POSIX shared-memory synchronization files. This
        # kernel namespace contains no project datasets or verifier assets.
        allow('/dev/shm',READ|WRITE)
        allow(write_root,READ|WRITE)
        if libc.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'no_new_privs')
        if libc.syscall(446,fd,0):raise OSError(ctypes.get_errno(),'restrict_self')
    finally:os.close(fd)
    return abi

def main():
    policy=json.loads(Path(sys.argv[1]).read_text());out=Path(policy['write_root'])
    # Trusted bootstrap only: CUDA's driver initialization needs OS resources
    # outside project files. Initialize before Landlock, without importing any
    # candidate, model, dataset or verifier. Keep that context (do not exec).
    cuda_bootstrap=False
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        import torch
        torch.cuda.init()
        cuda_bootstrap=True
    abi=restrict(policy['read_paths'],str(out),policy.get('list_paths',[]))
    evidence={}
    for name,path in policy['deny_probes'].items():
        try:
            with open(path,'rb') as f:f.read(1)
        except PermissionError:evidence[name]='DENIED'
        else:raise RuntimeError(f'candidate isolation failure: {name}')
    with (out/policy['probe_output']).open('x') as f:json.dump({'landlock_abi':abi,'read_denials':evidence,'write_root':str(out),'TCP_network':'DENIED','trusted_CUDA_bootstrap':cuda_bootstrap},f)
    args=sys.argv[2:]
    if args[0]=='-c':
        exec(compile(args[1],'<sandbox-probe>','exec'),{'__name__':'__main__'})
    else:
        sys.argv=args;sys.path.insert(0,str(Path(args[0]).resolve().parent))
        runpy.run_path(args[0],run_name='__main__')

if __name__=='__main__':main()

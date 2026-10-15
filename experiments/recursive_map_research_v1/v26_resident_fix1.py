"""One queue-publication repair, same models/samples; preserve original worker.

First worker used269.84GPU-seconds; replacement lease capped630seconds,
so both physical lease windows together remain below the original900s budget.
"""
from pathlib import Path
source=Path(__file__).with_name('v26_resident.py').read_text().split("\nif __name__=='__main__':")[0]
source=source.replace("save(OUT/'MODEL_READY.json'", "save(OUT/'MODEL_READY_FIX1.json'")
source=source.replace('hits=[]',"hits=read(OUT/'CACHE_HITS.json') if (OUT/'CACHE_HITS.json').exists() else []")
source=source.replace('completed=[];idle=time.time()',"completed=[read(p)['id'] for p in sorted((OUT/'chunks').glob('*/COMPLETE.json'))];idle=time.time()")
source=source.replace("d=waiting[0];p=read(d/'PLAN.json');request=read(d/'REQUEST.json')", "d=waiting[0];p=read(d/'PLAN.json')\n        try:request=read(d/'REQUEST.json')\n        except json.JSONDecodeError:time.sleep(.1);continue")
for name in ['ONE_OWNER','OWNER_LAUNCH','GPU_COST']:
    source=source.replace(f"OUT/'{name}.json'",f"OUT/'{name}_FIX1.json'")
source=source.replace("OUT/'worker.log'","OUT/'worker_FIX1.log'")
source=source.replace('>=900','>=630').replace('limit_seconds=900','limit_seconds=630')
source=source.replace('code=child.wait()',"code=child.wait()\n            if code:status('failed',reason='worker_process_exit',returncode=code)")
exec(compile(source,str(Path(__file__).with_name('v26_resident.py')),'exec'),globals())
if __name__=='__main__':
    if '--work'in sys.argv:work()
    elif '--detach'in sys.argv:
        # This repair may resume only a request that never began a sample.
        assert not (OUT/'chunks/walk-0--5/ONE_ATTEMPT.json').exists()
        assert read(OUT/'GPU_COST.json')['returncode']!=0
        save(OUT/'QUEUE_REPAIR.json',dict(epoch=time.time(),source_sha256=sha(Path(__file__)),
            reason='SFTP request visible before write/close; use atomic publication',
            prior_cost=read(OUT/'GPU_COST.json'),remaining_lease_limit_seconds=630,
            original_assets_unchanged=True,no_new_sample_or_seed=True))
        with (OUT/'owner_FIX1.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--owner'],cwd=JOB,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(owner_pid=child.pid,queue_repair_only=True)),flush=True)
    else:owner()

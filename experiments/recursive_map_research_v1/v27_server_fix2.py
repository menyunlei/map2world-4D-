"""Same persistent HTTP cycle; watch final bounded recovery lease as well."""
from pathlib import Path
p=Path(__file__).with_name('v27_server_fix1.py')
s=p.read_text().replace("'GPU_COST_FIX1.json','OWNER_LAUNCH_FIX1.json'", "'GPU_COST_FIX1.json','OWNER_LAUNCH_FIX1.json','GPU_COST_FIX2.json','OWNER_LAUNCH_FIX2.json'")
s=s.replace("if (OUT/'GPU_COST_FIX1.json').exists():return", "if (OUT/'GPU_COST_FIX2.json').exists():return")
exec(compile(s,str(p),'exec'),globals())

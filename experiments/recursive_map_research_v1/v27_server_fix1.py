"""Reattach to same request/assets after hash compatibility recovery, no new cycle."""
from pathlib import Path
original=Path(__file__).with_name('v27_server.py')
source=original.read_text()
source=source.replace("['MODEL_READY.json','GPU_COST.json','OWNER_LAUNCH.json']",
    "['MODEL_READY.json','GPU_COST.json','OWNER_LAUNCH.json','GPU_COST_FIX1.json','OWNER_LAUNCH_FIX1.json','COMPAT_REPAIR.json','RASTER_PREFLIGHT.json']")
source=source.replace("if (OUT/'GPU_COST.json').exists():return","if (OUT/'GPU_COST_FIX1.json').exists():return")
source=source.replace("['FEEDBACK.json','CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','QUARANTINED.json']",
    "['FEEDBACK.json','CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','QUARANTINED.json','verify_FIX1.log','RETRY_BLOCKED.json']")
exec(compile(source,str(original),'exec'),globals())

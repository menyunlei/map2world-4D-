"""Final same-cycle reconnect, preserve earlier worker/error records."""
from pathlib import Path
p=Path(__file__).with_name('v27_server.py');source=p.read_text()
source=source.replace("['MODEL_READY.json','GPU_COST.json','OWNER_LAUNCH.json']",
 "['MODEL_READY_FINAL.json','MODEL_READY_INITIAL.json','GPU_COST.json','GPU_COST_FIX1.json','GPU_COST_FIX2.json','GPU_COST_FINAL.json','OWNER_LAUNCH_FINAL.json','COMPAT_REPAIR.json','RASTER_PREFLIGHT.json']")
source=source.replace("if (OUT/'GPU_COST.json').exists():return","if (OUT/'GPU_COST_FINAL.json').exists():return")
source=source.replace("state['worker']=worker", "state['worker']=worker\n                    if worker['stage'] not in ('failed','stopped'):state['status']='active' if state['cycle']['status']!='waiting_camera' else 'waiting_camera'")
source=source.replace("['FEEDBACK.json','CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','QUARANTINED.json']",
 "['FEEDBACK.json','CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','QUARANTINED.json','verify_FIX2.log','RETRY_BLOCKED.json']")
exec(compile(source,str(p),'exec'),globals())

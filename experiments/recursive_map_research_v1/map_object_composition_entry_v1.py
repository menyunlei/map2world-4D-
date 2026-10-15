"""Isolated official composition runtime. Does not overwrite existing environments."""
from pathlib import Path
import os,sys,subprocess
JOB=Path(__file__).resolve().parent
DEST=JOB/'map_object_world_v1'
WS=JOB.parents[2]/'WorldStereo/.venv_ws/lib/python3.10/site-packages'
# Resolve from the known repository parent, not a guessed Python parent index.
WS=Path('/hpc2hdd/home/xhe989/Documents/2541/project2/WorldStereo/.venv_ws/lib/python3.10/site-packages')
phase=DEST/'world_composition_v1'
includes=[str(p) for p in (WS/'nvidia').glob('*/include')]
env={**os.environ,'CUDA_HOME':str(JOB/'trellis2_native_quality_v1/environment'),
    'PATH':str(JOB/'trellis2_native_quality_v1/environment/bin')+':'+os.environ['PATH'],
    'CPATH':':'.join(includes),'CPLUS_INCLUDE_PATH':':'.join(includes),
    'TORCH_CUDA_ARCH_LIST':'8.0','MAX_JOBS':'8',
    'TORCH_EXTENSIONS_DIR':str(phase/'extensions'),
    'HF_HOME':'/hpc2hdd/home/xhe989/.cache/huggingface',
    'HF_HUB_CACHE':'/hpc2hdd/home/xhe989/.cache/huggingface/hub'}
subprocess.run([str(DEST/'environment/bin/python'),'-B','-u',str(JOB/'map_object_composition_v1.py'),*sys.argv[1:]],env=env,check=True)

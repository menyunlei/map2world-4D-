"""Preserve the actual WorldStereo venv stack in an additive overlay.

Nested venv --system-site-packages alone inherits the base /opt Python, not
WorldStereo's torch2.7.1 packages. Explicit ordering avoids that false reuse.
"""
from pathlib import Path
import sys
import os

PROJECT=Path('/hpc2hdd/home/xhe989/Documents/2541/project2')
WS_SITE=PROJECT/'WorldStereo/.venv_ws/lib/python3.10/site-packages'
NEW_SITE=Path(__file__).resolve().parent/'map_object_world_v1/environment/lib/python3.10/site-packages'
for p in reversed([WS_SITE,NEW_SITE]):
    while str(p) in sys.path:sys.path.remove(str(p))
    sys.path.insert(0,str(p))
os.environ['PYTHONPATH']=str(WS_SITE)+':'+str(NEW_SITE)

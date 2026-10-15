"""Same fixed views, idle GPU selection and recovered-asset path only."""
from pathlib import Path
p=Path(__file__).with_name('v28_views_fallback.py')
s=p.read_text().replace("s=p.read_text().replace(","s=p.read_text().replace(\"OUT/'native64/world.glb'\",\"OUT/'native64_contiguous/world.glb'\").replace(")
exec(compile(s,str(p),'exec'),globals())

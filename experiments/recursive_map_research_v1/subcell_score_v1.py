"""Reuse fixed narrow verifier and null-safe aggregation; same acceptance thresholds."""
from paths import *
import sys
# The original aggregator reads only prediction outputs; thresholds and private labels stay frozen.
base=(JOB/'narrow_score_v1.py').read_text()
assert "NEW=JOB/'narrow_structure_v1'" in base
base=base.replace("NEW=JOB/'narrow_structure_v1'", "NEW=JOB/'subcell_structure_v1'")
# Replace only null-safe means at the same original aggregation site.
old="summary[label]={k:float(np.mean([r[k] for r in selected])) if selected else None for k in keys}"
assert base.count(old)==1
base=base.replace(old,"summary[label]={k:float(np.mean([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in keys}")
needle='inputs.recheck();old.finite_tree(result);'
assert base.count(needle)==1
base=base.replace(needle,"result['unscorable_narrow_units']=[{'arm':rec['arm'],'seed':rec['seed'],'scene':r['scene'],'generation_seed':r['generation_seed']} for rec in records for r in rec['per_family_rows'] if r['kind']=='challenge' and r['overhead_column_fraction'] is None]; result['means_scope']='defined units only, unscorable units never passes'; "+needle)
ns={'__name__':'__main__','__file__':str(JOB/'narrow_score_v1.py')}
exec(compile(base,str(JOB/'narrow_score_v1.py'),'exec'),ns)

"""Python3.10 hash API compatibility only; frozen geometry queries/gates unchanged."""
from pathlib import Path
original=Path(__file__).with_name('v27_verify.py')
source=original.read_text()
old="    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()"
new="""    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()"""
assert source.count(old)==1
exec(compile(source.replace(old,new),str(original),'exec'),globals())

"""Same native saved assets and cameras with fixed constant lighting, no EXR dependency."""
from pathlib import Path
original=Path(__file__).with_name('trellis2_raw_asset_views_v3.py')
source=original.read_text()
source=source.replace("image=cv2.cvtColor(cv2.imread(str(REPO/'assets/hdri/studio.exr'),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)",
                      "image=np.full((256,512,3),0.8,dtype=np.float32)")
source=source.replace("'native_views'","'native_views_fix1'")
source=source.replace("'HDRI':str(REPO/'assets/hdri/studio.exr')","'lighting':'constant white environment radiance0.8'")
source=source.replace('NATIVE_RELOAD_COMPLETE.json','NATIVE_RELOAD_COMPLETE_FIX1.json')
exec(compile(source,str(original),'exec'))

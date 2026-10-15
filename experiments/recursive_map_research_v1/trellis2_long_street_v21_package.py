"""CPU-only evidence archive and one declared uniform scene scale; no geometry edits."""
from paths import read,save,sha
from trellis2_long_street_v21 import OUT
import json,struct,shutil,tarfile

def metric_export():
    a=read(OUT/'ASSET.json');bounds=a['native_bounds']
    scale=100/max(bounds[1][0]-bounds[0][0],bounds[1][1]-bounds[0][1]);floor=bounds[0][2]
    source=OUT/'native_street_basecolor.glb';dest=OUT/'large_street_100m.glb'
    with source.open('rb') as src,dest.open('xb') as dst:
        magic,version,total=struct.unpack('<III',src.read(12));assert magic==0x46546c67 and version==2
        n,t=struct.unpack('<II',src.read(8));assert t==0x4e4f534a
        doc=json.loads(src.read(n));scene=doc['scenes'][doc.get('scene',0)]
        children=scene['nodes'];new=len(doc['nodes'])
        # glTF Y up. One scene-global transform; unchanged vertex/index/material buffers.
        matrix=[scale,0,0,0,0,scale,0,0,0,0,scale,0,0,-floor*scale,0,1]
        doc['nodes'].append({'name':'assumed_100m_extent_not_surveyed','matrix':matrix,'children':children})
        scene['nodes']=[new]
        text=json.dumps(doc,separators=(',',':')).encode();text+=b' '*((-len(text))%4)
        dst.write(struct.pack('<III',magic,version,total+len(text)-n));dst.write(struct.pack('<II',len(text),t));dst.write(text)
        shutil.copyfileobj(src,dst,8*1024*1024)
    save(OUT/'ASSUMED_SCALE.json',{'global_native_to_meter_scale':scale,'native_floor_for_display':floor,
        'glTF_root_matrix':matrix,'target_longest_XY_extent_m':100,
        'not_real_metric_reconstruction':True,'not_additional_generated_coverage':True,
        'source_geometry':'new model-generated long street, not scaled/repeated v17',
        'vertex_index_and_material_buffers_unchanged':True,'asset':str(dest),'sha256':sha(dest)})
    print('V21_METRIC_ASSET',dest.stat().st_size,sha(dest),flush=True)

def package():
    if not (OUT/'large_street_100m.glb').exists():metric_export()
    archive=OUT/'fixed_evidence.tar.gz'
    with tarfile.open(archive,'x:gz') as pack:
        for p in sorted(OUT.rglob('*')):
            if not p.is_file() or any(x in p.relative_to(OUT).parts for x in ['work','frames','tmp','cache','extensions']):continue
            if p.suffix in ['.json','.png','.jpg','.mp4','.log'] or p.name=='ORTHO_WORLD.npz':pack.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    print('V21_EVIDENCE',archive.stat().st_size,sha(archive),flush=True)

if __name__=='__main__':package()

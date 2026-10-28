"""Freeze one authorized data-coverage comparison; no model calls or new targets."""
from paths import JOB,read,save,sha
from trellis2_map_object_data_v6_fix1 import local_map,write_npz
from trellis2_map_geometry_entry_v2_owner import GPUS
import numpy as np,hashlib,collections,copy

OUT=JOB/'map_object_world_v1/trellis2_coverage_v12'
V8=JOB/'map_object_world_v1/trellis2_structure_learnability_v8'
V6=JOB/'map_object_world_v1/trellis2_map_object_contract_v6'
ARMS=['six_object_rehearsal','bounded_coverage'];SEEDS=[953101,953102]

def footprint(r):
    assert sha(r['map_input'])==r['map_sha256']
    with np.load(r['map_input']) as z:return z['footprints'][int(z['target_object'])]>.5

def prepare():
    assert not OUT.exists(),'Preserve existing experiment; no duplicate launch'
    old=read(V8/'PLAN.json'); manifest=read(V6/'PAIRED_NATIVE_MANIFEST.json')
    train=[r for r in manifest['records'] if r['split']=='adapter_TRAIN']
    held=[r for r in manifest['records'] if r['split']=='adapter_HELDOUT']
    assert len(train)==381 and len(held)==102
    assert not ({r['family_id'] for r in train}&{r['family_id'] for r in held})
    selected=copy.deepcopy(old['selected_records']);seen={r['geometry_key'] for r in selected}
    fps={r['map_input']:footprint(r) for r in train}
    family=collections.Counter(r['family_id'] for r in selected);assets=collections.Counter(r['source_asset_id'] for r in selected)
    # Input-only deterministic selection. Output quality and Map10 are not read.
    for _ in range(18):
        candidates=[r for r in train if r['geometry_key'] not in seen]
        assert candidates,'INSUFFICIENT_PAIRED_TRAIN_DIVERSITY'
        def rank(r):
            fp=fps[r['map_input']]
            distance=min(float(np.logical_xor(fp,fps[s['map_input']]).sum()/max(1,np.logical_or(fp,fps[s['map_input']]).sum())) for s in selected)
            return (family[r['family_id']],assets[r['source_asset_id']],-distance,r['scene_id'],r['object_id'])
        r=min(candidates,key=rank);selected.append(copy.deepcopy(r));seen.add(r['geometry_key'])
        family[r['family_id']]+=1;assets[r['source_asset_id']]+=1
    assert len(selected)==24 and len(seen)==24 and len(assets)>=12,'INSUFFICIENT_PAIRED_TRAIN_DIVERSITY'
    groups=collections.defaultdict(list)
    for r in held:groups[r['family_id']].append(r)
    assert len(groups)==8
    validation=[copy.deepcopy(r) for fam in sorted(groups) for r in groups[fam][:2]]
    assert len(validation)==16
    records=selected+validation
    sources={r['scene_id']:r for r in read(JOB/'map_object_world_v1/trellis2_map_geometry_tokens_v3/PLAN.json')['records']}
    freeze={};audit=[];edits=[]
    for index,r in enumerate(records):
        for p,h in [(r['map_input'],r['map_sha256']),(r['native_target'],r['native_target_sha256']),
                    (r['source_manifest'],r['manifest_sha256'])]:
            assert sha(p)==h;freeze[p]=h
        with np.load(r['native_target']) as z:
            assert z['S0'].shape==(1,8,16,16,16) and np.isfinite(z['S0']).all()
            support=z['support64'][0,0].astype(bool);assert support.any()
        fp=footprint(r);yy,xx=np.where(fp)
        audit.append({'index':index,'id':r['scene_id']+':'+str(r['object_id']),'split':r['split'],
          'family':r['family_id'],'source_asset':r['source_asset_id'],'geometry_key':r['geometry_key'],
          'footprint_extent':[int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
          'footprint_sha256':hashlib.sha256(fp.tobytes()).hexdigest(),'native_cells':int(support.sum()),
          'reference_map_projection_IoU':float((support.any(2).T&fp).sum()/max(1,(support.any(2).T|fp).sum()))})
        src=sources[r['scene_id']];assert sha(src['map_input'])==src['map_sha256'];freeze[src['map_input']]=src['map_sha256']
        with np.load(src['map_input']) as z:m={k:z[k].copy() for k in z.files}
        original,_=local_map(m,r['target_object'],r['frame'])
        with np.load(r['map_input']) as z:
            for k in z.files:assert np.array_equal(original[k],z[k]),('CONTRACT_MISMATCH',index,k)
        j=r['target_object'];yy,xx=np.where(m['footprints'][j]>.5)
        removed=(m['footprints'][j]>.5)&(np.indices((64,64))[1]<xx.min()+.25*(xx.max()-xx.min()+1))
        assert removed.any();m['footprints'][j][removed]=0;m['open_space'][0][removed]=1;m['open_space_known'][0][removed]=1
        edit,_=local_map(m,j,r['frame']);path=OUT/'inputs'/f'edit_{index:03d}.npz';write_npz(path,**edit)
        er={**r,'map_input':str(path),'map_sha256':sha(path)};edits.append(er);freeze[str(path)]=sha(path)
    parent={str(s):read(V8/'small/structure_lora'/f'seed{s}/train/CHECKPOINT_001000.json') for s in SEEDS}
    for c in parent.values():assert sha(c['path'])==c['sha256'];freeze[c['path']]=c['sha256']
    proposal=JOB/'map_object_world_v1/trellis2_map10_block_v11/log_review_v1/BOUNDED_TRAINING_PLAN.json'
    plan={**copy.deepcopy(old),'version':'v12_coverage','arms':ARMS,'parent_checkpoints':parent,
      'selected_records':selected,'six_records':selected[:6],'heldout_records':validation,'evaluation_records':records,'edit_records':edits,
      'small':{'updates':1000,'checkpoints':[100,300,1000],'effective_batch':2,'primary_checkpoint':1000},
      'starting_updates':1000,'optimizer_reset':'fresh identical state across paired arms, not a resumed optimizer',
      'proposal':str(proposal),'proposal_sha256':sha(proposal),'authorization':'user: Start!',
      'only_variable':'six vs24 existing TRAIN object exposure; same adapter+LoRA and native flow objective',
      'selection_algorithm':'minimum selected family count, then asset count, then maximum minimum footprint-XOR/union distance, then scene/object lexicographic',
      'selection_limits':'24 distinct native geometry keys; repeated authored assets/rotations not independent families',
      'generation_seeds':[970201,970202],'shape_address_seed':980201,
      'mesh_indices':[0,1,24,26,28,30,32,34],
      'Shape_condition_checkpoints':old['parent_checkpoints'],
      'GPUs':GPUS[:2],'scheduling':'two matched seed batches; GPU2/3 occupied by unrelated PID2549947',
      'GPU_cap_seconds':10800,'training_seconds_each':1500,'no_automatic_extension':True,
      'no_Map10_generation':True,'no_material_generation':True,'no_mesh_repair':True,
      'milestone_evaluation':[0,100,300,1000],
      'noise_training_base':1971000,'order_seed':971800,
      'map_edit':'west quarter of each selected target, fixed original pose; update open-space; never change target geometry',
      'success':'both training seeds improve paired held-out layout and edit response without increased empty/sheetlike outputs or preservation loss; mesh evidence required',
      'large_conditional':{'authorized':False,'updates':0}}
    save(OUT/'PLAN.json',plan);save(OUT/'SELECTION_AUDIT.json',{'rows':audit,'TRAIN_families':dict(family),'TRAIN_source_assets':dict(assets),'heldout_families':sorted(groups)})
    files=['trellis2_coverage_v12_prepare.py','trellis2_coverage_v12_train.py','trellis2_coverage_v12_owner.py',
      'trellis2_structure_learnability_v8_train.py','trellis2_structure_lora_v8.py','trellis2_map_object_contract_v6.py',
      'trellis2_object_confidence_fix_v7.py','trellis2_map_object_data_v6.py','trellis2_map_object_data_v6_fix1.py',
      'trellis2_map_geometry_token_module_v3.py','trellis2_map_object_module_v1.py','trellis2_map_learning_train_v1.py']
    freeze.update({str(JOB/f):sha(JOB/f) for f in files});freeze[str(OUT/'PLAN.json')]=sha(OUT/'PLAN.json')
    save(OUT/'SOURCE_FREEZE.json',freeze)
    print('COVERAGE_REGISTERED',len(selected),len(validation),len(assets),dict(family),flush=True)

if __name__=='__main__':prepare()

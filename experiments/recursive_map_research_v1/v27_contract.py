"""Freeze one map-content update cycle. CPU preparation, never training."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
JOB=Path(__file__).resolve().parent
OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
REMOTE='/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1'
R=REMOTE+'/map_object_world_v1/trellis2_autonomous_v27'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def exclusive(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def prepare():
    assert not (OUT/'CONTRACT.json').exists(),'Preserve the frozen experiment'
    OUT.mkdir(parents=True,exist_ok=True)
    from v22_map_scaffold import load_inputs
    meta,a,sources,checks=load_inputs()
    scaffold=JOB/'map_object_world_v1/trellis2_streaming_v22'
    for n in ['map10_scaffold.json','MAP10_PROVENANCE.json','map10_reference.png']:shutil.copyfile(scaffold/n,OUT/n)
    prior=JOB/'map_object_world_v1/trellis2_boundary_online_v24/chunks/road-boundary-0--1'
    p=read(prior/'PLAN.json');floor=p['boundary']['ground_native_z'];scale=100.
    frame=[100,0,0,0,0,100,0,0,0,0,100,0,0,-100*floor,0,1]
    roads=(a['roads'][0]>=.75)&(a['roads_known'][0]>=.5)
    assert roads.any(),'No reliable positive road patches'
    c=(np.arange(64)+.5)/64-.5;zi=int(np.argmin(abs(c-floor)))
    ground=np.zeros((64,64,64),bool);ground[:,:,zi]=roads.T
    known=roads.T[:,:,None]&((c>=floor-.03)&(c<=floor+.08))[None,None,:]
    # Registered query locations differ from conditioner cell centers: 4 fixed
    # offsets per reliable road cell. Never select queries using candidate output.
    rc=np.argwhere(roads);offsets=np.array([[-.25,-.25],[-.25,.25],[.25,-.25],[.25,.25]])
    xy=[]
    for row,col in rc:
        for dy,dx in offsets:xy.append([(col+.5+dx)/64-.5,(row+.5+dy)/64-.5])
    np.savez_compressed(OUT/'EVALUATION_QUERIES.npz',road_xy_native=np.array(xy,np.float32),road_cells=rc)
    for i in range(2):
        d=OUT/'attempts'/f'attempt-{i}';d.mkdir(parents=True,exist_ok=True)
        image=JOB/'map_object_world_v1/trellis2_streaming_v22/chunk/native-adjacent-001/input.png'
        assert sha(image)==p['input_sha256'],'Fixed reference differs from successful native pipeline'
        shutil.copyfile(image,d/'input.png')
        np.savez_compressed(d/'ROAD_OBSERVATION.npz',target=ground.astype(np.float32),known=known,ground=ground)
        plan=dict(p)
        plan.update(seed=942271,input=R+f'/attempts/attempt-{i}/input.png',input_sha256=sha(d/'input.png'),
                    chunk_id='map10-0',attempt=i,fixed_chunk_translation_GLTF=[0,0,0],
                    no_map_control=True,map_control_claim=False,no_automatic_extension=True,
                    map_context='Four existing oblique roof proxies and disconnected positive road patches; no surveyed map',
                    matched_control='attempt-0: same native prior, image, noise and fixed frame',
                    intervention='none: native image-conditioned proposal' if i==0 else 'feedback: local map-road observation during Structure sampling only',
                    placement='Fixed Map10 normalized frame before output. No fitting, per-object placement, repair or rotation.',
                    boundary={**p['boundary'],'ground_native_z':floor,'scope':'Map10 existing positive road patches only, not the v21 straight road',
                              'source_parent_asset':None,'source_parent_sha256':None,
                              'fixed_world_frame':{'glTF_root_matrix':frame,'global_native_to_meter_scale':scale,
                              'not_real_metric_reconstruction':True,'native_floor_for_display':floor}})
        exclusive(d/'PLAN.json',plan)
    contract=dict(version='v27',map_id='Map10',persistent_chunk_id='map10-0',
        fixed_seed=942271,max_generations=2,max_feedback_retries=1,model_training=False,
        lease_seconds=720,idle_seconds=240,per_attempt_seconds=180,
        automatic_retry_only_for='Measured positive-road column obstruction or missing road surface; not subjective aesthetics',
        intervention='Native proposal, then at most one reused v24 localized Structure observation, same reference/noise/frame; no global v20 projection',
        query_semantics='Opaque saved-mesh vertical rays; not Gaussian centers, not horizontal collision certification',
        gates={'road_surface_near_display_ground_fraction_min':.95,'road_air_column_obstruction_fraction_max':.05,
               'ground_height_tolerance_display_units':.5,'clearance_height_display_units':8.,
               'floor_lower_tolerance_display_units':.5,'finite_nonempty_geometry_required':True,
               'semantic_building_identity_required':True,'neighbor_compatibility_required_when_neighbor_exists':True},
        unresolved={'roof_polygons':'Visible roof proxies from oblique screenshot, not cadastral footprints',
                    'building_semantic_labels':'Native whole-scene mesh has no verified building/object segmentation',
                    'neighbor_seams':'No previously accepted map-aligned AI neighbor; NOT_TESTED, not a seam pass',
                    'unknowns':'Height, hidden roofs, terrain, untraced roads/free space remain unknown'},
        acceptance='Fail closed on unresolved required map-object correspondence. Diagnostic candidates never auto-install.',
        cache='Accepted SHA manifest immutable; CAS commit; rejected candidate cannot replace accepted world',
        coordinates={'native_to_gltf':'[x,y,z] -> [100*x,100*(z-floor),-100*y]',
                     'floor':floor,'units':'normalized display units, not meters','output_alignment_fit':False},
        map_sources=sources,source_checks=checks,
        geometry_sources={'gray_scaffold':'Existing map annotations, arbitrary display height4, not AI',
                         'candidate_mesh':'TRELLIS.2 native learned geometry and base color, full PBR retained',
                         'retry_road_observation':'Map-defined road floor/air prior, not generated ground truth'},
        reference_conflict='Fixed AI-produced native-street reference may conflict with Map10; not real-site appearance evidence')
    exclusive(OUT/'CONTRACT.json',contract)
    exclusive(OUT/'MODEL_IMPROVEMENT_POLICY.json',dict(enabled=False,training_authorized=False,
        proposed_train_inputs='Legitimate independently paired TRAIN maps/cameras/native3D only; rejected self-generations are not GT',
        validation='Sealed from candidate agents; independent held-out map families, fixed seeds, edits, visual-quality and preservation regressions',
        promotion='Versioned candidate checkpoint; all predeclared joint gates, not best seed; atomic model pointer switch',
        rollback='Retain previous checkpoint and accepted-world hashes; fail or regression -> keep current version',
        prohibited=['automatic weight updates','training on DEV','self-scoring pseudo-GT','changing verifier gates']))
    preserved=JOB/'map_object_world_v1/trellis2_long_street_v21/large_street_100m.glb'
    exclusive(OUT/'BASELINE_CACHE.json',{preserved.relative_to(JOB).as_posix():sha(preserved),
        (scaffold/'map10_scaffold.json').relative_to(JOB).as_posix():sha(scaffold/'map10_scaffold.json')})
    files=[JOB/n for n in ['v27_contract.py','v27_verify.py','v27_worker.py','v27_server.py','v24_boundary_sampler.py','trellis2_native_street_v17.py']]
    files += [OUT/'CONTRACT.json',OUT/'EVALUATION_QUERIES.npz',OUT/'map10_scaffold.json',OUT/'BASELINE_CACHE.json']
    for i in range(2):files += [OUT/'attempts'/f'attempt-{i}'/n for n in ['PLAN.json','input.png','ROAD_OBSERVATION.npz']]
    exclusive(OUT/'SOURCE_FREEZE.json',{str(p.relative_to(JOB)).replace('\\','/'):sha(p) for p in files})
    print(json.dumps({'prepared':True,'roads':int(roads.sum()),'evaluation_queries':len(xy),'max_generations':2}))
if __name__=='__main__':prepare()

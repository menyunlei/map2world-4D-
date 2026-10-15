"""Collect actual cycle evidence. No inference, training, or historical edits."""
from pathlib import Path
import json,hashlib,time
import numpy as np
JOB=Path(__file__).resolve().parent;OUT=JOB/'map_object_world_v1/trellis2_autonomous_v27'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()] if p.exists() else []
def main():
    state=read(OUT/'STATE.json');contract=read(OUT/'CONTRACT.json');events=rows(OUT/'EVENTS.jsonl')
    telemetry=rows(OUT/'BROWSER_TELEMETRY.jsonl');snaps=[x['payload']['snapshot'] for x in telemetry]
    reports=[]
    for i in range(2):
        d=OUT/'attempts'/f'attempt-{i}'
        if (d/'REPORT.json').exists():reports.append(dict(attempt=i,complete=read(d/'COMPLETE.json'),report=read(d/'REPORT.json')))
    accepted=state['cycle'].get('accepted')
    baseline=read(OUT/'BASELINE_CACHE.json');preserved={p:sha(JOB/p)==h for p,h in baseline.items()}
    triggers=[x for x in events if x['event']=='camera_trigger']
    first=triggers[0]['epoch'] if triggers else None
    session=triggers[0]['request']['session'] if triggers else None
    primary=[s for s in snaps if s['session']==session]
    visible=[s for s in primary if s['visible']]
    livefps=[s['fps'] for s in visible if s['fps']>0 and s['world']=='map']
    instruction=dict(ordinary_map_proxies=True,road_condition_during_Structure=True,
        no_model_training=True,same_seed_and_reference_and_frame=True,
        one_native_proposal_plus_at_most_one_feedback_retry=True,
        independent_saved_mesh_verifier=True,quarantine_before_commit=True,
        accepted_asset_count=int(bool(accepted)),new_AI_map_world_success=bool(accepted),
        repeated_old_world_hashes_unchanged=all(preserved.values()))
    tests=dict(camera_trigger_count=len(triggers),duplicate_requests_after_revisit=len(triggers)>1,
        no_installed_rejected_asset=all(s.get('accepted_sha256') in ([accepted['sha256']] if accepted else [None]) for s in snaps),
        baseline_SHA_preservation=preserved,accepted_world_revisit_test='NOT_TESTED_NO_ACCEPTED_AI_ASSET' if not accepted else 'SEE_BROWSER_REVISIT_RECORDS')
    gen=[]
    for x in reports:
        end=x['complete']['completed_epoch'];during=[s for s in visible if first and first<=s['epoch']/1000<=end and s['world']=='map']
        movement=float(max((s['cameraMovementDuringGeneration']['distance'] for s in during),default=0))
        gen.append(dict(attempt=x['attempt'],native_active_seconds=x['complete']['seconds'],
            camera_request_to_native_asset_seconds=end-first if first else None,
            walking_distance_while_generating_display_units=movement,
            road_ground_fraction=x['report']['road_surface_near_display_ground_fraction'],
            road_obstructed_fraction=x['report']['road_air_column_obstruction_fraction'],
            verdict=x['report']['decision'],asset_sha256=x['complete']['sha256']))
    costs=[read(p) for p in sorted(OUT.glob('GPU_COST*.json'))]
    final_cost=(OUT/'GPU_COST_FINAL.json').exists()
    raster=read(OUT/'RASTER_PREFLIGHT.json') if (OUT/'RASTER_PREFLIGHT.json').exists() else None
    result=dict(epoch=time.time(),status=state['cycle']['status'],execution=instruction,tests=tests,
        rendering=dict(map_initial_navigation_ms=min((s['initialNavigationMs'] for s in primary if s.get('initialNavigationMs') is not None),default=None),
            visible_map_FPS_quantiles=np.quantile(livefps,[.1,.5,.9]).tolist() if livefps else None,
            actual_render_sizes=sorted({tuple(s['renderSize']) for s in visible}),
            renderer=primary[-1]['renderer'] if primary else None,not_game_resolution_certification=True),
        generation=gen,physical_GPU_lease_windows=costs,raster_calibration=raster,
        total_GPU_seconds=(sum(x['gpu_seconds'] for x in costs)+raster['gpu_seconds']) if costs and raster else None,
        resident_pipeline_reused_between_samples=(len({x['complete']['resident_pipe_id'] for x in reports})==1) if len(reports)>1 else None,
        recovery_note='Original candidate never regenerated. Hash API/deployment/save-marker failures retained; all reload and recovery GPU time included. Do not report these as warm latency.',
        map_ground_caveat='Display-floor compatibility, not measured terrain truth. Road-surface score at Y=0 is not semantic XY building IoU.',
        remaining_blocker='No verified map-building/object correspondence in native whole-scene mesh; road-only constraints cannot certify a map-aligned block. Rejecting an uncertain result is not map-generation success.',
        conclusion='CONTENT_LOOP_EXECUTED_NO_ACCEPTABLE_MAP_CHUNK' if final_cost and reports and not accepted else 'IN_PROGRESS' if not final_cost else 'CHECK_ACTUAL_VERDICTS',
        next_step='Keep this bounded negative evidence. Learned spatial continuation needs object-addressed geometry with valid paired supervision and independent preservation gates; no automatic weight training.',
        no_automatic_extension=True,no_autonomous_training=True)
    result['visual_checks']=dict(actual_saved_GLBS_independently_reloaded=True,
        saved_mesh_top_and_fixed_Map10_street_lateral_inspected=True,
        observations='Both native street proposals remain visibly inconsistent with orange Map10 roof proxies. Retry has large foliage sheets near the fixed camera and does not restore object correspondence.',
        semantic_building_metric='UNRESOLVED, not a height-proxy score',human_review='PENDING_NOT_REQUIRED_FOR_THIS_BOUNDED_TEST')
    result['query_counts_are_not_independent_world_samples']=True
    result['independent_world_trials']=1
    # This is a live results snapshot, not a replacement of historical verdicts.
    target=OUT/'RESULTS.json';tmp=target.with_suffix('.next');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(target)
    if final_cost:
        evidence=dict(result_sha256=sha(target),contract_sha256=sha(OUT/'CONTRACT.json'),
            assets=[dict(attempt=x['attempt'],path=f"attempts/attempt-{x['attempt']}/world.glb",sha256=x['complete']['sha256'],
                         bytes=(OUT/f"attempts/attempt-{x['attempt']}/world.glb").stat().st_size,accepted=False) for x in reports],
            actual_code={n:sha(JOB/n) for n in ['v27_worker.py','v27_finish.py','v27_verify.py','v27_verify_fix1.py','v27_server.py','v27_live.py','v24_boundary_sampler.py','trellis2_native_street_v17.py','v26_resident.py']},
            visual_client={n:sha(OUT/n) for n in ['app.js','index.html','style.css','mesh_worker.js']},
            replay_viewer='python -B experiments/recursive_map_research_v1/v27_live.py',
            no_more_inference='Worker budget exhausted; attempt markers prevent duplicate inference; viewer loads saved quarantined meshes only',
            native_raw_assets_retained_on_server=True,AI_model_weight_updates=0)
        tmp=OUT/'ASSET_MANIFEST.next';tmp.write_text(json.dumps(evidence,indent=2),encoding='utf8');tmp.replace(OUT/'ASSET_MANIFEST.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

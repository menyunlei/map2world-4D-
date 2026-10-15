"""CPU aggregation of all fixed cases; never treats rays/objects as families."""
from paths import read,save
from trellis2_coverage_v12_prepare import OUT,SEEDS,ARMS
import collections,json,time

def mean(rows,key):
    values=[r[key] for r in rows if isinstance(r.get(key),(int,float,bool))]
    return sum(values)/len(values) if values else None

def main():
    summaries=[];per_family=[];raw=[];exposure=[];costs=[];missing=[]
    for p in (OUT/'evaluation').glob('seed*/*/update*/RESULTS.json'):
        result=read(p);raw.append(str(p));seed=result['seed'];arm=result['arm'];update=result['additional_updates']
        for group in ['original_six','added_TRAIN','heldout']:
            rows=[r for r in result['rows'] if r['mode']=='correct' and r['group']==group]
            edits=[r for r in result['edit_rows'] if r['group']==group]
            family_rows=[]
            for family in sorted({r['family'] for r in rows}):
                rr=[r for r in rows if r['family']==family];er=[r for r in edits if r['family']==family]
                row={'seed':seed,'arm':arm,'update':update,'group':group,'family':family,
                  'object_instances':len({r['index'] for r in rr}),'generation_seeds':sorted({r['noise'] for r in rr}),
                  'map_IoU':mean(rr,'map_projection_IoU'),'footprint_coverage':mean(rr,'required_footprint_coverage'),
                  'outside_fraction':mean(rr,'outside_footprint_fraction'),'empty_fraction':mean(rr,'empty'),
                  'thin_axis_count_mean':mean(rr,'axes_below_three_cells'),
                  'exact_native_target_IoU_diagnostic':mean(rr,'native_target_IoU_diagnostic'),
                  'removed_occupancy_after':mean(er,'removed_occupancy_after'),
                  'retained_coverage_after':mean(er,'retained_coverage_after'),
                  'protected_Q_change_fraction':mean(er,'protected_Q_change_fraction')}
                per_family.append(row);family_rows.append(row)
            summaries.append({'seed':seed,'arm':arm,'update':update,'group':group,'families':len(family_rows),
              **{k:mean(family_rows,k) for k in ['map_IoU','footprint_coverage','outside_fraction','empty_fraction',
                  'thin_axis_count_mean','exact_native_target_IoU_diagnostic','removed_occupancy_after',
                  'retained_coverage_after','protected_Q_change_fraction']}})
    for seed in SEEDS:
        for arm in ARMS:
            p=OUT/'small'/arm/f'seed{seed}/train/TRAIN_COMPLETE.json'
            if not p.exists():missing.append(str(p));continue
            start=read(p);counts=collections.Counter();conditional=collections.Counter();times=[]
            with (p.parent/'train_log.jsonl').open() as handle:
                for line in handle:
                    row=json.loads(line);times.append(row['sec_update'])
                    for slot in row['slots']:
                        counts[slot['object']]+=1;conditional[slot['object']]+=not slot['drop']
            exposure.append({'seed':seed,'arm':arm,'updates':start['actual_updates'],'slots':sum(counts.values()),
              'per_object':dict(counts),'conditioned_per_object':dict(conditional),
              'seconds_per_update_mean':sum(times)/max(1,len(times))})
    for p in (OUT/'gpu_cost').glob('*.json'):costs.append(read(p))
    decisions=[]
    for seed in SEEDS:
        def get(arm,update,group):return next((x for x in summaries if (x['seed'],x['arm'],x['update'],x['group'])==(seed,arm,update,group)),None)
        a=get(ARMS[0],1000,'heldout');b=get(ARMS[1],1000,'heldout')
        if not a or not b:decisions.append({'seed':seed,'status':'FINAL_STRUCTURE_PENDING'});continue
        checks={'heldout_layout_gain':b['map_IoU']>a['map_IoU'],
          'heldout_removed_occupancy_reduction':b['removed_occupancy_after']<a['removed_occupancy_after'],
          'retained_coverage_not_lower':b['retained_coverage_after']>=a['retained_coverage_after'],
          'empty_fraction_not_higher':b['empty_fraction']<=a['empty_fraction'],
          'thin_axis_count_not_higher':b['thin_axis_count_mean']<=a['thin_axis_count_mean']}
        decisions.append({'seed':seed,'predeclared_numeric_directions':checks,'all_directions':all(checks.values()),
          'not_mesh_acceptance':True,'map_IoU_delta':b['map_IoU']-a['map_IoU'],
          'removed_occupancy_delta':b['removed_occupancy_after']-a['removed_occupancy_after'],
          'retained_coverage_delta':b['retained_coverage_after']-a['retained_coverage_after']})
    payload={'created_epoch':time.time(),'summaries_family_balanced':summaries,'per_family':per_family,
      'seed_decisions':decisions,'training_exposure':exposure,'GPU_cost':costs,
      'GPU_seconds_including_loading':sum(r['gpu_seconds'] for r in costs),
      'source_results':raw,'missing_training':missing,
      'mesh_render_completion':{str(seed):(OUT/'evaluation'/f'seed{seed}/RENDER_COMPLETE.json').exists() for seed in SEEDS},
      'status':'NUMERICAL_REPORT_NOT_VISUAL_ACCEPTANCE','AI_visual_review_pending':True,'human_review_not_claimed':True,
      'no_new_Map10':'v11 remains failed; this is a bounded synthetic paired-data coverage test',
      'limitations':['Repeated source assets are not independent geometry priors.',
        'Protected target support is measured; whole-neighbor regenerated mesh preservation is not measured.',
        'No solid-volume or closed-surface success follows from extent or footprint IoU alone.'],
      'no_more_training_authorized':True,'no_automatic_budget_extension':True}
    path=OUT/'summary'/f'RESULTS_{time.time_ns()}.json';save(path,payload)
    print('SUMMARY',path,flush=True)
    for row in decisions:print(row,flush=True)

if __name__=='__main__':main()

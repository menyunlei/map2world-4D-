"""Independent native-support verifier. No candidate imports, inference, or repair.

Exact registered target fidelity is NOT synonymous with map observability.
Unknown heights are evaluated only as paired-reference fidelity. Surface support
is NOT occupancy. Gaussian/visual evidence remains a separate mandatory gate.
"""
import numpy as np

GRID = 64
CELL_METRES = .5
ORIGIN_CENTRE = np.array([-15.75, -15.75, -1.75])
THRESHOLDS = {'legacy_non_ground_IoU': .8586968193727308,
              'closed_closure_recall': .3973420414807664,
              'required_surface_recall': .5510684171003499}
NG = np.broadcast_to((ORIGIN_CENTRE[2]+CELL_METRES*np.arange(GRID))[None,None,:]>.5, (GRID,GRID,GRID))
FAMILIES = tuple(f'courtyard_new_{n:03d}' for n in (1,2,5,6,18,19,21,31))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def dense(coords):
    q = np.asarray(coords)
    require(q.ndim == 2 and q.shape[1] in (3,4), 'coordinate shape')
    if q.shape[1] == 4:
        require((q[:,0] == 0).all(), 'single scene only')
    q = q[:,-3:]
    require(np.isfinite(q).all() and np.equal(q,np.floor(q)).all(), 'noninteger/nonfinite coordinates')
    require(((q>=0)&(q<GRID)).all(), 'coordinates outside grid')
    q = q.astype(np.int64)
    out = np.zeros((GRID,GRID,GRID),dtype=bool)
    out[tuple(q.T)] = True
    return out

def fraction(numerator,denominator):
    return float(numerator/denominator) if denominator else None

def recall(prediction, region):
    return fraction(np.count_nonzero(prediction&region),np.count_nonzero(region))

def iou(prediction, target, region):
    return fraction(np.count_nonzero(prediction&target&region),np.count_nonzero((prediction|target)&region))

def score(prediction, target, masks, variant):
    p,t=np.asarray(prediction,dtype=bool),np.asarray(target,dtype=bool)
    require(p.shape==t.shape==(GRID,GRID,GRID), 'support shape')
    m={k:np.asarray(v,dtype=bool) for k,v in masks.items()}
    for key in ['free_air','overhead','required_surface','closed_closure','roi','protected','non_ground','unknown_geometry']:
        require(m[key].shape==p.shape, (key,m[key].shape))
    require(m['eroded_xy'].shape==(GRID,GRID), 'core shape')
    require(not (t&m['free_air']).any(), 'reference occupies certified free air')
    count=lambda v:int(np.count_nonzero(v))
    diff=p^t
    result={
        'predicted_Q_cells':count(p),'target_Q_cells':count(t),
        'free_air_wrong_cells':count(p&m['free_air']), 'free_air_total_cells':count(m['free_air']),
        'free_air_wrong_fraction':recall(p,m['free_air']),
        'overhead_wrong_cells':count(p&m['overhead']), 'overhead_total_cells':count(m['overhead']),
        'overhead_column_count':count((p&m['overhead']).any(2)), 'core_XY_columns':count(m['eroded_xy']),
        'overhead_column_fraction':fraction(count((p&m['overhead']).any(2)),count(m['eroded_xy'])),
        'required_surface_cells':count(m['required_surface']), 'required_surface_recalled':count(p&m['required_surface']),
        'required_surface_recall':recall(p,m['required_surface']),
        'closed_closure_cells':count(m['closed_closure']),
        'closed_closure_recall':recall(p,m['closed_closure']) if variant=='closed' else None,
        'overall_native_IoU':iou(p,t,np.ones_like(p)), 'non_ground_native_IoU':iou(p,t,m['non_ground']),
        'target_region_symmetric_difference_cells':count(diff&m['roi']), 'target_region_total_cells':count(m['roi']),
        'target_region_symmetric_difference_fraction':recall(diff,m['roi']),
        'protected_region_symmetric_difference_cells':count(diff&m['protected']), 'protected_region_total_cells':count(m['protected']),
        'protected_region_symmetric_difference_fraction':recall(diff,m['protected']),
        'surface_support_not_solid_occupancy':True, 'unknown_geometry_cells':count(m['unknown_geometry']),
        'empty_eroded_core_not_scored_success':bool(variant!='closed' and not m['eroded_xy'].any()),
        'non_ground_precision':fraction(count(p&t&NG),count(p&NG)),
        'non_ground_recall':recall(p,t&NG), 'non_ground_support_ratio':fraction(count(p&NG),count(t&NG)),
        'unknown_reference_IoU':iou(p,t,m['unknown_geometry']),
        'map_horizontal_support_IoU':iou((p&NG).any(2),(t&NG).any(2),np.ones((GRID,GRID),bool)),
    }
    return result

def roof_diagnostic(prediction, closed_mask):
    xyz=np.argwhere(closed_mask); require(len(xyz)>0, 'empty closed mask')
    z=np.arange(GRID)
    columns=(prediction&NG)[xyz[:,0],xyz[:,1],:]
    dz=np.where(columns,z[None,:]-xyz[:,2:3],10000)
    nearest=dz[np.arange(len(xyz)),np.abs(dz).argmin(1)]
    exists=columns.any(1)
    return {'exact_recall':recall(prediction,closed_mask),
        'same_xy_any_non_ground_fraction':float(exists.mean()),
        'within_0p5m_fraction':float((np.abs(nearest)<=1).mean()),
        'within_1m_fraction':float((np.abs(nearest)<=2).mean()),
        'signed_nearest_offset_median_m':float(np.median(nearest[exists])*.5) if exists.any() else None,
        'abs_nearest_offset_median_m':float(np.median(np.abs(nearest[exists]))*.5) if exists.any() else None,
        'missing_entire_column_fraction':float((~exists).mean()),
        'height_observable_from_map':False,'tolerance_does_not_relax_exact_gate':True}

def paired(closed,opened,truth_closed,truth_open,shared,protected,free_open,noise_equal):
    require(isinstance(noise_equal,(bool,np.bool_)) and bool(noise_equal), 'paired intervention changed generation noise')
    require(shared.any(), 'uncertified/empty shared wall mask')
    actual=closed^opened; expected=truth_closed^truth_open
    removal=truth_closed&free_open&~truth_open
    return {'shared_closed_recall':recall(closed,shared),'shared_open_recall':recall(opened,shared),
        'shared_both_recalled':recall(closed&opened,shared),
        'intended_change_recall':recall(actual,expected),
        'intended_change_precision':fraction(np.count_nonzero(actual&expected),np.count_nonzero(actual)),
        'directionally_correct_changed_fraction':recall(closed&~opened,removal),
        'protected_region_change_fraction':recall(actual,protected),
        'unrelated_non_ground_change_fraction':recall(actual,(~expected)&NG),
        'initial_noise_exactly_equal':True}

def native_gate(summary, baseline, per_family, baseline_family, seed_gains):
    require(isinstance(per_family,list), 'family records must be a materialized list')
    ids=[x['family_id'] for x in per_family]
    complete=len(ids)==8 and len(set(ids))==8 and set(ids)==set(FAMILIES)==set(baseline_family)
    values=[summary[k] for k in THRESHOLDS]+[summary[k] for k in ['free_air_wrong_fraction','overhead_column_fraction']]
    values += [baseline[k] for k in ['free_air_wrong_fraction','overhead_column_fraction']]
    values += [x['courtyard_error'] for x in per_family]+[x['courtyard_error'] for x in baseline_family.values()]
    require(all(np.isfinite(x) and 0<=x<=1 for x in values), 'invalid metric range/nonfinite')
    require(all(np.isfinite(x) and -1<=x<=1 for x in seed_gains), 'invalid seed gains')
    margins={k:summary[k]-v for k,v in THRESHOLDS.items()}
    gains={k:baseline[k]-summary[k] for k in ['free_air_wrong_fraction','overhead_column_fraction']}
    improved=sum(f['courtyard_error']<baseline_family[f['family_id']]['courtyard_error'] for f in per_family)
    acquisition=complete and improved>=6 and all(v>0 for v in gains.values()) and len(seed_gains)==2 and all(v>0 for v in seed_gains)
    return {'margins':margins,'component_gains':gains,'better_families':improved,
            'generation_seed_gains':seed_gains,'acquisition_pass':bool(acquisition),
            'preservation_pass':bool(all(v>=0 for v in margins.values())),
            'native_Q_joint_pass':bool(acquisition and all(v>=0 for v in margins.values()))}

def final_gate(native_pass,decoded_integrity,visual_ratings,heldout_pass,all_training_seeds):
    """No empty evidence, numerical-only promotion or UNCERTAIN-as-PASS."""
    required=['opening','enclosing_walls','closed_roof','shared_walls','unrelated_geometry','decoded_surface_quality']
    if not all(isinstance(v,(bool,np.bool_)) and bool(v) for v in [native_pass,decoded_integrity,heldout_pass,all_training_seeds]):
        return False
    reviewed=isinstance(visual_ratings,list) and len(visual_ratings)>0 and all(isinstance(v,dict) and all(v.get(k)=='PASS' for k in required) for v in visual_ratings)
    return bool(native_pass and decoded_integrity and reviewed and heldout_pass and all_training_seeds)

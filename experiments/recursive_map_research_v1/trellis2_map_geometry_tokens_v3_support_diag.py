"""Read saved support only to locate failure; not a new final-geometry gate."""
from paths import read,save,sha
from trellis2_map_geometry_tokens_v3_prepare import OUT
import numpy as np
from PIL import Image

if __name__=='__main__':
    rows=[]
    for arm in ['original','separated']:
        for seed in [953101,953102]:
            for mode,name in [('correct','map.npz'),('edit','explicit_remove_last_building_map.npz'),('shuffled','shuffled_map.npz')]:
                if arm=='original' and mode=='shuffled':continue
                directory=OUT/arm/f'seed{seed}/assets'/mode
                path=directory/'SS_GENERATED.npz'
                if not path.exists():continue
                with np.load(path) as trace:
                    support=trace['native_support64'][0,0].astype(bool)
                assert support.shape==(64,64,64)
                with np.load(OUT/'real_input'/name) as m:
                    footprints=m['footprints']
                    valid=m['object_valid']
                # Canonical TRAIN floor is -.4375, index4 at native support64.
                # This is a support diagnostic, NOT mesh/volume visibility.
                above=support[:,:,6:].any(-1).T
                union=footprints[valid].max(0)>0 if valid.any() else np.zeros((64,64),bool)
                overlap=(above&union).sum()/max(1,(above|union).sum())
                per_object=[float(above[footprints[i]>0].mean()) for i in np.flatnonzero(valid)]
                colors=np.zeros((64,64,3),np.uint8)
                colors[above]=[155,155,155]
                palette=[[255,95,100],[90,220,120],[80,150,255],[230,140,250]]
                for i in np.flatnonzero(valid):
                    mask=footprints[i]>0
                    colors[mask]=palette[i%len(palette)]
                    colors[mask&above]=np.clip(np.array(palette[i%len(palette)])*.55+110,0,255)
                Image.fromarray(colors[::-1].copy()).resize((512,512),Image.Resampling.NEAREST).save(directory/'SUPPORT_MAP_XY_DIAGNOSTIC.png')
                rows.append({'arm':arm,'seed':seed,'mode':mode,'native_support_cells':int(support.sum()),
                    'support_above_canonical_floor_XY_cells':int(above.sum()),
                    'input_roof_proxy_XY_cells':int(union.sum()),
                    'above_floor_support_proxy_overlap_DIAGNOSTIC':float(overlap),
                    'per_object_support_presence_DIAGNOSTIC':per_object,
                    'canonical_floor_layer4_support_fraction_DIAGNOSTIC':float(support[:,:,4].mean()),
                    'support_layer_Z_histogram':support.sum((0,1)).tolist(),
                    'not_solid_occupancy_or_final_Gaussian_or_mesh_measurement':True,
                    'input_not_geographically_registered':True,'source_sha256':sha(path)})
    save(OUT/'SAVED_SUPPORT_FAILURE_LOCALIZATION.json',{'records':rows,
         'model_updates':0,'model_resampling':False,'diagnostic_only':True,
         'historical_thresholds_unchanged':True})
    print('SAVED_SUPPORT_FAILURE_LOCALIZATION',len(rows),flush=True)

"""Reuse frozen DreamX/data semantics through TRAIN-only interfaces."""
from paths import *
import collections,sys,random
for folder in [C4/'code',HIER/'code',ROOT/'experiments/dreamx_injection_readout_topofix_v1/code',
               ROOT/'experiments/dreamx_injection_readout_screen_v1/code',ROOT/'experiments/dreamx_courtyard_preservation_v1/code',
               D2/'code',NATIVE/'sources',NATIVE/'TRELLIS',ROOT]:
    sys.path.append(str(folder))
import numpy as np
import torch
from train_map_native_prior import PairedData,per_scene_flow,decoded_dice,structure_decoder
from hierarchy_compile_v2 import compile_relation,empty_relation
from relation_compile_v1 import relation_to_device
from objective_runtime import Runtime as AuxiliaryRuntime,decode,terms

class TrainData(PairedData):
    def __init__(self,stage):
        self.stage='structure' if stage=='structure' else 'slat'
        c=read(JOB/'TRAIN_DATA.json');self.lookup=c['records'];self.train=[self.lookup[s] for s in c['original_order']]
        self.new={s:r for s,r in self.lookup.items() if r['split']=='SUPPLEMENTARY_TRAIN'}
        self.cache=collections.OrderedDict();self.orders={};self.text_cache={}
        assert sha(c['normalization_path'])==c['normalization_sha256']
        self.normalization=read(c['normalization_path']);self.mean=torch.tensor(self.normalization['mean']);self.std=torch.tensor(self.normalization['std'])
        assert sha(c['empty']['path'])==c['empty']['sha256']
        self.empty=torch.load(c['empty']['path'],map_location='cpu',weights_only=True)['context']
    def read(self,record):
        sid=record['scene_id'];assert sid in self.lookup and record['split'] in ['train','SUPPLEMENTARY_TRAIN']
        if sid in self.cache:self.cache.move_to_end(sid);return self.cache[sid]
        r=self.lookup[sid]
        if self.stage=='structure':
            z=torch.load(r['structure'],map_location='cpu',weights_only=True)
            result={'clean':z['S0'][0],'surface':z['surface_active'][0]}
        else:
            assert r['detail'] is not None,'No legitimate paired Detail target'
            with np.load(r['detail']) as z:result={'clean':(torch.from_numpy(z['L0'].copy())-self.mean)/self.std,'coords':torch.from_numpy(z['coords'].copy()).int()}
        text=r['text'];key=text['sha256']
        if key not in self.text_cache:
            assert sha(text['path'])==key
            self.text_cache[key]=torch.load(text['path'],map_location='cpu',weights_only=True)['context']
        with np.load(r['map_path']) as z:raster=torch.from_numpy(z['map_tensor'].copy()).float()
        result.update(text=self.text_cache[key],map=raster)
        self.cache[sid]=result
        if len(self.cache)>128:self.cache.popitem(last=False)
        return result

def auxiliary(data):
    # Reuse the audited TRAIN-only O1 labels and exact equation, not its old
    # dataset constructor (which exposes an unfiltered dataset manifest).
    r=AuxiliaryRuntime.__new__(AuxiliaryRuntime);r.data=data;r.lookup=data.lookup
    r.audit=read(C4/'SUPERVISION_LABEL_AUDIT.json');assert r.audit['status']=='PASS'
    r.groups=read(C4/'AUXILIARY_GROUP_COUNTS.json')['groups'];r.topologies={};r.masks={}
    r.zero=relation_to_device(empty_relation(),'cuda')
    return r

def initialize(candidate,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if candidate=='g1_a':
        sys.path.insert(0,str(JOB/'candidates/g1_a'))
        from column_model import ColumnSceneFlow
        model=ColumnSceneFlow(device='cuda');stage='structure'
    elif candidate=='g1_b':
        from detail_model import SupportSceneFlow
        model=SupportSceneFlow(device='cuda');stage='detail'
    elif candidate=='g2_a':
        sys.path.insert(0,str(JOB/'candidates/g2_a'))
        from profile_model import ProfileSceneFlow
        model=ProfileSceneFlow(device='cuda');stage='structure'
    else:raise ValueError('candidate not predeclared')
    fixed={'structure':'39e0e496721327ec6120473fd09a496bdb688f2723b6a51dddcb3b8df44797a4',
           'detail':'3ff81d0518c74616a9ee82241831258e2518880dc91d912dee14c815e87f1fb6'}
    meta=read(D2/f'D2_local_map/{stage}/CHECKPOINT_005000.json');assert sha(meta['path'])==meta['sha256']==fixed[stage]
    saved=torch.load(meta['path'],map_location='cpu',weights_only=False)
    current=dict(model.named_parameters());parent={k:current[k] for k in saved['trainable_state']}
    with torch.no_grad():
        for name,p in parent.items():
            assert p.shape==saved['trainable_state'][name].shape
            p.copy_(saved['trainable_state'][name])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('branch.'))
    branch={k:p for k,p in model.named_parameters() if p.requires_grad}
    restore=saved['rng_by_rank'][0];del saved
    assert branch and all(not p.requires_grad for p in parent.values())
    return model,branch,meta,restore

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state()}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state(state['torch_cuda'])

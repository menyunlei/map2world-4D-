"""Reuse the exact v8 trainer; only object coverage and registered parent differ."""
import trellis2_structure_learnability_v8_train as train
from trellis2_coverage_v12_prepare import OUT,ARMS,SEEDS
from paths import read,save,sha
import torch

original_setup=train.setup
train.OUT=OUT;train.ARMS=ARMS;train.SEEDS=SEEDS

def setup(arm,seed):
    plan,native,branch,reader,params=original_setup('structure_lora',seed)
    parent=plan['parent_checkpoints'][str(seed)]
    state=torch.load(parent['path'],map_location='cpu',weights_only=False)
    train.load_lora_state_dict(native,state['lora_state'])
    assert len(state['lora_state'])==18 and sum(p.numel() for p in params)==857344
    assert state['updates']==1000 and state['arm']=='structure_lora'
    plan={**plan,'selected_records':plan['six_records'] if arm==ARMS[0] else plan['selected_records']}
    assert len(plan['selected_records']) in [6,24]
    return plan,native,branch,reader,params

train.setup=setup
original_step=train.step
def step(record,key,components,probe=False):
    # Independent continuation noise; matched between arms at each slot.
    return original_step(record,key+1000000,components,probe)
train.step=step

if __name__=='__main__':train.main()

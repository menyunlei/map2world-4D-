"""G1-B adds geometry communication inside native Detail, never after decoding."""
import sys
from paths import JOB
sys.path.insert(0,str(JOB/'candidates/g1_b'))
from support_attention import SupportBranch
import dreamx_scene_flow as dsf
import torch
from torch.utils.checkpoint import checkpoint

class SupportSceneFlow(dsf.DreamXSceneFlow):
    def __init__(self,device='cuda'):
        super().__init__('detail','D2',device=device,activation_checkpointing=True)
        self.requires_grad_(False);self.branch=SupportBranch().to(device)
    def _flow(self,tokens,rotary_zyx,native_xyz,timestep,text,map_tensor):
        h=self.adapter.encode(tokens)
        with torch.autocast('cuda',enabled=False):
            e=self.native.time_embedding(dsf.implementation.sinusoidal_embedding_1d(256,timestep.float()).float())
            e0=self.native.time_projection(e).unflatten(1,(6,3072))
        context=self.native.text_embedding(text)
        prepared=self.map.prepare(map_tensor,native_xyz);support=self.branch.prepare(native_xyz,map_tensor)
        length=torch.tensor([tokens.shape[1]],device=tokens.device,dtype=torch.int32)
        grid=torch.tensor([[1,1,tokens.shape[1]]])
        for i,block in enumerate(self.native.blocks):
            def run(hidden,block=block,i=i):
                handle=dsf.POSITIONS.set(rotary_zyx)
                try:
                    output=block(hidden,e0,length,grid,self.native.freqs,context,None,dtype=torch.bfloat16,cam_emb=None)
                    if i in dsf.SITES:
                        site=dsf.SITES.index(i);output=self.map.read(output,prepared,site)
                        output=output+self.branch.read(output,support,site)
                    return output
                finally:dsf.POSITIONS.reset(handle)
            h=checkpoint(run,h,use_reentrant=False) if torch.is_grad_enabled() else run(h)
        return self.adapter.decode(h).float()

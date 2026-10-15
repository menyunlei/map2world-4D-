"""Distribute the existing free-space gradient budget across occupied heights."""
import torch
from trellis2_mapguided_v18_guidance import MapGuidedSampler

class VolumeSampler(MapGuidedSampler):
    def energy(self,y,origin):
        logits=self.decoder(y).float()[0,0,:,:,self.config['building_z_start']:]
        maximum=logits.amax(-1)
        positive=torch.stack([torch.nn.functional.softplus(-maximum[m]).mean() for m in self.masks]).mean()
        # Stop-gradient scale: each column has exactly the previous logit-gradient
        # L1 norm sigmoid(max)/outside_count. Only its allocation over Z changes.
        # This is a surrogate gradient; its scalar loss is not comparable to v18.
        with torch.no_grad():
            scale=maximum.sigmoid()/logits.sigmoid().sum(-1).clamp_min(1e-12)
        negative=(torch.nn.functional.softplus(logits).sum(-1)*scale)[self.outside].mean()
        prior=(y-origin).square().mean()
        return positive+negative+self.config['prior_weight']*prior,positive,negative

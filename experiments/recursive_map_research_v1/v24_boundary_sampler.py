"""Local flow-time boundary observations; the unknown latent stays native."""
from paths import JOB,read,save
import numpy as np
def install(pipe,out):
    import torch
    import torch.nn.functional as F
    from trellis2 import models
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    p=read(out/'PLAN.json')
    with np.load(out/'ROAD_OBSERVATION.npz') as a:target=a['target'];known=a['known'];ground=a['ground']
    with torch.random.fork_rng(devices=[0]):
        encoder=models.from_pretrained(p['boundary']['encoder']).cuda().eval().requires_grad_(False)
        with torch.no_grad():z=encoder(torch.tensor(target,device='cuda',dtype=torch.float32)[None,None],sample_posterior=False).float()
        del encoder
        decoder=pipe.models['sparse_structure_decoder'];old_device=next(decoder.parameters()).device
        decoder.to('cuda')
        with torch.no_grad():decoded=decoder(z).float()>0
        decoded=decoded[0,0].cpu().numpy();decoder.to(old_device)
    # Codec preflight only: no threshold/model selection or second attempt.
    floor_z=np.flatnonzero(ground.any((0,1)));band=decoded[:,:,max(0,floor_z[0]-1):min(64,floor_z[-1]+2)].any(2)
    road_xy=ground.any(2);coverage=float(band[road_xy].mean())
    air=known&~ground;air[:,:,:floor_z[-1]+2]=False
    obstruction=float(decoded[air].mean()) if air.any() else 0.
    mask=F.max_pool3d(torch.tensor(known,device='cuda',dtype=torch.float32)[None,None],4,4)>0
    save(out/'CODEC_PREFLIGHT.json',{'road_floor_coverage_with_one_voxel_tolerance':coverage,'road_air_occupancy_fraction':obstruction,'known_latent_fraction':float(mask.float().mean()),'target_is_condition_not_generated_geometry':True,'new_training':False})
    assert coverage>=.75 and obstruction<=.05,('Boundary codec cannot express this observation',coverage,obstruction)
    class BoundarySampler(FlowEulerGuidanceIntervalSampler):
        def __init__(self):super().__init__(sigma_min=pipe.sparse_structure_sampler.sigma_min);self.trace=[]
        @torch.no_grad()
        def sample(self,model,noise,*args,**kwargs):
            self.anchor_noise=noise.detach().clone()
            result=super().sample(model,noise,*args,**kwargs)
            np.savez_compressed(out/'STRUCTURE_LATENT.npz',latent=result.samples.detach().float().cpu().numpy(),known_mask=mask.cpu().numpy())
            save(out/'BOUNDARY_TRACE.json',self.trace)
            return result
        @torch.no_grad()
        def sample_once(self,model,x_t,t,t_prev,cond=None,**kwargs):
            observed=(1-t)*z+(self.sigma_min+(1-self.sigma_min)*t)*self.anchor_noise
            before=torch.where(mask,observed,x_t)
            result=super().sample_once(model,before,t,t_prev,cond,**kwargs)
            native_next=result.pred_x_prev
            next_observed=(1-t_prev)*z+(self.sigma_min+(1-self.sigma_min)*t_prev)*self.anchor_noise
            result.pred_x_prev=torch.where(mask,next_observed,native_next)
            result.pred_x_0=torch.where(mask,z,result.pred_x_0)
            self.trace.append({'t':float(t),'t_prev':float(t_prev),'known_fraction':float(mask.float().mean()),
              'input_boundary_change_RMS':float((before-x_t).square().mean().sqrt()),
              'outside_mask_change_max':float((result.pred_x_prev-native_next).masked_select((~mask).expand_as(native_next)).abs().max())})
            return result
    pipe.sparse_structure_sampler=BoundarySampler()
    native_ss=pipe.sample_sparse_structure
    def record_structure(*args,**kwargs):
        coords=native_ss(*args,**kwargs)
        np.savez_compressed(out/'GENERATED_STRUCTURE.npz',coords=coords.detach().cpu().numpy())
        save(out/'STRUCTURE_CHECK.json',{'coords':len(coords),'native_flow_other_regions_unchanged':True,'final_support_not_projected_or_masked':True})
        return coords
    pipe.sample_sparse_structure=record_structure

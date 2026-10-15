"""Map energy acts on predicted-clean SS latents during native Euler sampling."""
import torch,numpy as np
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

class MapGuidedSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self,decoder,fps,config,enabled=True,**kwargs):
        super().__init__(**kwargs);self.decoder=decoder;self.config=config;self.enabled=enabled;self.trace=[]
        self.masks=torch.from_numpy(np.asarray(fps).transpose(0,2,1).copy()).bool().cuda()
        self.outside=~self.masks.any(0)
    def energy(self,y,origin):
        logits=self.decoder(y).float()[0,0,:,:,self.config['building_z_start']:].amax(-1)
        positive=torch.stack([torch.nn.functional.softplus(-logits[m]).mean() for m in self.masks]).mean()
        negative=torch.nn.functional.softplus(logits[self.outside]).mean()
        prior=(y-origin).square().mean()
        return positive+negative+self.config['prior_weight']*prior,positive,negative
    @torch.no_grad()
    def sample_once(self,model,x_t,t,t_prev,cond=None,**kwargs):
        out=super().sample_once(model,x_t,t,t_prev,cond,**kwargs)
        if not self.enabled or t>self.config['active_t_max']:return out
        origin=out.pred_x_0.detach().float()
        with torch.enable_grad():
            y=origin.clone().requires_grad_(True);opt=torch.optim.Adam([y],lr=self.config['lr'])
            before=None
            for i in range(self.config['inner_steps']):
                opt.zero_grad(set_to_none=True);loss,pos,neg=self.energy(y,origin)
                assert torch.isfinite(loss)
                if before is None:before=float(loss.detach())
                loss.backward();assert y.grad is not None and torch.isfinite(y.grad).all()
                grad=float(y.grad.norm());opt.step()
                with torch.no_grad():
                    delta=y-origin;rms=delta.square().mean().sqrt()
                    if rms>self.config['trust_RMS']:y.copy_(origin+delta*(self.config['trust_RMS']/rms))
            with torch.no_grad():after,p,n=self.energy(y,origin)
        y=y.detach();v=self._xstart_to_pred(x_t,t,y)
        out.pred_x_0=y;out.pred_x_prev=x_t-(t-t_prev)*v
        self.trace.append({'t':t,'loss_before':before,'loss_after':float(after),'positive':float(p),'negative':float(n),
            'last_gradient_norm':grad,'clean_latent_delta_RMS':float((y-origin).square().mean().sqrt())})
        return out

class AddressedSparseNoise:
    def __init__(self,sampler,seed):self.sampler=sampler;self.seed=seed;self.calls=0
    def sample(self,model,noise,*args,**kwargs):
        coords=noise.coords.detach().cpu().long();assert (coords[:,0]==0).all() and (coords[:,1:]>=0).all() and (coords[:,1:]<128).all()
        channels=noise.feats.shape[-1]
        field=torch.randn((128,128,128,channels),generator=torch.Generator().manual_seed(self.seed+self.calls*1000))
        values=field[coords[:,1],coords[:,2],coords[:,3]].to(noise.device,dtype=noise.feats.dtype)
        self.calls+=1;del field
        return self.sampler.sample(model,noise.replace(values),*args,**kwargs)

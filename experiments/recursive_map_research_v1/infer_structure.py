"""Own learned Structure from noise; no target read and no postprocessing."""
from runtime import *
import argparse,time
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

class TopologyCondition:
    def __init__(self,text,raster,topology):self.text,self.map,self.topology=text,raster,topology
    @property
    def shape(self):return self.text.shape
    def repeat(self,*repeats):
        assert repeats[0]==1
        return TopologyCondition(self.text.repeat(*repeats),self.map.repeat(repeats[0],1,1,1),self.topology)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--update',type=int,required=True);a=ap.parse_args()
    assert a.candidate in ['g1_a','g2_a']
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    model,branch,parent,_=initialize(a.candidate,a.seed)
    meta=read(JOB/'runs'/a.candidate/f'seed{a.seed}'/f'CHECKPOINT_{a.update:06d}.json');assert sha(meta['path'])==meta['sha256']
    state=torch.load(meta['path'],map_location='cpu',weights_only=False)
    assert set(branch)==set(state['branch_state'])
    for k,p in branch.items():p.copy_(state['branch_state'][k])
    del state;model.eval().requires_grad_(False)
    decoder=structure_decoder();sampler=FlowEulerGuidanceIntervalSampler(1e-5)
    zero=relation_to_device(empty_relation(),'cuda')
    packet=read(JOB/'packets'/f'{a.candidate}_{a.update}.json')
    output=JOB/'evaluation'/a.candidate/f'seed{a.seed}'/f'plus{a.update:03d}';output.mkdir(parents=True,exist_ok=True)
    contexts={};calls={'mapped_nonzero':0,'unknown_exact_zero':0}
    original=model.branch.read
    def checked(hidden,prepared,site):
        value=original(hidden,prepared,site)
        if site==0:
            if prepared[3].count_nonzero()==0:
                assert value.count_nonzero()==0;calls['unknown_exact_zero']+=1
            elif value.count_nonzero()>0:calls['mapped_nonzero']+=1
        return value
    model.branch.read=checked
    for r in packet['rows']:
        out=output/r['scene_id']/f'seed{r["seed_index"]}';out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():
            for path,h in read(out/'COMPLETE.json')['files'].items():assert sha(out/path)==h
            continue
        assert not (out/'RAW_STRUCTURE.pt').exists(),'saved native sample requires additive recovery'
        tick=time.time();assert sha(r['map_path'])==r['map_sha256']
        with np.load(r['map_path']) as z:cpu=z['map_tensor'].copy()
        m=torch.from_numpy(cpu)[None].float().cuda();unknown=torch.zeros_like(m);unknown[:,5]=1
        topo=relation_to_device(compile_relation(cpu),'cuda')
        tensors=[]
        for spec in [r['text'],r['negative']]:
            key=spec['sha256']
            if key not in contexts:
                assert sha(spec['path'])==key
                contexts[key]=torch.load(spec['path'],map_location='cpu',weights_only=True)['context'][None].cuda()
            tensors.append(contexts[key])
        noise=torch.randn((1,8,16,16,16),device='cuda',generator=torch.Generator(device='cuda').manual_seed(r['noise_seed']))
        def prediction(x,t,c):return model(x,t,c.text,c.map,c.topology)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=sampler.sample(prediction,noise,TopologyCondition(tensors[0],m,topo),neg_cond=TopologyCondition(tensors[1],unknown,zero),
                 steps=25,rescale_t=3,cfg_strength=7.5,cfg_interval=[.5,.95],verbose=False).samples
            logits=decoder(s.float());q=torch.argwhere(logits>0)[:,[0,2,3,4]].int()
        assert torch.isfinite(s).all()
        torch.save({'predicted_S':s.cpu(),'predicted_Q':q.cpu(),'structure_noise':noise.cpu(),'checkpoint':meta,'no_target_input':True},out/'RAW_STRUCTURE.pt')
        save(out/'COMPLETE.json',{'files':{'RAW_STRUCTURE.pt':sha(out/'RAW_STRUCTURE.pt')},'packet':r,'seconds':time.time()-tick,'branch_calls':calls.copy()})
        print('NATIVE_SAMPLE',r['scene_id'],r['seed_index'],len(q),time.time()-tick,flush=True)
    assert calls['mapped_nonzero'] and calls['unknown_exact_zero']
    save(output/'COMPLETE.json',{'cases':len(packet['rows']),'checkpoint_sha256':meta['sha256'],'branch_calls':calls})

if __name__=='__main__':main()

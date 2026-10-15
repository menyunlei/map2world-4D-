"""Actual Gaussian alpha composition via background difference, no centre proxy."""
from runtime import *
import argparse,utils3d
from trellis.representations.gaussian import Gaussian
from scene_render import render_gs
NEW=JOB/'narrow_structure_v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--update',type=int,required=True);a=p.parse_args()
    torch.set_grad_enabled(False);torch.cuda.set_device(0);torch.set_num_threads(4)
    protocol=read(NEW/'TRANSMITTANCE_PROTOCOL.json')
    packet=read(NEW/a.arm/'packets'/f'g2_b_{a.update}.json')
    labels=read(JOB/'verifier_private/LABEL_MANIFEST.json')
    out=NEW/a.arm/'transmittance'/f'seed{a.seed}/plus{a.update:03d}.json';assert not out.exists()
    result=[]
    for r in packet['rows']:
        if not r.get('native_witness'):continue
        sid=r['scene_id']; d=NEW/a.arm/'native/g2_b'/f'seed{a.seed}/plus{a.update:03d}/own'/sid/'seed0'
        receipt=read(d/'COMPLETE.json');path=d/'NATIVE_GAUSSIANS.pt'
        assert sha(path)==receipt['files']['NATIVE_GAUSSIANS.pt']
        state=torch.load(path,map_location='cpu',weights_only=True)
        g=Gaussian(**state['init_params'],device='cuda')
        for k in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']:
            v=state[k];setattr(g,k,None if v is None else v.cuda())
        labelid=r['family']+'_open3m' if r['variant']=='closed' else sid
        lm=labels[labelid];assert sha(lm['path'])==lm['sha256']
        with np.load(lm['path']) as z:core=z['eroded_xy'].copy()
        rays=[]
        for ix,iy in np.argwhere(core):
            xy=np.array([ix,iy])*.5-15.75
            em=np.r_[xy,protocol['eye_height_m']];look=em+[0,0,10]
            eye=torch.tensor((em-[0,0,14])/32,dtype=torch.float32,device='cuda')
            target=torch.tensor((look-[0,0,14])/32,dtype=torch.float32,device='cuda')
            E=utils3d.torch.extrinsics_look_at(eye,target,torch.tensor([0.,1.,0.],device='cuda'))
            f=1/(2*np.tan(np.deg2rad(protocol['vertical_fov_degrees'])/2))
            K=torch.tensor([[f,0,.5],[0,f,.5],[0,0,1]],device='cuda',dtype=torch.float32)
            black=render_gs(g,E,K,9,9,bg=(0,0,0));white=render_gs(g,E,K,9,9,bg=(1,1,1))
            diff=(white-black)[4,4].float();assert torch.isfinite(diff).all()
            assert float(diff.max()-diff.min())<1e-3,'RGB background-difference inconsistency'
            value=float(diff.mean());assert -.001<=value<=1.001
            value=min(1.,max(0.,value))
            rays.append({'xy_m':xy.tolist(),'transmittance':value,'opacity':1-value})
        opacity=np.array([v['opacity'] for v in rays])
        result.append({'scene':sid,'variant':r['variant'],'native_sha256':sha(path),'core_reference_sha256':lm['sha256'],
            'rays':rays,'core_count':len(rays),'mean_opacity':float(opacity.mean()) if len(rays) else None,
            'fraction_opacity_ge0p5':float((opacity>=.5).mean()) if len(rays) else None,
            'fraction_opacity_ge0p9':float((opacity>=.9).mean()) if len(rays) else None,
            'empty_core_not_pass':not bool(len(rays)),
            'interpretation':'closed opacity is closure diagnostic, not an opening failure' if r['variant']=='closed' else 'actual upward native Gaussian obstruction'})
        del g,state
    save(out,{'arm':a.arm,'training_seed':a.seed,'update':a.update,'protocol_sha256':sha(NEW/'TRANSMITTANCE_PROTOCOL.json'),
        'source_sha256':sha(__file__),'rows':result,'actual_alpha_composition':True,'HUMAN_REVIEW_PENDING':True})
    print('TRANSMITTANCE_COMPLETE',str(out),flush=True)


if __name__=='__main__':main()

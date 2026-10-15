"""Keep original inference CuDNN flag. A diagnostic setting is not model repair.

The initial trace used cudnn.allow_tf32=False from TRAIN, whereas v6 Structure
inference used the native default True. SS velocity/latent was bit-identical;
one decoded binary support differed at threshold zero. Preserve the first
trace and run only the missing seed into a new output directory.
"""
import inspect
import trellis2_structure_path_diagnosis_v7 as probe
probe.OUT=probe.OUT/'decoder_math_fix1'
if __name__=='__main__':
    source=inspect.getsource(probe.main).replace('torch.backends.cudnn.allow_tf32=False','torch.backends.cudnn.allow_tf32=True')
    scope=dict(probe.__dict__)
    exec(compile(source,__file__,'exec'),scope)
    scope['main']()

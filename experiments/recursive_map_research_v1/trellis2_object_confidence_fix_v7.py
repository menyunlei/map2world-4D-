"""One input bug fix: object reliability is not whole-crop observation density.

Do not multiply unknown-background area into the object's confidence. All
rasters, poses, object slots, unknown masks and adapter weights are unchanged.
Use the existing [0,1] known evidence on the object's footprint support only.
TRAIN known masks may also cover known exterior: intersect via minimum, not
sum of all known background. Real confidence already includes annotation/UI
uncertainty; it must NOT be set to one or manually strengthened.
"""
import numpy as np

def correct_object_confidence(m):
    out={k:v.copy() for k,v in m.items()}
    fp=np.asarray(m['footprints'],np.float64)
    known=np.asarray(m['footprint_known'],np.float64)
    mass=fp.sum((1,2))
    confidence=np.minimum(fp,known).sum((1,2))/np.maximum(mass,1e-12)
    confidence=np.where(m['object_valid'],confidence,0.).clip(0,1).astype(np.float32)
    out['object_summary'][:,5]=confidence
    for k in m:
        if k!='object_summary':assert np.array_equal(m[k],out[k]),k
    assert np.array_equal(m['object_summary'][:,:5],out['object_summary'][:,:5])
    assert np.array_equal(m['object_summary'][:,6:],out['object_summary'][:,6:])
    return out

if __name__=='__main__':
    import unittest
    class ConfidenceTests(unittest.TestCase):
        def sample(self):
            fp=np.zeros((1,64,64),np.float32);fp[:,20:36,20:36]=1
            known=fp*.6
            return {'footprints':fp,'footprint_known':known,'object_valid':np.ones(1,bool),
                'object_summary':np.array([[0,0,.25,.25,.0625,known.mean(),0,1]],np.float32),
                'roads':np.zeros((1,64,64),np.float32),'target_object':np.array(0,np.int64)}
        def test_background_is_not_object_uncertainty(self):
            m=self.sample();fixed=correct_object_confidence(m)
            self.assertAlmostEqual(float(fixed['object_summary'][0,5]),.6,places=6)
            self.assertAlmostEqual(float(m['object_summary'][0,5]),.0375,places=6)
            # Known exterior must not change reliability either.
            n={k:v.copy() for k,v in m.items()};n['footprint_known'][n['footprints']==0]=1
            self.assertEqual(float(correct_object_confidence(n)['object_summary'][0,5]),float(fixed['object_summary'][0,5]))
        def test_unobserved_foreground_stays_unknown(self):
            m=self.sample();m['footprint_known'][:,:,:]=0
            self.assertEqual(float(correct_object_confidence(m)['object_summary'][0,5]),0)
        def test_full_TRAIN_confidence_unchanged(self):
            m=self.sample();m['footprint_known'][:]=1;m['object_summary'][0,5]=1
            np.testing.assert_array_equal(correct_object_confidence(m)['object_summary'],m['object_summary'])
        def test_partial_UI_occlusion_preserved(self):
            m=self.sample();m['footprint_known'][:,20:28,20:36]=0
            self.assertAlmostEqual(float(correct_object_confidence(m)['object_summary'][0,5]),.3,places=6)
            np.testing.assert_array_equal(correct_object_confidence(m)['footprint_known'],m['footprint_known'])
    unittest.main()

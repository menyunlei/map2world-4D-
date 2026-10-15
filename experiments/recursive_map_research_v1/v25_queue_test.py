"""CPU-only queue invariants. No SSH or GPU work."""
import tempfile,unittest,json,concurrent.futures
from pathlib import Path
from unittest.mock import patch
import v25_server as s

class QueueTests(unittest.TestCase):
    def test_move_prefetch_and_revisit_never_duplicate(self):
        initial=json.loads(json.dumps(s.state))
        for c in initial['chunks']: c['status']='waiting_camera'
        initial['budget']['used']=0
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp,patch.object(s,'OUT',Path(tmp)),patch.object(s,'state',initial),patch.object(s,'generate') as generate:
            payload=dict(world='native',position=[-.7,2.73,45],forward=[0,0,-1],session='cpu-test')
            self.assertEqual(s.camera(payload)['accepted'],[])
            payload['position'][2]=35
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                outputs=list(pool.map(lambda _:s.camera(payload),range(12)))
            self.assertEqual(sum(len(o['accepted']) for o in outputs),2)
            self.assertEqual(generate.call_count,2)
            self.assertEqual(s.state['budget']['used'],2)
            self.assertEqual(len(list(Path(tmp).rglob('ATTEMPT.json'))),2)
            payload['position'][2]=45
            self.assertEqual(s.camera(payload)['accepted'],[])
            payload['position'][2]=-350
            self.assertEqual(s.camera(payload)['accepted'],[])
    def test_invalid_camera_rejected(self):
        with self.assertRaises(ValueError):s.camera(dict(position=[float('nan'),0,0],forward=[0,0,-1]))

if __name__=='__main__':unittest.main()

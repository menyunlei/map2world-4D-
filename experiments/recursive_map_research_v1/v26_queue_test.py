"""Actual navigation gating, CPU only; no SSH/GPU."""
import unittest,tempfile,json,os
from pathlib import Path
from unittest.mock import patch
import v26_server as s
class Queue(unittest.TestCase):
 def test_request_publication_is_atomic(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
   root=Path(temp);upload=root/'REQUEST.upload';ready=root/'REQUEST.json'
   with upload.open('x') as f:
    f.write('{"chunk_id":');f.flush()
    self.assertFalse(ready.exists())
    f.write('"fixed"}')
   os.rename(upload,ready)
   self.assertEqual(json.loads(ready.read_text()),{'chunk_id':'fixed'})
 def test_walk_each_frontier_once(self):
  original=json.loads(json.dumps(s.state))
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,patch.object(s,'OUT',Path(temp)),patch.object(s,'state',original),patch.object(s,'generate') as gen:
   p=dict(world='native',position=[-.7,2.73,-160],forward=[0,0,-1],loaded=[])
   self.assertEqual(s.camera(p)['accepted'],[])
   p['position'][2]=-162
   self.assertEqual(s.camera(p)['accepted'],['walk-0--4'])
   self.assertEqual(s.camera(p)['accepted'],[])
   p['position'][2]=-240
   self.assertEqual(s.camera(p)['accepted'],[]) # prior job still active
   first=s.state['chunks'][0];first.update(status='ready',sha256='fixed')
   p['position'][2]=-250
   self.assertEqual(s.camera(p)['accepted'],[]) # not yet loaded in browser
   p['loaded']=[dict(id=first['id'],sha256='fixed')]
   self.assertEqual(s.camera(p)['accepted'],[]) # standing still cannot request
   p['position'][2]=-251
   self.assertEqual(s.camera(p)['accepted'],['walk-0--5'])
   p['position'][2]=-160
   self.assertEqual(s.camera(p)['accepted'],[]) # revisit does not regenerate
   self.assertEqual(s.state['budget']['used'],2)
   self.assertEqual(gen.call_count,2)
if __name__=='__main__':unittest.main()

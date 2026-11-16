import sys,unittest,copy
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent/'candidate'))
import numpy as np
import ground_recovery as g
import warning_surface as w

class Recovery(unittest.TestCase):
 def setUp(self):
  self.ctx={'ready':True,'motion_ready':True,'speed_mps':2.,'yaw_rate_rps':0.}
  self.q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':.01}
  self.ref={'abc':[0.,0.,0.],'at':10.,'travel_m':0.,'turn_rad':0.}
  x,y=np.meshgrid(np.arange(4.1,15,.3),np.arange(-2.9,2.9,.3));self.points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
 def test_same_fresh_ground_recovers(self):
  r=g.recover_fresh(self.ref,{'abc':[0.,0.,.01],'stable_frames':1},self.points)
  self.assertIsNotNone(r);self.assertFalse(r['uses_held_geometry'])
 def test_changed_grade_does_not_recover(self):
  for abc in ([0,0,.3],[.04,0,0],[0,.08,0]):self.assertIsNone(g.recover_fresh(self.ref,{'abc':abc,'stable_frames':1},self.points))
 def test_unsupported_ground_does_not_recover(self):
  self.assertIsNone(g.recover_fresh(self.ref,{'abc':[0,0,0],'stable_frames':1},self.points+[0,0,.3]))
 def test_expiry_travel_turn_and_invalid_quality(self):
  self.assertIsNone(g.advance_reference(self.ref,self.ctx,self.q,.1,11))
  self.assertIsNone(g.advance_reference(self.ref,{**self.ctx,'speed_mps':30},self.q,.2,10.2))
  self.assertIsNone(g.advance_reference(self.ref,{**self.ctx,'yaw_rate_rps':1},self.q,.1,10.1))
  for q in ({'arrival_age_s':1},{'completeness':.8},{'timestamp_valid':False},{'processing_overrun':True}):self.assertIsNone(g.advance_reference(self.ref,self.ctx,{**self.q,**q},.1,10.1))
  self.assertIsNone(g.advance_reference(self.ref,self.ctx,self.q,.1,10.1,True))
 def test_fresh_healthy_reference_only(self):
  self.assertIsNone(g.remember_fresh(None,None,self.ctx,self.q,10))
  self.assertIsNone(g.remember_fresh(None,{'abc':[0,0,0],'stable_frames':4},self.ctx,self.q,10))
  self.assertIsNotNone(g.remember_fresh(None,{'abc':[0,0,0],'stable_frames':5},self.ctx,self.q,10))

class Repeat(unittest.TestCase):
 def setUp(self):
  self.track=SimpleNamespace(position=np.array([10.,-2.]),lo=np.array([9.,-2.5,0.]),hi=np.array([11.,-1.5,1.]))
  self.c={'track_id':1,'direction':'RIGHT','corridor':{'in_path':True,'min_body_gap_m':-.03},'motion_evidence':{'trusted':True},'ground_velocity_mps':[0.,0.]}
  self.state=w.remember_surface(self.c,self.track,10.)
 def test_reidentified_same_surface(self):
  c={**self.c,'track_id':2,'motion_evidence':{'trusted':False}}
  self.assertTrue(w.repeat_surface(self.state,c,self.track,11.))
  self.assertFalse(w.repeat_surface(self.state,c,self.track,11.,SimpleNamespace(misses=0)))
 def test_distinct_deep_moving_and_expired_keep_authority(self):
  c={**self.c,'track_id':2,'motion_evidence':{'trusted':False}}
  far=SimpleNamespace(position=self.track.position+np.array([5,0]),lo=self.track.lo+np.array([5,0,0]),hi=self.track.hi+np.array([5,0,0]))
  self.assertFalse(w.repeat_surface(self.state,c,far,11.))
  for change in ({'corridor':{'min_body_gap_m':-.3}}, {'motion_evidence':{'trusted':True}}, {'oncoming_candidate':True}, {'direction':'LEFT'}):
   self.assertFalse(w.repeat_surface(self.state,{**c,**change},self.track,11.))
  self.assertFalse(w.repeat_surface(self.state,c,self.track,14.))
 def test_no_evidence_no_memory(self):
  self.assertIsNone(w.remember_surface({**self.c,'motion_evidence':{'trusted':False}},self.track,10.))
  self.assertIsNone(w.remember_surface({**self.c,'ground_velocity_mps':[4,0]},self.track,10.))
 def test_gap_drops_surface(self):
  ctx={'ready':True,'motion_ready':True,'speed_mps':5.,'yaw_rate_rps':0.};q={'completeness':1,'timestamp_valid':True,'arrival_age_s':.01}
  s=w.advance_surface(self.state,ctx,q,.1,10.1);self.assertAlmostEqual(s['position'][0],9.5)
  self.assertIsNone(w.advance_surface(self.state,ctx,q,.3,10.3))

class Checkpoint(unittest.TestCase):
 def test_new_state_roundtrips_and_old_codec_rejected(self):
  from fcw_engine import Engine
  from telemetry import Telemetry
  from registration import ScanMotion
  from frames import FrameAssembler
  from learning.replay_checkpoint import encode_state,decode_state,CheckpointRejected
  import json
  engine=Engine();engine._settled_road_reference={'abc':[0.,0.,.01],'at':10.,'travel_m':1.,'turn_rad':0.}
  engine._settled_road_reference=g.advance_reference(engine._settled_road_reference,{'ready':True,'motion_ready':True,'speed_mps':2.,'yaw_rate_rps':.01},{'completeness':1.,'timestamp_valid':True,'arrival_age_s':.01},.1,10.1)
  engine._warning_surface={'track_id':1,'direction':'RIGHT','at':10.,'position':[10.,-2.],'lo':[9.,-2.5],'hi':[11.,-1.5]}
  raw=encode_state(engine,Telemetry(),FrameAssembler(),ScanMotion());restored=decode_state(raw)[0]
  for name in ('_settled_road_reference','_warning_surface'):self.assertEqual(getattr(engine,name),getattr(restored,name))
  old=json.loads(raw);old['codec_version']=6
  with self.assertRaises(CheckpointRejected):decode_state(json.dumps(old).encode())
if __name__=='__main__':unittest.main()

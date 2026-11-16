"""Synthetic exact geometry; never advertised as naturalistic recall."""
import sys,json
import numpy as np
sys.path.insert(0,sys.argv[1])
from fcw_engine import Engine,Mount
from corridor_occupancy import assess
m=Mount(measured=True,alignment_verified=True);edge=m.width_m/2
p=np.array([[10.,edge-.05],[10.1,edge+.02],[10.2,edge+.05],[10.,edge+.10],[10.1,edge+.15],[10.2,edge+.20]])
checks={'partial_body_overlap_preserved':bool(assess(p,np.array([[0.,0.],[40.,0.]]),m,40.,0.)['in_path'])}
x,y=np.meshgrid(np.linspace(5,39,80),np.linspace(-8,8,45));ground=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
yy,zz=np.meshgrid(np.linspace(-.8,.8,15),np.linspace(.25,1.7,12))
for label,change in [('healthy',{}),('stale',{'arrival_age_s':1.}),('incomplete',{'completeness':.8}),('invalid_time',{'timestamp_valid':False})]:
 engine=Engine(m);alerts=[]
 for i in range(14):
  obstacle=np.column_stack([np.full(yy.size,15-.5*i),yy.ravel(),zz.ravel()]);body=np.vstack([ground,obstacle]);points=np.column_stack([body-np.asarray(m.xyz),np.ones(len(body))*40])
  ctx={'at_ns':100_000_000_000+i*100000000,'speed_mps':5.,'yaw_rate_rps':0.,'acceleration_mps2':0.,'ready':True,'motion_ready':True,'reasons':[],'path':np.column_stack([np.linspace(0,120,61),np.zeros(61)])}
  q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':.001,**change}
  alerts.extend(engine.update(points,ctx,q)['alerts'])
 checks[label+'_warning_gate']=bool(alerts) if label=='healthy' else not alerts
print(json.dumps({'synthetic':True,'checks':checks,'passed':all(checks.values())},indent=2))
raise SystemExit(0 if all(checks.values())else 1)

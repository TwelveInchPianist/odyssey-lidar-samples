"""Synthetic commissioning demonstration, not sensor-performance validation."""
import runtime_clock as time
import numpy as np
from fcw_engine import Engine,Mount

LIMITED_SCENARIO='LIMITED RANGE — synthetic ground loss / near-field only'

SCENARIOS=[
 ('Obstacle ahead',lambda t:(47-15*t,0),1.6,.7,False),
 ('Oncoming threat in our path',lambda t:(90-35*t,0),1.6,1.8,False),
 ('Opposing-lane traffic — should stay quiet',lambda t:(90-35*t,3.6),1.6,1.8,False),
 ('Small obstacle crossing from right',lambda t:(35-15*t,-4+2*t),.65,.7,False),
 ('Roadside object — should stay quiet',lambda t:(47-15*t,5),1.6,1.,False),
 ('Overhead object — should stay quiet',lambda t:(47-15*t,0),1.,1.,True),
 (LIMITED_SCENARIO,lambda t:(20-15*t,0) if t<.8 else (18,5),1.6,1.,False)
]

def points_for(x,y,height,width,overhead=False):
    rng=np.random.default_rng(33)
    gx,gy=np.meshgrid(np.linspace(5,39,62),np.linspace(-8,8,35))
    g=np.column_stack([gx.ravel(),gy.ravel(),rng.normal(0,.015,gx.size),np.ones(gx.size)*80])
    yy,zz=np.meshgrid(np.linspace(y-width/2,y+width/2,7),np.linspace(.25,height,9))
    o=np.column_stack([x+rng.normal(0,.035,yy.size),yy.ravel(),zz.ravel()+(3 if overhead else 0),np.ones(yy.size)*5])
    p=np.concatenate([g,o]);p[:,2]-=1.79034
    return p

def points_without_ground(x,y,height,width):
    """Synthetic upright returns only: actual Ground.fit must fail naturally."""
    rng=np.random.default_rng(33)
    yy,zz=np.meshgrid(np.linspace(y-width/2,y+width/2,7),np.linspace(.25,height,9))
    return np.column_stack([x+rng.normal(0,.035,yy.size),yy.ravel(),zz.ravel()-1.79034,np.full(yy.size,5.)])

def run_demo(session):
    session.record_status='synthetic scenario logs only; no lidar connection'
    session.comma_status='simulated ideal path, motion and clock alignment'
    while not session.stop_event.is_set():
        for name,trajectory,height,width,overhead in SCENARIOS:
            session.engine=Engine(Mount(measured=True,alignment_verified=True),session.config.get('perception_policy'))
            for i in range(40 if name==LIMITED_SCENARIO else 26):
                if session.stop_event.is_set():return
                started=time.monotonic();t=i*.1;x,y=trajectory(t)
                ctx={'at_ns':time.monotonic_ns(),'speed_mps':15.,'yaw_rate_rps':0.,'acceleration_mps2':0,
                     'scenario':name,
                     'path':np.column_stack([np.linspace(0,120,80),np.zeros(80)]),'ready':True,'reasons':[]}
                q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':.001,'synthetic':True}
                points=(points_without_ground(x,y,height,width) if name==LIMITED_SCENARIO else points_for(x,y,height,width,overhead))
                session.process_observation(points,ctx,q)
                with session.lock:session.state['scenario']=name
                session.stop_event.wait(max(0,.1-(time.monotonic()-started)))

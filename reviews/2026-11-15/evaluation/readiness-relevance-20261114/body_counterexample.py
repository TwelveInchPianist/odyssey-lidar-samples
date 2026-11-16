"""Exact synthetic geometry, not a claim of a recorded encounter."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
from corridor_occupancy import assess
from fcw_engine import Mount
m=Mount();edge=m.width_m/2
# Six returns inside the padded corridor, with one genuinely inside the body.
# An upright object's other returns can lie outside the body while its edge clips it.
p=np.array([[10.,edge-.05],[10.1,edge+.02],[10.2,edge+.05],[10.,edge+.10],[10.1,edge+.15],[10.2,edge+.20]])
r=assess(p,np.array([[0.,0.],[40.,0.]]),m,40.,0.)
print(json.dumps({'case':'partial_body_overlap_six_return_object','synthetic':True,'known_body_intrusion_m':.05,'points_inside_physical_body':1,'result':r,'pass':r['in_path']},indent=2))

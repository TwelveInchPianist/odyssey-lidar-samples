"""Stable data contracts; numerical backends may change without rewriting evidence.

Frames: sensor -> body FLU -> continuous segment odom -> revision map -> ECEF.
A pose T_parent_child maps child column vectors into the parent frame. Never
overwrite an observation when a corrected pose or calibration becomes available.
"""
import hashlib,json,math
from dataclasses import asdict,dataclass,field
from typing import Optional,Protocol
import numpy as np

CONTRACT_VERSION=1

def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def identity(value):return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def checked_pose(value):
    T=np.asarray(value,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-8):raise ValueError('Invalid homogeneous pose')
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:raise ValueError('Pose rotation is not proper orthonormal')
    return T

def checked_uncertainty(value,dimensions=6):
    # These are declared standard-deviation estimates, not calibrated posterior
    # confidence. Correlations may be supplied through a future covariance field.
    v=np.asarray(value,dtype=float)
    if v.shape!=(dimensions,) or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Positive finite uncertainty required')
    return v.tolist()

@dataclass(frozen=True)
class Keyframe:
    id:str
    session_id:str
    segment_id:str
    at_ns:int
    cloud_hash:str
    calibration_id:str
    # Local original odometry estimate, not a corrected map pose.
    odom_from_body:list
    source_ref:dict
    quality:dict=field(default_factory=dict)
    location:Optional[dict]=None
    descriptor_version:str=''

    def validate(self):
        if not all(isinstance(v,str) and v for v in (self.id,self.session_id,self.segment_id,self.calibration_id)):raise ValueError('Stable nonempty keyframe identities required')
        if type(self.at_ns) is not int or self.at_ns<0:raise ValueError('Invalid monotonic timestamp')
        if len(self.cloud_hash)!=64 or any(c not in '0123456789abcdef' for c in self.cloud_hash):raise ValueError('Invalid cloud content hash')
        checked_pose(self.odom_from_body)
        canonical_json(asdict(self));return self

@dataclass(frozen=True)
class Constraint:
    id:str
    source_id:str
    target_id:str
    # Transform from target body into source body.
    source_from_target:list
    std:list
    kind:str
    provenance:dict
    state:str='candidate'

    def validate(self):
        if self.kind not in ('odometry','scan_registration','revisit'):raise ValueError('Unknown constraint kind')
        if self.state not in ('candidate','accepted','rejected'):raise ValueError('Unknown constraint state')
        if not self.id or not self.source_id or not self.target_id or self.source_id==self.target_id:raise ValueError('Invalid constraint identities')
        checked_pose(self.source_from_target);checked_uncertainty(self.std)
        canonical_json(asdict(self));return self

class PlaceIndex(Protocol):
    def candidates(self,cloud,location=None,exclude_session=None,limit=24):...

class RegistrationBackend(Protocol):
    def align(self,source,target,seed=None):...

class FrameSource(Protocol):
    version:str
    def signature(self,folder):...
    def __call__(self,folder,cancel=None):...

class GraphBackend(Protocol):
    def solve(self,keyframes,constraints,location_priors,cancel=None):...

class MapBackend(Protocol):
    def rebuild(self,revision_id,cancel=None):...


"""Small injected backend profile; no dynamic plugin loading or extra services."""
from dataclasses import dataclass
import numpy as np
from fcw_engine import Ground
from route_map import features
from .places import DescriptorIndex,ICPRegistration
from .graph import PoseGraph
from .static_map import StaticMap
from .contracts import identity

class StaticFeatures:
    version='above-ground-static-features-v1'
    def __init__(self):self.ground=Ground()
    def extract(self,body,boxes=()):
        ground=self.ground.fit(body)
        if ground is None:return {'points':None,'reason':'no_supported_ground'}
        heights=body[:,2]-(body[:,0]*ground['abc'][0]+body[:,1]*ground['abc'][1]+ground['abc'][2])
        p=body[(heights>.4)&(heights<5)]
        for box in boxes:p=p[~((p>=np.asarray(box['lo']))&(p<=np.asarray(box['hi']))).all(1)]
        return {'points':features(p),'ground':ground}

@dataclass(frozen=True)
class BackendProfile:
    version:str='own-M1-profile-v1'
    registration_factory:object=ICPRegistration
    feature_factory:object=StaticFeatures
    index_factory:object=DescriptorIndex
    graph_factory:object=PoseGraph
    map_factory:object=StaticMap

    def declaration(self,store):
        return {'profile':self.version,'registration':self.registration_factory().version,
                'features':self.feature_factory().version,'index':self.index_factory(store).version,
                'graph':self.graph_factory().version,'map':self.map_factory(store).version}
    def fingerprint(self,store):return identity(self.declaration(store))

"""Conservative temporal evidence for offline radar/lidar correspondence."""
import numpy as np
class TemporalPairs:
    def __init__(self):self.state={}
    def reset(self):self.state.clear()
    def confirm(self,lidar_id,radar_id,at_ns,source_ns):
        key=(str(lidar_id),str(radar_id));old=self.state.get(key)
        # An identity switch invalidates both directions, not just one edge.
        for k in list(self.state):
            if (k[0]==key[0] or k[1]==key[1]) and k!=key:self.state.pop(k)
        if old is None or at_ns<=old['last'] or at_ns-old['last']>250_000_000 or source_ns<old['source']:
            old={'first':at_ns,'last':at_ns,'source':source_ns,'samples':1}
        else:
            old['samples']+=source_ns!=old['source'];old.update(last=at_ns,source=source_ns)
        self.state[key]=old
        self.state={k:v for k,v in self.state.items() if at_ns-v['last']<=250_000_000}
        return {'confirmed':old['samples']>=3 and at_ns-old['first']>=200_000_000,
                'distinct_radar_samples':old['samples'],'duration_s':(at_ns-old['first'])/1e9}

def extent_distances(objects,positions,radar_points,publication_ns,rotation=None,translation=None):
    """Distance to visible lidar extent at object time, not centroid identity.

    This is only candidate ranking. Raw centroid/extent/time evidence is retained
    for the separate calibration fit and all original acceptance limits.
    """
    r=np.eye(2) if rotation is None else rotation;t=np.zeros(2) if translation is None else translation
    rows=[]
    for obj in objects:
        lo,hi=np.asarray(obj['lo'])[:2],np.asarray(obj['hi'])[:2]
        corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])@r.T+t
        low,high=corners.min(0),corners.max(0)
        dt=(publication_ns-obj['at_ns'])/1e9
        matched=np.asarray(positions).copy();matched[:,0]-=np.array([p['vRel'] for p in radar_points])*dt
        delta=np.maximum(np.maximum(low-matched,matched-high),0.)
        rows.append(np.linalg.norm(delta,axis=1))
    return np.asarray(rows)

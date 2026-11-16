"""Specific, short-lived person-image correspondence; no class/range inference.

Native features may connect camera identities. Missing/ambiguous features never
remove an observation or delay its semantic confirmation. There is no learned
similarity cutoff, geographic mask or replacement object tracker here.
"""
from collections import Counter, deque
import math
import uuid

METHOD = 'native-person-features-v1'
MIN_GAP_NS = 50_000_000
MAX_GAP_NS = 300_000_000
SUPPORT_NS = 600_000_000
CAMERAS = {'narrow', 'wide'}


def supported(match):
    """Validate the small feature-support record at the policy boundary."""
    if not isinstance(match, dict): return False
    counts = match.get('distinct_sites')
    areas = match.get('hull_area_px2')
    fraction = match.get('inlier_fraction')
    return (type(match.get('inliers')) is int and match['inliers'] >= 6
        and isinstance(counts, list) and len(counts) == 2
        and all(type(n) is int and 6 <= n <= match['inliers'] for n in counts)
        and isinstance(areas, list) and len(areas) == 2
        and all(type(n) in (int, float) and math.isfinite(n) and n >= 4 for n in areas)
        and type(fraction) in (int, float) and math.isfinite(fraction) and .8 <= fraction <= 1)


def peer_keys(observation, own_keys):
    """Only current proof for an actually observed person can affect identity."""
    links = observation.get('visual_identity_links')
    sensor = observation.get('sensor')
    epoch = observation.get('visual_identity_epoch')
    if (sensor not in {'camera:narrow', 'camera:wide'} or not isinstance(epoch, str) or not epoch
        or observation.get('source_frame_valid') is not True
        or observation.get('class_confirmed') is not True
        or observation.get('label') not in {'person', 'cyclist', 'motorcycle', 'scooter'}
        or not isinstance(links, list) or len(links) != 1): return set()
    link = links[0]
    if not isinstance(link, dict): return set()
    current, peer = link.get('source_ns'), link.get('peer_source_ns')
    key, peer_key = link.get('source_key'), link.get('peer_key')
    other = 'camera:wide' if sensor == 'camera:narrow' else 'camera:narrow'
    pairs = link.get('capture_pairs')
    if (link.get('method') != METHOD or link.get('epoch') != epoch
        or type(current) is not int or type(peer) is not int or not 0 < peer < current
        or not MIN_GAP_NS <= current-peer <= MAX_GAP_NS
        or current != observation.get('source_ns') or link.get('frame_id') != observation.get('frame_id')
        or not isinstance(key,str) or key not in own_keys or not key.startswith(sensor+':')
        or not isinstance(peer_key, str) or not peer_key.startswith(other+':') or len(peer_key) > 200
        or not supported(link.get('features'))
        or not isinstance(pairs, list) or not 3 <= len(pairs) <= 8): return set()
    samples = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
            or not all(type(t) is int and t > 0 for t in pair)
            or not MIN_GAP_NS <= abs(pair[0]-pair[1]) <= MAX_GAP_NS
            or not 0 <= current-max(pair) <= SUPPORT_NS): return set()
        samples.append(tuple(pair))
    expected = (current,peer) if sensor=='camera:narrow' else (peer,current)
    if len(set(samples)) != len(samples) or expected not in samples: return set()
    return {peer_key}


class VisualIdentity:
    def __init__(self):
        self.cv2 = self.sift = self.matcher = None
        self.epoch = uuid.uuid4().hex
        self.frames = {c: deque(maxlen=4) for c in CAMERAS}
        self.history = {}
        self.last = {}
        self.domain = None

    def _backend(self):
        if self.cv2 is None:
            import cv2
            self.cv2=cv2
            self.sift=cv2.SIFT_create(nfeatures=3000)
            self.matcher=cv2.BFMatcher(cv2.NORM_L2)

    def reset(self):
        self.epoch = uuid.uuid4().hex
        for values in self.frames.values(): values.clear()
        self.history.clear(); self.last.clear(); self.domain = None

    def extract(self, gray):
        import numpy as np
        self._backend()
        h,w = gray.shape
        key,des = self.sift.detectAndCompute(gray,None)
        keep = [i for i,k in enumerate(key) if .1*w <= k.pt[0] <= .9*w and .1*h <= k.pt[1] <= .9*h]
        return {'points':np.array([key[i].pt for i in keep],np.float32).reshape(-1,2),
                'descriptors':des[keep] if des is not None else np.empty((0,128),np.float32),
                'width':w,'height':h}

    def compare(self, a, b):
        import numpy as np
        self._backend()
        da,db = a['descriptors'],b['descriptors']
        if min(len(da),len(db)) < 2: return None
        def good(x,y):
            return {m.queryIdx:m.trainIdx for group in self.matcher.knnMatch(x,y,k=2)
                    if len(group)==2 for m,n in [group] if m.distance < .75*n.distance}
        ab,ba = good(da,db),good(db,da)
        pairs = [(i,j) for i,j in ab.items() if ba.get(j)==i]
        if len(pairs)<6: return None
        pa = np.array([a['points'][i] for i,j in pairs])
        pb = np.array([b['points'][j] for i,j in pairs])
        sites = lambda p: int(len(np.unique(np.rint(p/2).astype(int),axis=0)))
        if min(sites(pa),sites(pb))<6: return None
        src,dst = (pb,pa) if a['width']*a['height'] < b['width']*b['height'] else (pa,pb)
        M,mask = self.cv2.estimateAffinePartial2D(src,dst,method=self.cv2.RANSAC,
            ransacReprojThreshold=3.,maxIters=3000,confidence=.995,refineIters=10)
        if M is None: return None
        m = mask[:,0]!=0
        result = {'inliers':int(m.sum()),'inlier_fraction':float(m.mean()),
                  'distinct_sites':[sites(p[m]) for p in (pa,pb)],
                  'hull_area_px2':[float(self.cv2.contourArea(self.cv2.convexHull(p[m]))) for p in (pa,pb)]}
        return result if supported(result) else None

    def observe(self, camera, stamp, frame_id, people, *, native_stamp, domain, writer):
        """Accept already extracted current features; also used by saved replay."""
        if camera not in CAMERAS: raise ValueError('Road cameras only')
        old = self.last.get(camera)
        if (not isinstance(domain,str) or not domain or type(stamp) is not int or stamp<=0
            or type(native_stamp) is not int or native_stamp<=0 or writer is None):
            self.reset(); return []
        if (self.domain is not None and domain!=self.domain) or (old and (
            stamp<=old[0] or native_stamp<=old[1] or writer!=old[2])):
            self.reset()
        self.domain=domain; self.last[camera]=(stamp,native_stamp,writer)
        # Each camera advances monotonically, but their processing order need
        # not follow exposure order. An earlier peer exposure must not erase
        # already received support from the other stream. Retain by watermark;
        # only causal support for the current exposure may be emitted below.
        watermark=max(value[0] for value in self.last.values())
        self.history={k:deque(((t,p) for t,p in values if 0<=watermark-t<=SUPPORT_NS),maxlen=8)
                      for k,values in self.history.items()}
        self.history={k:v for k,v in self.history.items() if v}
        other='wide' if camera=='narrow' else 'narrow'
        eligible=[r for r in self.frames[other] if MIN_GAP_NS<=stamp-r['stamp']<=MAX_GAP_NS
                  and MIN_GAP_NS<=native_stamp-r['native_stamp']<=MAX_GAP_NS]
        peer=eligible[-1] if eligible else None
        links=[]
        if peer is not None:
            candidates=[]
            for a in people:
                for b in peer['people']:
                    match=self.compare(a,b)
                    if match is not None: candidates.append((a,b,match))
            na=Counter(a['key'] for a,b,m in candidates); nb=Counter(b['key'] for a,b,m in candidates)
            for a,b,match in candidates:
                nk,wk=(a['key'],b['key']) if camera=='narrow' else (b['key'],a['key'])
                key=(nk,wk)
                if na[a['key']]!=1 or nb[b['key']]!=1:
                    self.history.pop(key,None); continue
                if not (a['class_confirmed'] or b['class_confirmed']): continue
                pair=(stamp,peer['stamp']) if camera=='narrow' else (peer['stamp'],stamp)
                h=self.history.setdefault(key,deque(maxlen=8))
                if not any(p==pair for t,p in h): h.append((stamp,pair))
                past=[p for t,p in h if 0<=stamp-t<=SUPPORT_NS]
                if len(past)>=3:
                    links.append({'method':METHOD,'epoch':self.epoch,'source_key':a['key'],'peer_key':b['key'],
                        'source_ns':stamp,'peer_source_ns':peer['stamp'],'frame_id':frame_id,
                        'capture_pairs':[list(p) for p in past],'features':match})
        self.frames[camera].append({'stamp':stamp,'native_stamp':native_stamp,'people':people})
        return links

    def update(self, camera, observations, bgr, metadata):
        """Decorate, never discard, current semantic measurements."""
        import numpy as np
        self._backend()
        stamp=metadata['source_ns'];frame_id=metadata['frame_id']
        people={}
        for o in observations:
            for p in o.get('components',[o]):
                if p.get('label')=='person' and p.get('source_frame_valid') is True:
                    people[p['key']]=p
        # Excess identity work is unavailable as a whole, not a ranked subset
        # that could turn a competing person into a falsely unique match.
        if len(people)>32:
            self.reset(); return self.decorate(observations,[])
        features=[]
        h,w=bgr.shape[:2]
        if bgr.dtype!=np.uint8 or bgr.shape!=(h,w,3): raise ValueError('Native BGR image required')
        for p in people.values():
            box=p.get('bbox_xyxy')
            if (p['source_ns']!=stamp or p.get('sensor')!='camera:'+camera or p.get('frame_id')!=frame_id
                or not isinstance(box,(list,tuple)) or len(box)!=4
                or not all(type(v) in (int,float) and math.isfinite(v) for v in box)
                or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h):
                self.reset(); return self.decorate(observations,[])
            x0,y0,x1,y1=map(int,(math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])))
            # Use the same PNG grayscale conversion as the native diagnostic;
            # no resize, brightening or depth is introduced.
            ok,png=self.cv2.imencode('.png',bgr[y0:y1,x0:x1])
            if not ok: raise ValueError('Cannot encode identity crop')
            gray=self.cv2.imdecode(png,self.cv2.IMREAD_GRAYSCALE)
            features.append({**self.extract(gray),'key':p['key'],'class_confirmed':p.get('class_confirmed') is True})
        links=self.observe(camera,stamp,frame_id,features,native_stamp=metadata.get('native_source_ns'),
            domain=metadata.get('native_clock_domain'),writer=metadata.get('writer'))
        return self.decorate(observations,links)

    def decorate(self, observations, links):
        """Only this call's evidence survives; never forward inherited proof."""
        result=[]
        for o in observations:
            owns={p['key'] for p in o.get('components',[o])}
            value={k:v for k,v in o.items() if k not in ('visual_identity_epoch','visual_identity_links')}
            value['visual_identity_epoch']=self.epoch
            linked=[v for v in links if v['source_key'] in owns]
            if linked: value['visual_identity_links']=linked
            result.append(value)
        return result

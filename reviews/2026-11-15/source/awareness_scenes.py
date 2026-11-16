"""Speech receipts for observed roadside groups, not physical identity merging.

One observed person may become a plural group. A plural delivered description
can cover nearby current members through ID churn while the vehicle/view stays
stationary. All raw observations and independent perception tracks remain.
"""
import math
from awareness import Awareness,Encounter,MAX_AGE_NS,MAX_HISTORY_NS,VRUS,normalise,eligibility,side,camera_identity_role
from awareness_motion import evidence_qualified as qualified

PLURAL={'person':'PEOPLE','bicycle':'BICYCLES','cyclist':'CYCLISTS','motorcycle':'MOTORCYCLES','scooter':'SCOOTERS','animal':'ANIMALS'}
def camera(o,now):
    n,why=normalise(o,now)
    if why or eligibility(n) or n['label'] not in VRUS or n.get('sensor') not in {'camera:narrow','camera:wide'} or n.get('source_frame_valid') is not True:return None
    b=n.get('bbox_xyxy');size=n.get('image_size')
    if (not isinstance(b,(list,tuple)) or len(b)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in b)
        or not isinstance(size,(list,tuple)) or len(size)!=2 or not all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in size)
        or not 0<=b[0]<b[2]<=size[0] or not 0<=b[1]<b[3]<=size[1]):return None
    return n
def near(a,b):
    if a['sensor']!=b['sensor'] or a['label']!=b['label'] or side(a['bearing_deg'])!=side(b['bearing_deg']) or a['image_size']!=b['image_size']:return False
    x,y=a['bbox_xyxy'],b['bbox_xyxy'];h=min(x[3]-x[1],y[3]-y[1])
    if max(x[3]-x[1],y[3]-y[1])>2*h or abs(x[3]-y[3])>.5*h:return False
    if max(0,max(x[0],y[0])-min(x[2],y[2]))>h or abs(a['bearing_deg']-b['bearing_deg'])>8:return False
    r,s=a.get('range_m'),b.get('range_m')
    if (r is None)!=(s is None):return False
    p,q=a.get('position_m'),b.get('position_m')
    if (p is None)!=(q is None):return False
    if p is not None and math.hypot(p[0]-q[0],p[1]-q[1])>3:return False
    return r is None or abs(r-s)<=3
def distinct(values):
    result=[]
    for o in sorted(values,key=lambda v:(v['bbox_xyxy'][2]-v['bbox_xyxy'][0])*(v['bbox_xyxy'][3]-v['bbox_xyxy'][1]),reverse=True):
        a=o['bbox_xyxy']
        if all(min(a[2],b['bbox_xyxy'][2])<=max(a[0],b['bbox_xyxy'][0]) or min(a[3],b['bbox_xyxy'][3])<=max(a[1],b['bbox_xyxy'][1]) for b in result):result.append(o)
    return result
def band(values):
    hs=[v['bbox_xyxy'][3]-v['bbox_xyxy'][1] for v in values];feet=[v['bbox_xyxy'][3] for v in values];angles=[v['bearing_deg'] for v in values]
    return max(hs)<=2*min(hs) and max(feet)-min(feet)<=.5*min(hs) and max(angles)-min(angles)<=8

def footprint(o):
    # Receipts retain the geometry actually described at submission time,
    # never mutable caller lists or dense future perception payloads.
    value={k:o.get(k) for k in ['key','sensor','label','source_ns','frame_id','bearing_deg','range_m']}
    value.update(bbox_xyxy=list(o['bbox_xyxy']),image_size=list(o['image_size']),
                 position_m=list(o['position_m']) if o.get('position_m') is not None else None)
    return value


def stationary_span(value):
    """Native-clock coverage of an already qualified stationary history."""
    native=value.get('native_source_ns');duration=value.get('duration_ns')
    if (type(native) is not int or not 0<native<2**63 or type(duration) is not int
            or not 0<duration<native):return None
    return native-duration,native

class StationaryGroupPolicy(Awareness):
    def __init__(self):
        super().__init__();self.active=False;self.epoch=0;self.receipts=[];self.current=[];self.coverage=[];self.native_anchor=None
    def reset(self):
        super().reset();self.active=False;self.epoch+=1;self.receipts=[];self.coverage=[];self.native_anchor=None
    def cluster(self,anchor):
        group=[anchor];keys={anchor['key']}
        for _ in range(len(self.current)):
            added=False
            for o in self.current:
                if o['key'] not in keys and o['source_ns']==anchor['source_ns'] and any(near(o,x) for x in group) and band(group+[o]):
                    group.append(o);keys.add(o['key']);added=True
            if not added:break
        return group
    def receipt_for(self,o):
        matches=[r for r in self.receipts if any(near(o,p) for p in r['footprints'])]
        if not matches:return None
        group=self.cluster(o)
        # A singular notice cannot silently become the first plural notice.
        if len(distinct(group))>=2 and not any(r['plural'] for r in matches):return None
        return min(matches,key=lambda r:r['spoken_ns'])
    def _match(self,o,now_ns):
        e=super()._match(o,now_ns)
        if not self.active or o['label'] not in VRUS or (e is not None and e.announced):return e
        c=camera(o,now_ns)
        if c is None:return e
        receipt=self.receipt_for(c)
        if receipt is None:return e
        if e is None:
            if len(self.encounters)>=512:return None
            e=Encounter(self.next_id,now_ns,now_ns,o['bearing_deg'],o['position_m'],o['range_m'],o['label'])
            self.next_id+=1;self.encounters[e.id]=e
        e.announced_labels.add(o['label']);e.announced=True
        if isinstance(receipt.get('delivery'),dict):
            # Preserve the actual delivered description that currently covers
            # this person. This is group coverage, not physical identity or a
            # fabricated utterance naming the new tracker key.
            e.delivery_receipt={**receipt['delivery'],
                'person_described':camera_identity_role(c)=='person',
                'coverage_key':c['key'],'coverage_source_ns':c['source_ns'],
                'stationary_scene_epoch':self.epoch,
                'scope':'Current positive presence covered by an actually delivered stationary-scene description; not an individual identity or a new utterance.'}
        self.coverage.append({'key':o['key'],'encounter_id':e.id,'frame_id':o.get('frame_id'),'sensor':o['sensor'],
            'source_ns':o['source_ns'],'coverage_at_ns':now_ns,'original_notice_ns':receipt['spoken_ns'],
            'scope':'Current nearby presence covered by a delivered stationary-scene description; not an individual identity or new utterance.'})
        return e
    def update(self,observations,now_ns,*,stationary_evidence=None,**kwargs):
        active=qualified(stationary_evidence,now_ns)
        was_active=self.active
        new_scene=False
        if active:
            span=stationary_span(stationary_evidence)
            covered=(span is not None and self.native_anchor is not None
                     and span[0]<=self.native_anchor<=span[1])
            # Missing context disables coverage immediately, but does not prove
            # that the observed scene changed. Reuse its delivered receipt only
            # after newly received history covers the entire uncertain interval.
            new_scene=((not was_active and not covered) or
                       (was_active and span is not None and self.native_anchor is not None and not covered))
            if new_scene:self.receipts=[]
            self.native_anchor=span[1] if span is not None else None
        if active!=was_active or new_scene:self.epoch+=1
        self.active=active;self.coverage=[]
        self.receipts=[r for r in self.receipts if now_ns-r['last_seen_ns']<=MAX_HISTORY_NS]
        self.current=[v for o in observations if (v:=camera(o,now_ns)) is not None]
        result=super().update(observations,now_ns,**kwargs)
        if self.active:
            ids={d['key']:d.get('encounter_id') for d in result['decisions'] if d.get('encounter_id') is not None}
            for event in result['events']:
                first=camera(event['evidence'][0],now_ns)
                if first is None:continue
                cluster=self.cluster(first)
                if len(distinct(cluster))>=2:
                    # Keep every original participant, including another camera.
                    union={(o['sensor'],o['key']):o for o in event['evidence']}
                    for o in cluster:union.setdefault((o['sensor'],o['key']),o)
                    event['evidence']=list(union.values());event['phrase']=PLURAL[event['label']]+' '+event['direction'];event['range_m']=None
                    event['encounter_ids']=sorted(set(event['encounter_ids'])|{ids[o['key']] for o in cluster if o['key'] in ids})
                    event['source_ns']=min(o['source_ns'] for o in event['evidence'])
                event['stationary_scene_epoch']=self.epoch
            # Refresh a fixed, actually announced footprint using current
            # positive observations. Do not grow a trail of excluded image space.
            for r in self.receipts:
                if any(near(o,p) for o in self.current for p in r['footprints']):r['last_seen_ns']=now_ns
        for c in self.coverage:
            for d in result['decisions']:
                if d.get('key')==c['key']:d.update(reason='covered_by_stationary_group_notice',coverage_at_ns=now_ns)
        result.update(stationary_group_coverage=list(self.coverage),stationary_scene_active=self.active,stationary_scene_epoch=self.epoch)
        return result
    def acknowledge(self,event_id,delivered,now_ns):
        event=self.pending.get(event_id);ok=super().acknowledge(event_id,delivered,now_ns)
        if ok and self.active and event.get('stationary_scene_epoch')==self.epoch:
            values=[v for o in event['evidence'] if (v:=camera(o,now_ns)) is not None]
            for sensor in sorted({o['sensor'] for o in values}):
                group=[o for o in values if o['sensor']==sensor]
                if len(self.receipts)>=64:break
                self.receipts.append({'spoken_ns':now_ns,'last_seen_ns':now_ns,'footprints':[footprint(o) for o in group],'plural':len(distinct(group))>=2,
                    'delivery':{'event_id':event['id'],'spoken_ns':now_ns,'phrase':event['phrase'],
                                'coverage_basis':'delivered_stationary_scene'}})
        return ok

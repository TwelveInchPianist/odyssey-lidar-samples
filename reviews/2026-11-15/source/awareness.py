"""Presence advisories, independent of collision-path and braking policy.

Inputs are timestamped observations, not the legacy engine's warning candidates.
Map matches never establish furniture identity. Delivery acknowledgements, rather
than proposed speech, commit encounter deduplication. No nuisance-rate mute exists.
"""
from dataclasses import dataclass, field
import math

VERSION = 'road-user-awareness-v1'
SCHEMA_VERSION = 1
MAX_AGE_NS = 300_000_000
MAX_HISTORY_NS = 10_000_000_000
SPEECH_GAP_NS = 1_500_000_000
LABELS = {'person', 'cyclist', 'bicycle', 'motorcycle', 'scooter', 'animal', 'vehicle', 'unknown', 'furniture'}
VRUS = {'person', 'cyclist', 'bicycle', 'motorcycle', 'scooter', 'animal'}
# Every requested road-user class deserves the same first-notice priority.
# A bicycle must not wait behind a newly confirmed person/motorcycle merely
# because its rider could not also be classified. Within a class tier, the
# existing measured-range and first-encounter ordering still applies.
# Vehicle eligibility already requires measured stopped motion and excludes
# positively known queues. Such a specific eligible object must not starve
# behind a continual supply of unclassified scenery. VRUs remain first.
PRIORITY = {'person': 0, 'cyclist': 0, 'motorcycle': 0, 'scooter': 0, 'bicycle': 0, 'animal': 0, 'vehicle': 1, 'unknown': 2}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_identity_gate(gap_s):
    """Shared existing positional gate for physical and description continuity."""
    return min(4., 1.0 + 3.0 * gap_s)


def compensated_position(position, dx, dy, cosine, sine):
    x,y=position[0]-dx,position[1]-dy
    return [cosine*x+sine*y,-sine*x+cosine*y]


def side(bearing_deg):
    return 'LEFT' if bearing_deg > 5 else 'RIGHT' if bearing_deg < -5 else 'AHEAD'


def normalise(value, now_ns):
    """Reject malformed/stale input rather than invent a position or class."""
    if not isinstance(value, dict):
        return None, 'malformed_observation'
    key = value.get('key')
    stamp, receipt = value.get('source_ns'), value.get('receipt_ns')
    if not isinstance(key, str) or not key or len(key) > 200:
        return None, 'invalid_identity'
    if type(stamp) is not int or type(receipt) is not int or not 0 < stamp <= receipt <= now_ns:
        return None, 'invalid_or_future_timestamp'
    if now_ns - stamp > MAX_AGE_NS:
        return None, 'stale_observation'
    if value.get('valid') is not True:
        return None, 'source_observation_invalid'
    if 'semantic_source_ns' in value:
        semantic=value['semantic_source_ns']
        if type(semantic) is not int or not 0<semantic<=stamp:return None,'invalid_semantic_timestamp'
    bearing = value.get('bearing_deg')
    if not finite(bearing) or abs(bearing) >= 90:
        return None, 'outside_forward_observation_area'
    label = value.get('label', 'unknown')
    if label not in LABELS:
        return None, 'unsupported_class'
    if label != 'unknown' and value.get('class_confirmed') is not True:
        label = 'unknown'
    position = value.get('position_m')
    if position is not None:
        if (not isinstance(position, (list, tuple)) or len(position) != 2
                or not all(finite(x) for x in position) or position[0] < 0):
            return None, 'invalid_position'
        position = [float(x) for x in position]
    distance = value.get('range_m')
    if distance is not None and (not finite(distance) or distance <= 0):
        return None, 'invalid_range'
    support, age = value.get('support', 0), value.get('history_s', 0.)
    if type(support) is not int or support < 0 or not finite(age) or age < 0:
        return None, 'invalid_persistence'
    return {**value, 'label': label, 'position_m': position,
            'range_m': float(distance) if distance is not None else None,
            'bearing_deg': float(bearing)}, None


def eligibility(observation):
    """A stationary or parallel VRU needs no collision or motion hypothesis."""
    o = observation
    if o['support'] < 3 or o['history_s'] < .2:
        return 'awaiting_observation_persistence'
    label = o['label']
    if o.get('vru_evidence') is True and label == 'furniture':
        return None  # conflicting furniture evidence must not hide a road user
    if label == 'furniture':
        if o.get('furniture_positive') is True and o.get('furniture_live_evidence') is True:
            return 'positively_identified_furniture'
        return None  # a map match or stationary velocity is not identification
    if label == 'vehicle':
        velocity = o.get('ground_speed_mps')
        if o.get('motion_confirmed') is not True or not finite(velocity):
            return 'vehicle_motion_unknown'
        if abs(velocity) > .75:
            return 'ordinary_moving_vehicle'
        if o.get('queue_confirmed') is True:
            return 'clear_stopped_queue'
    return None


def observed_description_keys(observation):
    """Keys explicitly covered by a confirmed rider description in one camera.

    This is the existing persistent, unique image relationship, not cross-sensor
    fusion. Arbitrary components, stale frames or an unconfirmed relationship
    cannot manufacture an announcement alias.
    """
    keys={observation['key']}
    parts=observation.get('components')
    expected={'cyclist':{'person','bicycle'},'motorcycle':{'person','motorcycle'},
              'scooter':{'person','scooter'}}.get(observation.get('label'))
    if (expected is None or observation.get('class_confirmed') is not True
            or type(observation.get('relationship_samples')) is not int or observation['relationship_samples']<3
            or not isinstance(parts,list) or len(parts)!=2 or not all(isinstance(p,dict) for p in parts)):
        return keys
    if {p.get('label') for p in parts}!=expected:return keys
    for part in parts:
        key=part.get('key')
        if (not isinstance(key,str) or not key or len(key)>200
                or part.get('sensor')!=observation.get('sensor')
                or not str(part.get('sensor','')).startswith('camera:')
                or part.get('source_ns')!=observation.get('source_ns')
                or part.get('valid') is not True or part.get('class_confirmed') is not True):
            return keys
    if not any(p['label']=='person' and p['key']==observation['key'] for p in parts):return keys
    return keys|{p['key'] for p in parts}

def observed_identity_keys(observation):
    """A rider description does not make its bicycle the same physical person."""
    return {observation['key']}


def camera_identity_role(observation):
    if observation.get('sensor')=='shared':
        description=shared_camera_description(observation)
        return camera_identity_role(description) if description is not None else None
    if observation.get('sensor') not in {'camera:narrow','camera:wide'}:
        return None
    if (observation.get('label') in {'person','cyclist'}
            or len(observed_description_keys(observation)) > 1):
        return 'person'
    if observation.get('label') in {'bicycle','motorcycle','scooter'}:
        return 'two_wheeler'
    return None


def shared_camera_description(observation):
    """Preserve a specific image description, without merging its components."""
    d=observation.get('camera_description')
    if (observation.get('sensor')!='shared' or not isinstance(d,dict)
        or d.get('sensor') not in {'camera:narrow','camera:wide'}
        or d.get('class_confirmed') is not True or d.get('label')!=observation.get('label')):return None
    if not any(e.get('key')==d.get('key') and e.get('source_ns')==d.get('source_ns') for e in observation.get('source_evidence',[]) if isinstance(e,dict)):return None
    return d


@dataclass
class Encounter:
    id: int
    first_ns: int
    last_ns: int
    bearing_deg: float
    position_m: object
    range_m: object
    label: str
    keys: set = field(default_factory=set)
    announced: bool = False
    announced_labels: set = field(default_factory=set)
    label_ns: int = 0
    last_observation: dict = field(default_factory=dict)
    identity_source_ns: int = 0
    delivery_receipt: dict = field(default_factory=dict)


class Awareness:
    def __init__(self):
        self.encounters = {}
        self.next_id = 1
        self.last_update_ns = None
        self.last_spoken_ns = -SPEECH_GAP_NS
        self.pending = {}
        self.merged_ids = {}
        self.component_receipts = {}

    def reset(self):
        self.encounters.clear()
        self.pending.clear()
        self.merged_ids.clear()
        self.component_receipts.clear()
        self.last_update_ns = None
        # Keep increasing IDs: a recording never reuses an encounter identity.

    @staticmethod
    def _compatible(a, b):
        return a == b or 'unknown' in (a, b) or ({a, b} <= {'person', 'cyclist', 'bicycle', 'motorcycle', 'scooter'})

    def _canonical_id(self,key):
        while key in self.merged_ids:key=self.merged_ids[key]
        return key

    def _observe_identity(self, observations, now_ns):
        """Maintain known camera identity without promoting weak class evidence.

        Only an existing exact key can stay alive, through a continuous sequence
        of fresh, valid image measurements. No key association, position/class
        update, new encounter or speech eligibility is created here.
        """
        from awareness_vision import MAX_GAP_NS
        updated=[]
        for o in observations:
            if not isinstance(o,dict) or o.get('source_frame_valid') is not True:continue
            sensor=o.get('sensor');key=o.get('key');stamp=o.get('source_ns');receipt=o.get('receipt_ns')
            label=o.get('label')
            shared=(sensor=='shared' and o.get('association_domain')==key and o.get('identity_continuity')=='observed_sensor_tracks')
            if (not isinstance(sensor,str) or (sensor not in {'camera:narrow','camera:wide'} and not shared)
                    or not isinstance(key,str) or not isinstance(label,str)):continue
            if (type(stamp) is not int or type(receipt) is not int or
                    not 0<stamp<=receipt<=now_ns or now_ns-stamp>MAX_AGE_NS):continue
            box=o.get('bbox_xyxy');shape=o.get('image_size')
            if (not isinstance(box,(list,tuple)) or len(box)!=4 or not all(finite(x) for x in box)
                    or not isinstance(shape,(list,tuple)) or len(shape)!=2 or
                    not all(finite(x) and x>0 for x in shape)
                    or not 0<=box[0]<box[2]<=shape[0] or not 0<=box[1]<box[3]<=shape[1]):continue
            for e in self.encounters.values():
                same_family=(label==e.label or
                    {label,e.label}<={'person','cyclist','bicycle','motorcycle','scooter'})
                if (key in e.keys and sensor==e.last_observation.get('sensor') and same_family
                        and 0<stamp-e.identity_source_ns<=MAX_GAP_NS):
                    e.identity_source_ns=stamp
                    updated.append({'encounter_id':e.id,'key':key,'source_ns':stamp,
                                    'class_promoted':False,'new_identity_association':False})
        return updated


    def _retain_component_observations(self, observations, now_ns):
        # Only existing described keys can stay live through weak but actual
        # measurements. This does not pair new objects or confirm their class.
        from awareness_vision import MAX_GAP_NS
        self.component_receipts={key:r for key,r in self.component_receipts.items()
                                 if 0<=now_ns-r['last_seen_ns']<=MAX_HISTORY_NS}
        for o in observations:
            if not isinstance(o,dict) or o.get('source_frame_valid') is not True:
                continue
            if not isinstance(o.get('sensor'),str) or not isinstance(o.get('key'),str):continue
            key=(o.get('sensor'),o.get('key'));r=self.component_receipts.get(key)
            t=o.get('source_ns');receipt=o.get('receipt_ns')
            if (r is None or camera_identity_role(o)!='two_wheeler'
                    or type(t) is not int or type(receipt) is not int
                    or not 0<t<=receipt<=now_ns or now_ns-t>MAX_AGE_NS
                    or not 0<t-r['last_seen_ns']<=MAX_GAP_NS):continue
            box=o.get('bbox_xyxy');shape=o.get('image_size')
            if (not isinstance(box,(list,tuple)) or len(box)!=4
                    or not all(finite(x) for x in box)
                    or not isinstance(shape,(list,tuple)) or len(shape)!=2
                    or not all(finite(x) and x>0 for x in shape)
                    or not 0<=box[0]<box[2]<=shape[0]
                    or not 0<=box[1]<box[3]<=shape[1]):continue
            r['last_seen_ns']=t

    def _extend_delivered_description(self, observation, encounter, now_ns):
        if observation.get('sensor')=='shared':
            description=shared_camera_description(observation)
            if description is None:return
            # A measured person/rider centroid is not a separately measured
            # bicycle position. Keep the original camera component provenance.
            observation=description
        # A fresh confirmed pair can explain a component of a person already
        # described. The reverse is deliberately invalid: a bike-only notice
        # never describes a person who was not in that delivered notice.
        if (encounter.delivery_receipt.get('person_described') is not True
                or camera_identity_role(observation)!='person'
                or len(observed_description_keys(observation))<2):return
        t=observation['source_ns']
        if not 0<=now_ns-t<=MAX_AGE_NS:return
        for part in observation['components']:
            if part['label']=='person':continue
            key=(part['sensor'],part['key'])
            if key not in self.component_receipts and len(self.component_receipts)>=512:
                continue
            self.component_receipts[key]={
                'sensor':part['sensor'],'key':part['key'],
                'person_key':observation['key'],'relation_source_ns':t,
                'relationship_samples':observation['relationship_samples'],
                'last_seen_ns':t,'delivery':dict(encounter.delivery_receipt),
                'metric_reference_m':list(observation['position_m']) if observation['position_m'] is not None else None,
                'metric_reference_source_ns':t if observation['position_m'] is not None else None,
                'metric_reference_kind':'described_object_measurement' if observation['position_m'] is not None else 'unavailable',
                'scope':'Component covered by a delivered person/rider description; physical identities remain separate.'}

    def _component_coverage(self, observation, now_ns):
        if camera_identity_role(observation)!='two_wheeler':return None
        key=(observation.get('sensor'),observation.get('key'))
        r=self.component_receipts.get(key)
        # Reuse the existing three-second encounter grace. Fresh measurements,
        # never repeated evaluation time, carry a known described component.
        if r is None or not 0<=observation['source_ns']-r['last_seen_ns']<=3_000_000_000:
            return None
        reference=r['metric_reference_m'];measured=observation['position_m']
        if reference is not None and measured is not None:
            gap=(now_ns-r['metric_reference_source_ns'])/1e9
            if math.hypot(reference[0]-measured[0],reference[1]-measured[1])>metric_identity_gate(gap):
                # Positive measured disagreement invalidates old description
                # coverage just as it rejects an encounter identity match.
                self.component_receipts.pop(key,None)
                return None
        r['last_seen_ns']=observation['source_ns']
        if measured is not None:
            r['metric_reference_m']=list(measured)
            r['metric_reference_source_ns']=observation['source_ns']
            r['metric_reference_kind']='component_measurement'
        result={**r,'delivery':dict(r['delivery'])}
        if r['metric_reference_m'] is not None:result['metric_reference_m']=list(r['metric_reference_m'])
        return result

    def _match(self, observation, now_ns):
        choices = []
        identities=observed_identity_keys(observation)
        from visual_identity import peer_keys
        visual_peers=peer_keys(observation,identities)
        for e in self.encounters.values():
            gap = (now_ns - e.last_ns) / 1e9
            previous = e.last_observation
            # Separate physical hypotheses cannot inherit each other's delivery
            # receipts through a nearby position or a reused key. Untagged
            # legacy observations retain their existing association behavior.
            if previous.get('association_domain') != observation.get('association_domain'):
                continue
            roles=(camera_identity_role(observation),camera_identity_role(previous))
            if all(roles) and roles[0]!=roles[1]:continue
            same_key = bool(identities & e.keys)
            shared_identity=(observation.get('sensor')=='shared' and observation.get('association_domain')==observation.get('key')
                and observation.get('identity_continuity')=='observed_sensor_tracks')
            continuous_identity=(same_key and (str(observation.get('sensor','')).startswith('camera:') or shared_identity)
                and observation.get('sensor')==previous.get('sensor')
                and 0<=now_ns-e.identity_source_ns<=MAX_AGE_NS)
            if (gap > 3 and not continuous_identity) or not self._compatible(e.label, observation['label']):
                continue
            if (previous.get('sensor') == observation.get('sensor')
                    and previous.get('source_ns') == observation['source_ns']
                    and previous.get('key') != observation['key'] and not same_key):
                continue  # distinct objects observed together are not ID churn
            if (visual_peers & e.keys and observation.get('visual_identity_epoch')
                    == previous.get('visual_identity_epoch')):
                # Specific current person-image features support this camera
                # handoff. They add no class, range or first-notice prerequisite.
                # Only actually observed keys are stored below, not arbitrary
                # foreign aliases from an unavailable/stale relationship.
                same_key=True
            if (same_key and (str(observation.get('sensor','')).startswith('camera:') or shared_identity)
                    and (e.position_m is None or observation['position_m'] is None)):
                # A confirmed continuing camera track can cross many degrees
                # during a confidence gap. Its ID, rather than the old bearing,
                # owns the announcement. Unrelated keys still need evidence;
                # contradictory measured positions still use the metric gate.
                choices.append((-3.,e))
                continue
            p, q = e.position_m, observation['position_m']
            if p is not None and q is not None:
                residual = math.hypot(p[0]-q[0], p[1]-q[1])
                gate = metric_identity_gate(gap)
                if residual > gate:
                    continue
                cost = residual / gate
            else:
                if not same_key:
                    # Equal bearing is not a cross-sensor identity: a roadside
                    # person and a distant object can lie on the same ray. Until
                    # metric ownership exists they keep separate announcements.
                    sensor=observation.get('sensor','')
                    if sensor!=previous.get('sensor') or not sensor.startswith('camera:'):
                        continue
                    # For same-camera tracker churn, require matching image
                    # extent as well as direction and a short gap. A new small
                    # person behind an old large one cannot inherit its cooldown.
                    a=previous.get('association_bbox_xyxy',previous.get('bbox_xyxy'))
                    b=observation.get('association_bbox_xyxy',observation.get('bbox_xyxy'))
                    if any(not isinstance(box,(list,tuple)) or len(box)!=4 or
                           not all(finite(x) for x in box) or box[0]>=box[2] or box[1]>=box[3]
                           for box in (a,b)):
                        continue
                    from awareness_vision import iou
                    if iou(a,b)<.5:continue
                residual = abs(e.bearing_deg-observation['bearing_deg'])
                if residual > 2.5:
                    continue
                # Without range, bridge only a short gap. No long-range spatial
                # exclusion cone can hide a different road user.
                # A continuing tracker identity is stronger evidence than a
                # bearing-only handoff. Brief confidence flicker must not create
                # a fresh encounter for the same still-tracked road user.
                if gap > .5 and observation['key'] not in e.keys:
                    continue
                cost = residual / 2.5
            choices.append((cost - (2 if same_key else 0), e))
        choices.sort(key=lambda row: (row[0], row[1].id))
        if not choices:
            return None
        exact=[e for cost,e in choices if cost==-3.]
        if exact:
            # Merge only identities supported by an actual shared key or
            # current cross-camera person-image evidence. A rider relationship
            # covers a description, not a union of physical person/bike identities.
            chosen=min(exact,key=lambda e:e.id)
            for other in exact:
                if other.id==chosen.id:continue
                chosen.keys.update(other.keys);chosen.announced_labels.update(other.announced_labels)
                chosen.first_ns=min(chosen.first_ns,other.first_ns)
                chosen.last_ns=max(chosen.last_ns,other.last_ns)
                chosen.identity_source_ns=max(chosen.identity_source_ns,other.identity_source_ns)
                if not chosen.delivery_receipt:chosen.delivery_receipt=dict(other.delivery_receipt)
                self.merged_ids[other.id]=chosen.id
                self.encounters.pop(other.id,None)
            if len(exact)>1:
                self.pending={key:{**event,'encounter_ids':sorted({self._canonical_id(i) for i in event['encounter_ids']})}
                              for key,event in self.pending.items()}
            return chosen
        if len(choices) > 1 and choices[1][0] - choices[0][0] < .15:
            return None  # ambiguous association never inherits a speech veto
        return choices[0][1]

    def update(self, observations, now_ns, *, ego_delta=None, audio_idle=True, active_speech_label=None):
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError('Positive integer monotonic decision timestamp required')
        if self.last_update_ns is not None and now_ns <= self.last_update_ns:
            self.reset()
        dt = 0 if self.last_update_ns is None else (now_ns-self.last_update_ns)/1e9
        self.last_update_ns = now_ns
        self.merged_ids = {}
        if ego_delta is not None:
            if len(ego_delta) != 3 or not all(finite(x) for x in ego_delta):
                raise ValueError('Invalid measured ego delta')
            dx, dy, yaw = ego_delta
            c, s = math.cos(yaw), math.sin(yaw)
            for e in self.encounters.values():
                if e.position_m is not None:
                    e.position_m = compensated_position(e.position_m,dx,dy,c,s)
                    e.bearing_deg = math.degrees(math.atan2(e.position_m[1], e.position_m[0]))
            for receipt in self.component_receipts.values():
                if receipt['metric_reference_m'] is not None:
                    receipt['metric_reference_m']=compensated_position(receipt['metric_reference_m'],dx,dy,c,s)
        elif dt > .5:
            # Identity and very recent bearing may still match; don't pretend
            # an old body-frame location is current without ego motion.
            for e in self.encounters.values():
                e.position_m = None
            for receipt in self.component_receipts.values():
                receipt['metric_reference_m']=None
                receipt['metric_reference_source_ns']=None
                receipt['metric_reference_kind']='unavailable'
        self._retain_component_observations(observations,now_ns)
        identity_updates=self._observe_identity(observations,now_ns)
        self.encounters = {k:e for k,e in self.encounters.items()
                           if now_ns-max(e.last_ns,e.identity_source_ns) <= MAX_HISTORY_NS}
        self.pending = {k:v for k,v in self.pending.items()
                        if now_ns-v['at_ns'] <= MAX_AGE_NS}
        decisions, visible, seen_keys = [], {}, set()
        for value in observations:
            o, reason = normalise(value, now_ns)
            if reason:
                decisions.append({'key': value.get('key') if isinstance(value,dict) else None,
                                  'eligible':False, 'reason':reason})
                continue
            if o['key'] in seen_keys:
                decisions.append({'key':o['key'],'eligible':False,'reason':'duplicate_observation'})
                continue
            seen_keys.add(o['key'])
            reason = eligibility(o)
            if reason:
                decisions.append({'key':o['key'],'eligible':False,'reason':reason,
                                  'label':o['label'],'source_ns':o['source_ns']})
                continue
            if o['label'] == 'furniture':
                o = {**o, 'label':'unknown'}
            e = self._match(o, now_ns)
            if e is None:
                if len(self.encounters) >= 512:
                    decisions.append({'key':o['key'],'eligible':False,'reason':'encounter_capacity_exceeded'})
                    continue
                e = Encounter(self.next_id, now_ns, now_ns, o['bearing_deg'],
                              o['position_m'], o['range_m'], o['label'])
                self.next_id += 1
                self.encounters[e.id] = e
            e.last_ns = now_ns
            e.identity_source_ns=max(e.identity_source_ns,o['source_ns'])
            e.keys.update(observed_identity_keys(o))
            if len(e.keys) > 32:
                e.keys = {o['key']}
            e.bearing_deg = o['bearing_deg']
            if o['position_m'] is not None:e.position_m = o['position_m']
            if o['range_m'] is not None:e.range_m = o['range_m']
            if o['label'] != 'unknown':
                e.label = o['label'];e.label_ns = o.get('semantic_source_ns',o['source_ns']);e.last_observation = o
            elif e.label == 'unknown' or now_ns-e.label_ns>MAX_AGE_NS:
                e.label = 'unknown';e.last_observation = o
            # A generic announcement cannot silence newly positive road-user
            # evidence. Once that road user was announced, classifier flicker
            # must not re-arm it on every reappearance.
            e.announced=bool(e.announced_labels & VRUS) if e.label in VRUS else bool(e.announced_labels)
            self._extend_delivered_description(o,e,now_ns)
            component_coverage=self._component_coverage(o,now_ns)
            if component_coverage is not None:e.announced=True
            visible[e.id] = e
            decisions.append({'key':o['key'],'encounter_id':e.id,'eligible':not e.announced,
                              'reason':'covered_by_delivered_rider_description' if component_coverage is not None else 'already_announced' if e.announced else 'observed_presence',
                              'description_coverage':component_coverage,
                              'label':e.label,'source_ns':o['source_ns'],
                              'map_match':o.get('map_match'),'map_suppressed':False})
        for decision in decisions:
            key=decision.get('encounter_id')
            if key in self.merged_ids:
                decision['merged_from_encounter_id']=key
                decision['encounter_id']=self._canonical_id(key)
        visible={self._canonical_id(key):self.encounters[self._canonical_id(key)] for key in visible}
        fresh = [e for e in visible.values() if not e.announced]
        events = []
        if fresh:
            chosen = min(fresh, key=lambda e:(PRIORITY[e.label], e.last_observation.get('range_m')
                                             if e.last_observation.get('range_m') is not None else 1e6, e.id))
            # Only an explicitly identified lower-priority awareness utterance
            # may be interrupted. Unknown external audio or collision speech
            # is never assumed to be interruptible.
            upgrade=(isinstance(active_speech_label,str) and active_speech_label in PRIORITY and PRIORITY[chosen.label]<PRIORITY[active_speech_label])
            if not ((audio_idle and now_ns-self.last_spoken_ns>=SPEECH_GAP_NS) or upgrade):
                return {'version':VERSION,'at_ns':now_ns,'observations':len(observations),'decisions':decisions,'events':[],
                        'active_encounters':len(visible),'identity_updates':identity_updates,'map_warning_override':False}
            direction = side(chosen.bearing_deg)
            group = [e for e in fresh if e.label == chosen.label and side(e.bearing_deg) == direction
                     and abs(e.bearing_deg-chosen.bearing_deg) <= 8]
            noun = {'person':('PERSON','PEOPLE'),'cyclist':('CYCLIST','CYCLISTS'),
                    'bicycle':('BICYCLE','BICYCLES'),
                    'motorcycle':('MOTORCYCLE','MOTORCYCLES'),'scooter':('SCOOTER','SCOOTERS'),
                    'animal':('ANIMAL','ANIMALS'),'vehicle':('STOPPED VEHICLE','STOPPED VEHICLES'),
                    'unknown':('OBJECT','OBJECTS')}[chosen.label][len(group)>1]
            event_id = 'aware-%d-%d' % (chosen.id, now_ns)
            event = {'schema_version':SCHEMA_VERSION,'id':event_id,'at_ns':now_ns,
                     'source_ns':min(e.last_observation['source_ns'] for e in group),
                     'encounter_ids':sorted(e.id for e in group),'kind':'awareness',
                     'label':chosen.label,'phrase':noun+' '+direction,'direction':direction,
                     # Cross-sensor speech deduplication is not metric fusion.
                     # A camera-only person cannot inherit a nearby tree's range.
                     'range_m':chosen.last_observation.get('range_m') if len(group)==1 else None,
                     'evidence':[e.last_observation for e in group],
                     'collision_assessment':False,'delivery':'proposed'}
            if upgrade and not audio_idle:event['interrupts_awareness_label']=active_speech_label
            self.pending[event_id] = event
            events.append(event)
        return {'version':VERSION,'at_ns':now_ns,'observations':len(observations),
                'decisions':decisions,'events':events,'active_encounters':len(visible),
                'identity_updates':identity_updates,
                'map_warning_override':False}

    def acknowledge(self, event_id, delivered, now_ns):
        event = self.pending.pop(event_id, None)
        if event is None or not delivered:
            return False
        if not event['source_ns'] <= now_ns <= event['source_ns'] + 500_000_000:
            return False
        for key in event['encounter_ids']:
            if key in self.encounters:
                self.encounters[key].announced = True
                self.encounters[key].announced_labels.add(event['label'])
                self.encounters[key].delivery_receipt={'event_id':event_id,'spoken_ns':now_ns,'phrase':event['phrase'],'label':event['label'],
                    'coverage_basis':'delivered_observation',
                    'person_described':any(o['key'] in self.encounters[key].keys and camera_identity_role(o)=='person' and o.get('class_confirmed') is True for o in event['evidence'])}
        for observation in event['evidence']:
            for encounter in self.encounters.values():
                if observation['key'] in encounter.keys and encounter.delivery_receipt.get('event_id')==event_id:
                    self._extend_delivered_description(observation,encounter,now_ns)
                    current=encounter.last_observation
                    if current.get('key') in encounter.keys:
                        # A relation may have arrived while the original person
                        # notice was pending. Only already received, still-fresh
                        # evidence can extend the now delivered description.
                        self._extend_delivered_description(current,encounter,now_ns)
        self.last_spoken_ns = now_ns
        return True

    def interrupted(self,event_id):
        """Withdraw only the interrupted submission's receipt, never queue it."""
        changed=0
        for e in self.encounters.values():
            if e.delivery_receipt.get('event_id')==event_id:
                e.announced_labels.discard(e.delivery_receipt.get('label'))
                e.delivery_receipt={};e.announced=bool(e.announced_labels & VRUS) if e.label in VRUS else bool(e.announced_labels);changed+=1
        self.component_receipts={k:v for k,v in self.component_receipts.items() if v['delivery'].get('event_id')!=event_id}
        return changed

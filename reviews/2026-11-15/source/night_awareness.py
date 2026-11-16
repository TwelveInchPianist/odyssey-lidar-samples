"""Experimental nighttime speech gate around the shared awareness policy.

Does not alter or consume FCW decisions. Caller supplies measured light state or
an explicitly identified astronomical/manual state; image brightness alone is
not lux. Delivery is a separate acknowledgement, never assumed from proposal.
"""
from awareness import Awareness

class NightAwareness:
    def __init__(self):
        self.policy=Awareness()
        self.enabled=False
        self.transition_ns=None
        self.desired=None
        self.last_ns=None
        self.pending={}

    def _light(self, evidence, now):
        if not isinstance(evidence,dict):return None,'missing_light_evidence'
        kind=evidence.get('basis');stamp=evidence.get('source_ns')
        if kind not in ('measured_ambient','solar_elevation','manual','replay_override'):
            return None,'unsupported_light_basis'
        if type(stamp) is not int or not 0<stamp<=now or now-stamp>30_000_000_000:
            return None,'stale_or_future_light_evidence'
        if type(evidence.get('dark')) is not bool:return None,'unknown_light_state'
        return evidence['dark'],kind

    def update(self, observations, now_ns, *, light, ego_delta=None,
               audio_idle=True, fcw_active=False, active_speech_label=None, context=None):
        if type(now_ns) is not int or now_ns<=0:raise ValueError('Monotonic time required')
        if self.last_ns is not None and now_ns<=self.last_ns:
            self.policy.reset();self.enabled=False;self.desired=None;self.transition_ns=None;self.pending.clear()
        self.last_ns=now_ns
        desired,basis=self._light(light,now_ns)
        if desired is None:
            # Unavailable light information does not disable geometric FCW.
            self.enabled=False;self.desired=None;self.transition_ns=None
        else:
            if desired!=self.desired:self.desired=desired;self.transition_ns=now_ns
            dwell=0 if basis in ('manual','replay_override') else (5_000_000_000 if desired else 30_000_000_000)
            if now_ns-self.transition_ns>=dwell:self.enabled=desired
        permitted=self.enabled and audio_idle and not fcw_active
        interruptible=active_speech_label if self.enabled and not fcw_active else None
        result=self.policy.update(observations,now_ns,ego_delta=ego_delta,audio_idle=permitted,active_speech_label=interruptible)
        events=[e for e in result['events'] if permitted or (self.enabled and not fcw_active and e.get('interrupts_awareness_label') is not None)]
        for event in events:
            event['night_mode']={'enabled':self.enabled,'light':dict(light),'experimental':True}
            event['review_context']=dict(context or {})
            event['review_context'].setdefault('camera','unavailable')
            event['review_context'].setdefault('lidar','unavailable')
            event['delivery']='proposed'
            self.pending[event['id']]=event
        self.pending={k:v for k,v in self.pending.items() if now_ns-v['at_ns']<=500_000_000}
        return {**result,'events':events,'night_enabled':self.enabled,'light_basis':basis,
                'fcw_priority_block':bool(fcw_active),'geometric_fcw_changed':False}

    def acknowledge(self,event_id,delivered,now_ns):
        event=self.pending.pop(event_id,None)
        if event is None:return False
        ok=self.policy.acknowledge(event_id,delivered,now_ns)
        event['delivery']='accepted' if ok else 'not_delivered'
        event['delivery_ns']=now_ns
        return ok

    def interrupted(self,event_id):
        return self.policy.interrupted(event_id)

"""Per-camera freshness gate. Does not own FCW or encounter/announcement state.

Scheduler must honor submit_after_ns and cancel_worker; this component alone
cannot stop a GPU process. Thresholds match the existing 300 ms observation
contract, not a new claim that 300 ms is acceptable for every hazard.
"""
class CameraHealth:
    MAX_AGE_NS=300_000_000
    STALL_NS=1_000_000_000
    PROBE_GAP_NS=1_000_000_000
    COOLDOWN_NS=2_000_000_000
    def __init__(self):
        self.state='probing';self.good=0;self.first_good=None
        self.last_source=None;self.last_result=None;self.submit_after_ns=0
        self.pending=None;self.events=[];self.last_now=None

    def _transition(self,state,reason,now):
        if self.state!=state:self.events.append({'at_ns':now,'from':self.state,'to':state,'reason':reason,'spoken':False})
        self.state=state

    def tick(self,now,*,fcw_pressure=False):
        if type(now) is not int or now<=0:raise ValueError('Monotonic timestamp required')
        cancel=False
        if self.last_now is not None and now<self.last_now:
            self.pending=None;self.good=0;self.first_good=None;self.last_source=None;self.last_result=None
            self._transition('probing','clock_reset',now);self.submit_after_ns=now
        self.last_now=now
        if fcw_pressure or (self.pending is not None and now-self.pending['submitted_ns']>self.STALL_NS):
            cancel=self.pending is not None;self.pending=None;self.good=0;self.first_good=None
            self._transition('suspended','fcw_pressure' if fcw_pressure else 'worker_stalled',now)
            self.submit_after_ns=now+self.COOLDOWN_NS
        elif self.state=='suspended' and now>=self.submit_after_ns:self._transition('probing','cooldown_elapsed',now)
        elif self.state=='online' and self.pending is None and (self.last_result is None or now-self.last_result>self.STALL_NS):
            self.good=0;self.first_good=None;self._transition('probing','result_stream_stalled',now)
        return {'state':self.state,'cancel_worker':cancel,'submit_after_ns':self.submit_after_ns,'fcw_changed':False}

    def submit(self,source_ns,now,token):
        self.tick(now)
        if self.pending is not None or now<self.submit_after_ns or self.state=='suspended':return False
        if type(source_ns) is not int or not 0<source_ns<=now or now-source_ns>self.MAX_AGE_NS:return False
        if self.last_source is not None and source_ns<=self.last_source:return False
        self.pending={'token':token,'source_ns':source_ns,'submitted_ns':now}
        if self.state!='online':self.submit_after_ns=now+self.PROBE_GAP_NS
        return True

    def complete(self,token,now,*,valid=True):
        self.tick(now)
        p=self.pending
        if p is None or p['token']!=token:return {'accepted':False,'reason':'obsolete_or_unknown_result'}
        self.pending=None
        fresh=valid and 0<=now-p['source_ns']<=self.MAX_AGE_NS
        self.last_source=p['source_ns'];self.last_result=now
        if not fresh:
            self.good=0;self.first_good=None;self.submit_after_ns=now+self.COOLDOWN_NS
            self._transition('suspended','late_or_invalid_result',now)
            return {'accepted':False,'reason':'late_or_invalid_result'}
        if self.first_good is None:self.first_good=now
        self.good+=1
        if self.good>=3 and now-self.first_good>=1_000_000_000:
            self._transition('online','timely_recovery_sequence',now);self.submit_after_ns=now
        return {'accepted':self.state=='online','reason':'timely' if self.state=='online' else 'recovery_probe','source_ns':p['source_ns']}

    def filter_observations(self,observations,now):
        self.tick(now)
        result=[]
        for o in observations:
            sensor=o.get('sensor','');camera=sensor.startswith('camera:') or any(str(e.get('sensor','')).startswith('camera:') for e in o.get('source_evidence',[]))
            if not camera:result.append(o);continue
            stamp=o.get('source_ns')
            if self.state=='online' and type(stamp) is int and 0<=now-stamp<=self.MAX_AGE_NS:result.append(o)
        return result

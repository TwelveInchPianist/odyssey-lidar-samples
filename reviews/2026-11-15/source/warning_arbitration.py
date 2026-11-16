"""Measured braking and bounded, distinct-hazard escalation rules."""
import math

def braking_adequate(required,measured):
    return (isinstance(required,(int,float)) and isinstance(measured,(int,float))
            and math.isfinite(required) and math.isfinite(measured)
            and required>0 and measured>=required)

def global_slot(engine,candidate,now):
    elapsed=now-engine.last_any_alert
    if elapsed>1.5:return True
    previous=getattr(engine,'last_warning',None)
    if elapsed<.35 or candidate.get('ttc_s',math.inf)>1.5 or not previous:return False
    if previous['track_id']==candidate['track_id']:return False
    # ID churn alone cannot bypass repetition control. Require a changed direction
    # or concurrent, spatially distinct tracks for two separate objects.
    distinct=candidate.get('direction')!=previous.get('direction')
    tracks=engine.tracker.tracks
    a=tracks.get(candidate['track_id']);b=tracks.get(previous['track_id'])
    if a is not None and b is not None and a.misses==0 and b.misses==0:
        distinct=distinct or math.hypot(*(a.position-b.position))>4.
    if not distinct:return False
    candidate['cooldown_override']='distinct_urgent_hazard'
    return True

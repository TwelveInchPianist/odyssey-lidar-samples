"""Independent, bounded awareness observation and speech coordinator.

No collision status, future path or driver-braking decision enters this module.
Camera and lidar updates can arrive independently. Sensor queues keep only the
latest observations; proposed speech is re-evaluated rather than queued.
"""
from collections import Counter
import json,math,threading
from pathlib import Path
import runtime_clock as clock
from awareness import MAX_AGE_NS
from awareness_scenes import StationaryGroupPolicy

SENSORS={'lidar','camera:narrow','camera:wide'}


def compact(observation):
    # Full current cloud membership remains in the shared perception frame.
    # The session's raw capture and context timestamp recover it for review.
    omit={'current_point_indices','current_points_m','components','legacy_collision_reasons'}
    result={k:v for k,v in observation.items() if k not in omit}
    parts=observation.get('components')
    if isinstance(parts,list) and len(parts)==2 and all(isinstance(part,dict) for part in parts):
        # The live policy needs the same bounded relationship evidence as
        # replay. Keep original validity/times so stale or cross-camera parts
        # cannot gain an identity alias merely by passing this boundary.
        identity_fields=('key','sensor','source_ns','label','valid','class_confirmed')
        result['components']=[{k:part.get(k) for k in identity_fields} for part in parts]
    return result


class AwarenessRuntime:
    def __init__(self,folder,deliver,audio_idle,enabled,on_error=None,clock_now=clock.monotonic_ns,on_submitted=None,
                 stationary_evidence=None):
        self.policy=StationaryGroupPolicy();self.deliver=deliver;self.audio_idle=audio_idle;self.enabled=enabled
        self.stationary_evidence=stationary_evidence or (lambda now:None)
        self.on_error=on_error or (lambda _:None);self.clock_now=clock_now;self.lock=threading.RLock()
        self.on_submitted=on_submitted
        self.wake=threading.Event();self.stop_event=threading.Event();self.thread=None
        self.latest={};self.pending={};self.status={};self.counts=Counter();self.last_result=None
        self.last_lidar_ns=None;self.next_motion=None;self.last_audio=None;self.journal_error=None
        self.log=(Path(folder)/'awareness-events.jsonl').open('w',encoding='utf-8',buffering=1)
        self.audio_log=(Path(folder)/'awareness-audio-events.jsonl').open('w',encoding='utf-8',buffering=1)

    def start(self):
        self.thread=threading.Thread(target=self._run,daemon=True,name='awareness-observer');self.thread.start()

    def offer(self,sensor,observations,source_ns,*,motion=None):
        if sensor not in SENSORS or type(source_ns) is not int or source_ns<=0:raise ValueError('Known sensor and source time required')
        if self.stop_event.is_set():return False
        if not self.lock.acquire(blocking=False):
            self.counts['offer_lock_busy']+=1;return False
        try:
            old=self.pending.get(sensor) or self.latest.get(sensor)
            if old and source_ns<=old['source_ns']:
                self.counts['duplicate_or_regressing_'+sensor]+=1;return False
            if sensor in self.pending:self.counts['superseded_'+sensor]+=1
            self.pending[sensor]={'source_ns':source_ns,'observations':[compact(o) for o in observations]}
            if sensor=='lidar':self.next_motion=motion
        finally:self.lock.release()
        self.wake.set();return True

    def camera_result(self,row):
        if row.get('kind')!='observations':
            key=str(row.get('kind','unknown'))+':'+str(row.get('camera',row.get('worker','stream')))
            with self.lock:self.status[key]=dict(row)
            return
        camera=row.get('camera');source=row.get('source_ns')
        if camera not in ('narrow','wide'):raise ValueError('Unexpected camera stream')
        observations=[]
        if row.get('fresh') is True:
            for o in row.get('observations',[]):
                observations.append({**o,'native_source_ns':row.get('native_source_ns'),
                    'stream_identity':row.get('stream_identity'),'inference_finished_ns':row.get('finished_ns'),
                    'source_interval_ns':row.get('source_interval_ns'),'clock_bound_ns':row.get('clock_bound_ns'),
                    'native_clock_domain':row.get('native_clock_domain')})
        self.offer('camera:'+camera,observations,source)
        with self.lock:self.status['camera:'+camera]={'fresh':row.get('fresh'),
            'source_ns':source,'consumer_age_ms':row.get('consumer_age_ms'),'inference_ms':row.get('inference_ms')}

    def _write(self,row):
        try:self.log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        except (OSError,ValueError,TypeError) as e:
            self.counts['journal_failures']+=1;self.journal_error=type(e).__name__+': '+str(e)
            self.on_error('Awareness journal: '+self.journal_error)

    def evaluate(self,now_ns):
        """Also used by deterministic integration tests with an explicit clock."""
        # The provider may use the recorder's telemetry lock. Query it BEFORE
        # taking the observer lock; status snapshots acquire them in that order.
        try:stationary=self.stationary_evidence(now_ns)
        except Exception as error:
            stationary=None
            with self.lock:self.counts['stationary_context_errors']+=1
            self.on_error('Awareness motion context: '+type(error).__name__+': '+str(error))
        with self.lock:
            updates=self.pending;self.pending={};self.latest.update(updates)
            motion=self.next_motion if 'lidar' in updates else None;self.next_motion=None
            delta=None
            if 'lidar' in updates:
                source=updates['lidar']['source_ns']
                if self.last_lidar_ns is not None and isinstance(motion,dict) and motion.get('valid') is True:
                    dt=(source-self.last_lidar_ns)/1e9;v=motion.get('speed_mps');yaw=motion.get('yaw_rate_rps')
                    if .025<=dt<=.5 and all(type(x) in (int,float) and math.isfinite(x) for x in (v,yaw)):
                        from fcw_engine import ego_delta
                        movement,angle=ego_delta(v,yaw,dt);delta=(*movement.tolist(),float(angle))
                self.last_lidar_ns=source
            fresh=[]
            for sensor,data in self.latest.items():
                if 0<=now_ns-data['source_ns']<=MAX_AGE_NS:fresh.extend(data['observations'])
            result=self.policy.update(fresh,now_ns,ego_delta=delta,audio_idle=self.audio_idle(now_ns),
                                      stationary_evidence=stationary)
            submissions=[]
            for event in result['events']:
                delivered=False;reason='muted'
                if self.enabled() and not self.stop_event.is_set():
                    try:
                        delivered=bool(self.deliver(event['phrase'],event['source_ns']))
                        reason='submitted_to_audio_device' if delivered else 'not_submitted'
                    except (OSError,RuntimeError,ValueError) as e:
                        reason='audio_error';self.on_error('Awareness audio: '+str(e))
                submission_ns=max(now_ns,self.clock_now())
                acknowledged=self.policy.acknowledge(event['id'],delivered,submission_ns)
                entry={**event,'delivery':reason,'acknowledged':acknowledged,'submission_ns':submission_ns,'virtual_delivery':False}
                if delivered and self.on_submitted is not None:
                    try:entry['camera_evidence']=self.on_submitted(entry)
                    except Exception as error:
                        entry['camera_evidence']={'state':'capture_failed','reason':type(error).__name__+': '+str(error)}
                        self.on_error('Camera evidence bookmark: '+str(error))
                if delivered:
                    try:self.audio_log.write(json.dumps(entry,separators=(',',':'),allow_nan=False)+'\n')
                    except (OSError,ValueError,TypeError) as error:
                        self.counts['audio_journal_failures']+=1;self.on_error('Awareness audio journal: '+str(error))
                if acknowledged:self.last_audio=entry
                submissions.append(entry);self.counts[reason]+=1
            result['events']=submissions;self.last_result=result;self.counts['updates']+=1
            self._write({'kind':'awareness_update','at_ns':now_ns,'updated_sensors':sorted(updates),
                'observations':fresh,'stationary_evidence':stationary,'result':result,'virtual_delivery':False})
            return result

    def _run(self):
        while not self.stop_event.is_set():
            self.wake.wait(.05);self.wake.clear()
            if self.stop_event.is_set():break
            try:self.evaluate(clock.monotonic_ns())
            except Exception as e:
                with self.lock:self.counts['observer_errors']+=1;self.status['observer_error']=type(e).__name__+': '+str(e)
                self.on_error('Awareness observer: '+type(e).__name__+': '+str(e))

    def snapshot(self,now_ns):
        with self.lock:
            return {'mode':'awareness','counts':dict(self.counts),'status':dict(self.status),
                'source_age_s':{k:max(0,(now_ns-v['source_ns'])/1e9) for k,v in self.latest.items()},
                'fresh_sources':[k for k,v in self.latest.items() if 0<=now_ns-v['source_ns']<=MAX_AGE_NS],
                'last_audio':self.last_audio,'journal_error':self.journal_error,
                'scene_selection_qualified':False,'deployment_qualified':False}

    def stop(self):
        self.stop_event.set();self.wake.set()
        if self.thread:self.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():
            self.on_error('Awareness observer did not finish closeout');return False
        self.log.close();self.audio_log.close();return True

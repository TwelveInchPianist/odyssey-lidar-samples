"""Experimental awareness beside (never instead of) the geometric FCW runtime."""
import math,threading,wave
from pathlib import Path
import runtime_clock as clock
from night_awareness import NightAwareness
from night_light import light_evidence
from lidar_presence import LidarPresence
from awareness_runtime import AwarenessRuntime
from awareness import PRIORITY
from camera_recovery import CameraRecovery

MODEL_SHA='451353a9cb5d793a2db78dd584d8ebef873960adcfc047cb53eb8cfd4f4fb48d'

class NightPolicy(NightAwareness):
    def update(self,observations,now_ns,*,stationary_evidence=None,**kwargs):
        state=stationary_evidence or {}
        return super().update(observations,now_ns,light=state.get('light'),fcw_active=state.get('fcw_active',False),active_speech_label=state.get('active_speech_label'),context=state.get('context'),**kwargs)

class NightCompanion:
    def __init__(self,session):
        self.session=session;self.presence=LidarPresence();self.video=None;self.client=None;self.evidence=None
        self.frame_context={};self.latest_camera={};self.last_error=None;self.lock=threading.Lock()
        self.pause_until_ns=0;self.camera_good={};self.camera_bad={};self.camera_live={}
        self.active_awareness=None
        self.recovery=CameraRecovery();self.camera_bad_source={}
        self.observer=AwarenessRuntime(session.folder,self.deliver,self.audio_idle,
            lambda:bool(session.config.get('audio_enabled')) and not session.stop_event.is_set(),
            self.error,on_submitted=self.bookmark,stationary_evidence=self.state_evidence)
        self.observer.policy=NightPolicy()

    def error(self,message):
        self.last_error=str(message);self.session.error('Experimental awareness: '+str(message))

    def state_evidence(self,now):
        s=self.session
        # Same lock order as snapshot: session lock before observer lock.
        with s.lock:
            light=light_evidence(s.config.get('night_awareness_mode','off'),s.telemetry.messages,now)
        with self.lock:context=dict(self.frame_context)
        if self.video:
            pause=light is None or not light.get('dark') or now<self.pause_until_ns
            changed=self.video.set_camera_paused(pause,'awareness_off' if s.config.get('night_awareness_mode','off')=='off' else 'protect_fcw_headroom')
            if changed and pause:
                self.recovery.invalidate()
                with self.lock:self.camera_live.clear();self.camera_good.clear();self.camera_bad.clear()
        active=self.active_label(now)
        return {'light':light,'fcw_active':not self.audio_idle(now) and active is None,'active_speech_label':active,'context':context}

    @staticmethod
    def phrase_label(phrase):
        noun=phrase.rsplit(' ',1)[0]
        return {'PERSON':'person','PEOPLE':'person','CYCLIST':'cyclist','CYCLISTS':'cyclist','BICYCLE':'bicycle','BICYCLES':'bicycle','MOTORCYCLE':'motorcycle','MOTORCYCLES':'motorcycle','SCOOTER':'scooter','SCOOTERS':'scooter','ANIMAL':'animal','ANIMALS':'animal','STOPPED VEHICLE':'vehicle','STOPPED VEHICLES':'vehicle','OBJECT':'unknown','OBJECTS':'unknown'}.get(noun)

    def active_label(self,now):
        active=self.active_awareness;last=self.session.speaker.last
        if now>=self.session.speaker.busy_until_ns or not active or not last:return None
        return active['label'] if last.get('at_ns')==active['speaker_at_ns'] and last.get('phrase')==active['phrase'] else None

    def camera_result(self,row):
        if row.get('kind')=='observations':
            cam=row.get('camera');fresh=row.get('fresh') is True;stamp=row.get('source_ns',0)
            with self.lock:
                if type(stamp) is int and stamp>self.camera_bad_source.get(cam,0):
                    self.camera_bad_source[cam]=stamp;self.camera_bad[cam]=0 if fresh else self.camera_bad.get(cam,0)+1
                    if self.camera_bad[cam]>=3:self.pause_until_ns=max(self.pause_until_ns,clock.monotonic_ns()+2_000_000_000)
            paused=self.video.pause_event.is_set() if self.video else True
            epoch=self.video.control_epoch.value if self.video else None
            row=self.recovery.accept(row,clock.monotonic_ns(),paused=paused,control_epoch=epoch)
        self.observer.camera_result(row)

    def audio_idle(self,now):
        return now>=self.session.speaker.busy_until_ns

    def deliver(self,phrase,source_ns):
        s=self.session
        with s.speech_lock:
            now=clock.monotonic_ns()
            label=self.phrase_label(phrase);active=self.active_label(now)
            if s.stop_event.is_set() or label is None:return False
            if not self.audio_idle(now) and not (active in PRIORITY and PRIORITY[label]<PRIORITY[active]):return False
            try:
                path=s.assets/'audio'/(phrase.replace(' ','_')+'.wav')
                with wave.open(str(path),'rb') as wav:
                    if wav.getnframes()<=0 or wav.getframerate()<=0:raise ValueError('Empty optional WAV')
                previous=self.active_awareness
                # Optional-asset errors never latch the geometric FCW audio fault.
                played=s.speaker.say(phrase,source_ns)
                if played:
                    if active is not None and previous and previous.get('event_id'):self.observer.policy.interrupted(previous['event_id'])
                    self.active_awareness={'label':label,'phrase':phrase,'speaker_at_ns':s.speaker.last['at_ns'],'event_id':None}
                return played
            except (OSError,RuntimeError,ValueError,wave.Error,EOFError) as error:
                self.error('Optional awareness audio unavailable: '+str(error));return False

    def packet(self,meta,payload):
        cam={'roadEncodeData':'narrow','wideRoadEncodeData':'wide'}.get(meta.get('service'))
        if cam:
            with self.lock:self.latest_camera[cam]=dict(meta)
        if self.evidence:self.evidence.offer(meta,payload)
        return self.video.on_packet(meta,payload) if self.video else False

    def bookmark(self,event):
        if self.active_awareness and self.active_awareness['phrase']==event['phrase']:
            self.active_awareness={**self.active_awareness,'event_id':event['id']}
        with self.lock:cameras=dict(self.latest_camera);frame=dict(self.frame_context)
        event['lidar_context']=frame.get('lidar','unavailable')
        event['experimental_noisy_mode']=True
        if not self.evidence:return {'state':'recorder_unavailable'}
        enriched=dict(event);enriched['evidence']=list(event['evidence'])
        now=clock.monotonic_ns()
        for cam,m in cameras.items():
            if 0<=now-m.get('source_ns',0)<=300_000_000:
                enriched['evidence'].append({'sensor':'camera:'+cam,'camera':cam,'frame_id':m['frame_id'],'source_ns':m['source_ns'],'native_source_ns':m.get('native_source_ns'),'stream_identity':m.get('stream_identity'),'role':'scene_context_only'})
        return self.evidence.bookmark(enriched)

    def start(self):
        s=self.session
        try:
            from video_evidence import VideoEvidence
            from runtime import recording_space_available
            self.evidence=VideoEvidence(s.folder,on_error=self.error,space_available=lambda n:recording_space_available(s.folder,s.config,n))
            self.evidence.start()
        except Exception as e:self.error('Native camera evidence unavailable: '+str(e))
        self.observer.start()
        if s.mode not in ('live','comma') or not s.config.get('comma_host'):return
        try:
            from video_processing import VideoProcessing
            from video_client import VideoClient
            model={'weights':str(s.assets/'models/yolo11m.onnx'),'sha256':MODEL_SHA,'provider':'DmlExecutionProvider','aggregate_inference_hz':4.0}
            if not Path(model['weights']).is_file():raise FileNotFoundError('Pinned FP32 camera model missing')
            self.video=VideoProcessing(model,self.camera_result);self.video.start()
            self.client=VideoClient(s.config,s.assets,self.packet,lambda row:self.observer.camera_result({'kind':'video_link',**row}));self.client.start()
        except Exception as e:
            self.error('Camera unavailable; geometric FCW continues: '+str(e))
            if self.client:self.client.stop()
            if self.video:self.video.stop()
            self.client=None;self.video=None

    def offer_lidar(self,result,context,quality):
        now=clock.monotonic_ns();stamp=result['at_ns']
        timing=result.get('processing_timing') or {}
        if quality.get('processing_overrun') or timing.get('engine_ms',0)>120:
            self.pause_until_ns=max(self.pause_until_ns,now+2_000_000_000)
        observations=self.presence.observations(result,context,quality,now)
        with self.lock:self.frame_context={'lidar':{'at_ns':stamp,'first_device_ns':quality.get('first_device_ns'),'last_packet_receipt_ns':quality.get('last_receipt_ns'),'track_ids':[o['key'] for o in observations]},'camera':'native encoded references recorded at advisory'}
        v=context.get('speed_mps');yaw=context.get('yaw_rate_rps')
        motion={'valid':all(type(x) in (int,float) and math.isfinite(x) for x in (v,yaw)),'speed_mps':v,'yaw_rate_rps':yaw}
        self.observer.offer('lidar',observations,stamp,motion=motion)

    def snapshot(self,now):
        return {**self.observer.snapshot(now),'mode':self.session.config.get('night_awareness_mode','off'),'experimental_noisy':True,'camera_error':self.last_error,'camera_worker':self.video.snapshot() if self.video else None,'camera_qualified_streams':self.recovery.status(),'geometric_fcw_active':True}

    def stop(self):
        if self.client:self.client.stop()
        if self.video:self.video.stop()
        self.observer.stop()
        if self.evidence:self.evidence.stop()

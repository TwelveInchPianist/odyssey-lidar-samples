"""Live reception, SSH clocks, inference, evidence recording and replay review."""
import base64
from collections import Counter,deque
from dataclasses import asdict
import datetime
import hashlib   # c099 v30d: build stamp (exe sha256)
import json
import os
from pathlib import Path
import queue
import re
import select
import shutil
import socket
import subprocess
import sys   # c099 v30d: build stamp (sys.frozen / sys.executable)
import threading
import runtime_clock as time
import uuid
import numpy as np
from m1_protocol import PcapWriter,read_pcap
from frames import FrameAssembler,decode_with_time
from telemetry import Telemetry
from fcw_engine import Engine,Mount,transform,deskew
from calibration import ground_proposal,timing_alignment
from evaluation import summarize
from registration import ScanMotion
from audio_output import Speaker,STATUS_PHRASES
from target_calibration import TargetCalibration
from policy import DEFAULT_POLICY
from offline_jobs import check_work
from radar import RadarBuffer
from camera_observations import CameraBuffer
from corridor import prepare_context
from fusion_release import FEATURE_LEVEL,capabilities
from commissioning import workflow_status
from learning.radar_live import RadarLiveForecaster

def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

APP_VERSION='0.9.17-ground-repeat'
# c099 v30d: 2026-10-21 live forensics §1 - session.json version was identical for v17/v26/v30, so the build that ran had to be
# reconstructed from deploy records and log fingerprints. APP_VERSION is deliberately unchanged (offline-status engine_version,
# /health and the review/learning queues compare it); the fork label and the exe sha256 travel in a separate 'build' stamp.
FORK_LABEL='astra-r5-longitudinal'
# Leave room above the requested reserve for buffered writes and clean closeout.
RECORDING_CLOSEOUT_BYTES=16*1024*1024
# c099 v30d: retry the comma ssh link until Stop. 2026-10-21 forensics §5: one attempt per session (ConnectTimeout 8, no retry) left
# 122812 (4.4 min driven, up to 38 mph, no warnings possible) and 115523 without comma data; the tablet had not yet joined the comma
# hotspot ('Unknown error' / 'Connection timed out').
COMMA_RETRY_FIRST_S=2.;COMMA_RETRY_MAX_S=15.;COMMA_LINK_HEALTHY_S=30.
COMMA_START_TIMEOUT_S=20.;COMMA_APPLICATION_IDLE_S=10.;COMMA_MONITOR_POLL_S=.2
DEFAULT_CONFIG={
 'sensor_ip':'192.168.1.200','bind_ip':'0.0.0.0','destination_ip':'192.168.1.102',
 'msop_port':6699,'difop_port':7788,'comma_host':'','ssh_key':'','comma_python':'auto',
 'http_port':18770,'record_limit_gb':0,'minimum_free_gb':10,
 'mount':asdict(Mount()),'lidar_time_adjustment_s':0.,'lidar_timing_bound_s':None,
 'timing_verified':False,'audio_enabled':False,'version':APP_VERSION,
 'hindsight_footprint':{'confirmed':False,'width_m':1.8,'front_m':2.,'rear_m':3.2,'center_y_m':0.},'map_enabled':True,
 'perception_policy':dict(DEFAULT_POLICY),'automatic_proposals':True,
 'mount_epoch':None,
 'self_calibration_enabled':True,
 'learning':{'enabled':True,'automatic_promotion':True,'memory_limit_mb':1024},
 'sync':{'enabled':False,'home_ssid':'','destination':''}
}

def lidar_budget_exhausted(config,written_bytes,next_bytes=0):
    """Zero means disk-space limited; a positive cap remains an optional limit."""
    limit=config['record_limit_gb']
    return limit>0 and written_bytes+next_bytes>limit*1e9


def recording_space_available(folder,config,next_bytes=0):
    return shutil.disk_usage(folder).free>config['minimum_free_gb']*1e9+RECORDING_CLOSEOUT_BYTES+next_bytes


# c099 v30d: build identity for session.json, final-state.json and /state (2026-10-21 live forensics §1). The running exe is hashed
# once per process on a daemon thread, away from the frame path; readers never wait on the hash ('hashing' until it completes).
_BUILD_STAMP={'fork':FORK_LABEL,'app_version':APP_VERSION,'frozen':False,'executable':None,'exe_sha256':None,'exe_bytes':None,'state':'not_started'}
_BUILD_LOCK=threading.Lock()

def build_stamp():
    with _BUILD_LOCK:return dict(_BUILD_STAMP)

def start_build_stamp(executable=None,frozen=None):
    """Idempotent and never raises. Returns the hashing thread (tests join it), else None."""
    try:
        frozen=getattr(sys,'frozen',False) if frozen is None else frozen
        with _BUILD_LOCK:
            if _BUILD_STAMP['state']!='not_started':return None
            if not frozen:_BUILD_STAMP['state']='not_frozen';return None
            path=Path(executable or sys.executable)
            _BUILD_STAMP.update(frozen=True,executable=str(path),state='hashing')
        def work():
            try:
                digest=hashlib.sha256();size=0
                with open(path,'rb') as stream:
                    while block:=stream.read(1024*1024):digest.update(block);size+=len(block)
                result={'exe_sha256':digest.hexdigest(),'exe_bytes':size,'state':'complete'}
                if Path(getattr(sys,'_MEIPASS','')).resolve()==(path.parent/'_internal').resolve():
                    from bundle_integrity import verify
                    result.update(verify(path))
            except Exception as e:result={'state':'failed: '+type(e).__name__+': '+str(e)[:200]}
            with _BUILD_LOCK:_BUILD_STAMP.update(result)
        t=threading.Thread(target=work,daemon=True,name='build-stamp');t.start()
        return t
    except Exception as e:
        with _BUILD_LOCK:_BUILD_STAMP['state']='failed: '+type(e).__name__+': '+str(e)[:200]
        return None


def ingest_telemetry_record(telemetry,item,receipt,clock_send=None):
    """Shared live/replay boundary; original records are preserved by the caller."""
    if not isinstance(item,dict):raise ValueError('telemetry record is not an object')
    if item.get('kind')=='pong':
        if clock_send is not None:
            if type(clock_send) is not int or any(type(item.get(k)) is not int or not 0<item[k]<2**63 for k in ('remote_receive_ns','remote_send_ns')):
                raise ValueError('invalid pong timestamps')
            telemetry.clock.add(clock_send,item['remote_receive_ns'],item['remote_send_ns'],receipt)
        return False
    if item.get('kind')=='sample' and not isinstance(item.get('service'),str):raise ValueError('sample service is not a string')
    invalidated=item.get('kind')=='sample' and not isinstance(item.get('data'),dict)
    telemetry.accept({**item,'data':{},'valid':False} if invalidated else item,receipt)
    return invalidated

def safe_telemetry_context(telemetry,at_ns):
    try:return telemetry.context(at_ns),None
    except Exception as e:
        context=Telemetry().context(at_ns)
        context['reasons'].append('malformed comma context; readiness withheld')
        return context,type(e).__name__+': '+str(e)

UNAVAILABLE_NOTICE_S=0.5   # Astra's default. Claude fork v13c: config['coverage_unavailable_notice_s'] overrides it (the tablet config sets 8.0:
                           # the night drive of 10/19 spoke 'unavailable' 50x for transient readiness losses); the tests keep the 0.5 s default.
ENGINE_THREADS=('process_frames','run_demo')   # c099 v30d: a dead frame thread (or demo frame loop) is an engine failure
# c099 v30d review: without comma data the speed falls back to 0, so forced_experiment adds 'motion too slow or turn too sharp' and the
# v12b transient rule below kept WARNING SYSTEM UNAVAILABLE silent for a whole comma outage (reviewer harness: 40 s at speed after a
# mid-drive drop, nothing spoken; 2026-10-21 122812 drove 4.4 min without the comma, one notice only during a lidar gap). While the
# speed is unknown that text is not evidence of a crawl. These markers mean 'speed unknown' (telemetry.fresh returns no carState
# while the clock is unbounded, so the second implies the first).
SPEED_UNKNOWN_MARKERS=('speed/vehicle state missing or stale','comma clock not bounded')
COMMA_START_NOTICE_S=30.   # c099 v30d review: no usable comma data since the session started -> one notice after 30 s (links take ~3 s)
ENGINE_NOTICE_REPEAT_S=60.   # c099 v30d review: while engine_failure is set the unavailable notice repeats (v30's restart loop re-announced)
ENGINE_STALL_S=10.   # c099 v30d review: scans offered but none taken by a live frame thread for this long -> engine failure (hung thread)

class CoverageMonitor:
    """Separate nominal readiness from degraded near-field warning eligibility.

    Limited notices rearm after one second of uninterrupted nominal frames;
    unavailable notices also rearm after sustained near-field recovery.
    Their timers use receipt time, and a source-time reset restarts debounce.
    This describes algorithm restrictions, never a promised detection distance.
    """
    def __init__(self,unavailable_notice_s=None):
        self.notice_s=UNAVAILABLE_NOTICE_S if unavailable_notice_s is None else float(unavailable_notice_s)
        self.current=None;self.last_at_ns=None;self.limited_since_ns=None;self.healthy_since_ns=None;self.available_since_ns=None
        self.announced=set()
        self.unavailable_since_ns=None
        self.speed_unknown_since_ns=None;self.speed_known_seen=False   # c099 v30d review: comma-outage notice (SPEED_UNKNOWN_MARKERS)

    @staticmethod
    def describe(result,stale=False):
        ground=result.get('ground') or {};source=ground.get('source','fit' if ground else 'unknown')
        eligible=result.get('status')=='eligible_experimental' and not stale
        if not eligible:
            return {'state':'unavailable','long_range_ready':False,'near_field_eligible':False,
                    'ground_source':source,'algorithm_range_limit_m':None,
                    'message':'Warnings unavailable: no fresh lidar result.' if stale else 'Warnings unavailable: '+('; '.join(result.get('reasons',[])) or 'readiness checks did not pass')}
        if ground.get('degraded') or source in ('held','mount_prior'):
            limited=source=='mount_prior'
            return {'state':'limited','long_range_ready':False,'near_field_eligible':True,
                    'ground_source':source,'algorithm_range_limit_m':25 if limited else None,
                    'message':('WARNING RANGE LIMITED — only close upright objects within a 25 m processing limit; sparse or low objects can be missed. This is not a guaranteed detection range.' if limited else
                               'WARNING RANGE LIMITED — using a recent road estimate; sparse or low objects can be missed. Long-range coverage is not established.')}
        if not ground or source!='fit':
            return {'state':'unavailable','long_range_ready':False,'near_field_eligible':False,
                    'ground_source':source,'algorithm_range_limit_m':None,'message':'Warnings unavailable: no stable fresh road estimate.'}
        corridor=result.get('independent_corridor') or {}
        if corridor.get('active'):
            forward=corridor.get('supported_forward_m',0)
            capped=forward>120
            limited=corridor.get('coverage')!='four_second_corridor' or capped
            return {'state':'limited' if limited else 'experimental','long_range_ready':False,
                    'near_field_eligible':True,'ground_source':source,
                    'algorithm_range_limit_m':min(120,forward),
                    'corridor_horizon_s':corridor.get('supported_horizon_s'),
                    'physical_warning_target_validated':False,
                    'message':('WARNING RANGE LIMITED — obstacle processing is capped at 120 m even when the motion corridor extends farther. The 125 m target at 70 mph is not achieved.' if capped else
                               'WARNING RANGE LIMITED — independent corridor supports only '+str(round(corridor.get('supported_horizon_s',0),2))+' seconds at this motion/uncertainty.' if limited else
                               'Commissioned motion corridor available; 4-second obstacle warning performance is still unvalidated.')}
        return {'state':'experimental','long_range_ready':True,'near_field_eligible':True,
                'ground_source':source,'algorithm_range_limit_m':None,
                'message':'Fresh road estimate; experimental readiness checks pass. Detection distance is not guaranteed.'}

    def observe(self,result,now_ns,stale=False):
        state=self.describe(result,stale);at=result.get('at_ns')
        discontinuity=(not stale and at is not None and self.last_at_ns is not None and
                       (at<=self.last_at_ns or at-self.last_at_ns>250_000_000))
        if not stale and at is not None:self.last_at_ns=at
        if state['state']=='unavailable':
            if self.unavailable_since_ns is None:self.unavailable_since_ns=now_ns
        else:self.unavailable_since_ns=None
        if not stale:   # c099 v30d review: how long the speed has been unknown; a stale-lidar observation carries no reasons and changes nothing
            if state['state']=='unavailable' and any(m in state['message'] for m in SPEED_UNKNOWN_MARKERS):
                if self.speed_unknown_since_ns is None:self.speed_unknown_since_ns=now_ns
            else:self.speed_unknown_since_ns=None;self.speed_known_seen=True
        if discontinuity:self.limited_since_ns=None;self.healthy_since_ns=None;self.available_since_ns=None
        # Regaining sustained near-field operation rearms the notice for a later
        # loss of even that coverage. Full-range recovery is not required.
        if state['state'] in ('limited','experimental'):
            if self.available_since_ns is None:self.available_since_ns=now_ns
            if now_ns-self.available_since_ns>=1_000_000_000:self.announced.discard('unavailable')
        else:self.available_since_ns=None
        if state['state']=='limited':
            if self.limited_since_ns is None:self.limited_since_ns=now_ns
            self.healthy_since_ns=None
        else:
            self.limited_since_ns=None
            if state['state']=='experimental':
                if self.healthy_since_ns is None:self.healthy_since_ns=now_ns
                if now_ns-self.healthy_since_ns>=1_000_000_000:self.announced.clear()
            else:self.healthy_since_ns=None
        prior=self.current;self.current=state
        transition=None
        if prior is None or (prior['state'],prior['ground_source'])!=(state['state'],state['ground_source']):
            transition={'kind':'coverage_transition','at_ns':at,'receipt_ns':now_ns,
                        'from':prior['state'] if prior else 'unknown','to':state['state'],**state}
        return dict(state),transition

    def pending(self,now_ns):
        if not self.current:return None
        # Claude fork v12b: the night drive of 10/19 spoke 'unavailable' 50 times in 40 min, almost all for transient readiness loss
        # (slow/sharp-turn provisional path, ground re-settling, a missed 150 ms deadline). Announce only a SUSTAINED loss (>= 8 s) and
        # never for the slow/turn reason, which is the forced mode's own by-design gap at crawl speed and in tight turns.
        msg=str(self.current.get('message',''))
        transient=('motion too slow or turn too sharp' in msg) or ('vehicle not in a forward driving gear' in msg)
        # c099 v30d review: with the speed unknown the slow/turn text is the 0 m/s fallback, not a crawl. A loss after usable comma data
        # this session is announced after the speed has been unknown for notice_s (the reason itself must persist, so a one-frame clock
        # blip at a red light does not speak); a session that never had usable comma data speaks once after COMMA_START_NOTICE_S.
        if transient and self.speed_unknown_since_ns is not None and any(m in msg for m in SPEED_UNKNOWN_MARKERS):
            hold=self.notice_s if self.speed_known_seen else max(self.notice_s,COMMA_START_NOTICE_S)
            if now_ns-self.speed_unknown_since_ns>=hold*1_000_000_000:transient=False
        if (self.current['state']=='unavailable' and 'unavailable' not in self.announced and not transient and
            self.unavailable_since_ns is not None and now_ns-self.unavailable_since_ns>=self.notice_s*1_000_000_000):return 'WARNING SYSTEM UNAVAILABLE'
        if (self.current['state']=='limited' and 'limited' not in self.announced and
            self.limited_since_ns is not None and now_ns-self.limited_since_ns>=500_000_000):return 'WARNING RANGE LIMITED'
        return None

    def submitted(self,phrase):
        self.announced.add('limited' if phrase=='WARNING RANGE LIMITED' else 'unavailable')

class Session:
    def __init__(self,config,assets,root,mode='live'):
        self.config=json.loads(json.dumps(config));self.assets=Path(assets);self.mode=mode
        self.config.pop('sync',None)  # Home SSID/destination are not sensor evidence.
        self.config['version']=APP_VERSION
        self.config['capture_mode']=mode
        self.config['session_id']=uuid.uuid4().hex
        self.config['capture_started_wall_ns']=time.time_ns()
        self.config['capture_clock']=time.describe()
        self.radar=RadarBuffer()
        self.camera=CameraBuffer()
        # Physical evidence is checked once before a session, never by disk I/O
        # on the warning path. An edited JSON flag alone cannot activate fusion.
        self.commissioning_integrity={'trusted':False,'reasons':['no accepted physical commissioning bundle']}
        if self.config.get('fcw_commissioning',{}).get('enabled'):
            try:
                from commissioning_store import validate_active_profile
                self.commissioning_integrity=validate_active_profile(self.config,Path(root))
            except Exception as e:
                self.commissioning_integrity={'trusted':False,'reasons':['commissioning evidence check failed: '+type(e).__name__]}
        if not self.commissioning_integrity.get('trusted'):
            self.config['fcw_commissioning']={**self.config.get('fcw_commissioning',{}),'enabled':False}
        self.empirical_integrity={'trusted':False,'reasons':['no applied ordinary-drive calibration']}
        if self.config.get('empirical_calibration'):
            try:
                from empirical_calibration import check_active,effective_config
                self.empirical_integrity=check_active(self.config,Path(root))
                if self.empirical_integrity.get('trusted'):
                    self.config=effective_config(self.config)
                    self.commissioning_integrity=self.empirical_integrity
            except Exception as e:
                self.empirical_integrity={'trusted':False,'reasons':['data-derived calibration check failed: '+type(e).__name__]}
        from forced_experiment import configured
        self.forced_experiment=configured(self.config)
        if self.forced_experiment:
            import forced_experiment as _fe,sys as _sys;from pathlib import Path as _P
            _mp=self.config.get('map_prior_file') or None
            if _mp and not _P(_mp).is_absolute():_mp=str((_P(_sys.executable if getattr(_sys,'frozen',False) else _sys.argv[0]).resolve().parent/_mp))   # v12b: relative to the exe
            _fe.MAP_PRIOR_FILE=_mp
            if _mp:
                try:
                    from map_live import live_map;_m=live_map(_mp)   # v12b: preload at startup, never inside the frame thread
                    self.config['map_prior_status']=('loaded %d cells'%len(_m.n)) if _m is not None else ('unreadable: '+__import__('map_live')._MAP_ERR[0])
                except Exception as e:self.config['map_prior_status']='unreadable: '+str(e)[:120]
        from limited_experiment import allow_audio
        if not self.forced_experiment and not allow_audio(self.config,self.empirical_integrity):self.config['audio_enabled']=False
        self.empirical_activation_withheld=bool(
            ('empirical_calibration' in self.config or self.config.get('calibration_basis')=='sensor_derived_empirical') and
            not self.empirical_integrity.get('trusted') and not self.commissioning_integrity.get('trusted'))
        if self.empirical_activation_withheld:
            # A recorded effective configuration may contain promoted legacy
            # flags. Those flags cannot survive loss of their empirical proof.
            self.config['mount'].update(measured=False,alignment_verified=False)
            self.config.update(timing_verified=False,lidar_timing_bound_s=None)
        if self.forced_experiment:
            self.config['forced_ignored_empirical_id']=(self.config.get('empirical_calibration') or {}).get('id')
            self.config.pop('empirical_calibration',None)
            self.config['fcw_commissioning']={**self.config.get('fcw_commissioning',{}),'enabled':False}
            self.config['mount']=self.forced_experiment.mount
            self.config.update(timing_verified=False,lidar_timing_bound_s=None,lidar_time_adjustment_s=0.,calibration_basis='forced_unverified_provisional')
            self.commissioning_integrity={'trusted':False,'reasons':['Forced experiment is not calibration approval']}
            self.empirical_integrity={'trusted':False,'reasons':['Forced experiment uses an unverified provisional model']}
        self.radar_forecaster=(RadarLiveForecaster(Path(root).parent,self.assets,APP_VERSION,self.config['session_id'])
                               if mode in ('live','comma') and not self.config.get('synthetic_fixture') else None)
        self.config['radar_contract_version']=1
        self.config['radar_forecast_model_id']=self.radar_forecaster.model_id if self.radar_forecaster else None
        self.processing_backend='isolated' if os.name=='nt' and mode=='live' and self.forced_experiment is not None else 'inline'
        self.config['processing_backend']=self.processing_backend
        self.folder=Path(root)/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1_000_000)+'-'+mode)
        self.folder.mkdir(parents=True,exist_ok=False)
        # c099 v30d: stamp the build (forensics §1). start_build_stamp is idempotent, so sessions made without Application are stamped
        # too; a session started while the exe is still being hashed shows state 'hashing' here and final-state.json carries the hash.
        # self.config itself is unchanged.
        start_build_stamp()
        (self.folder/'session.json').write_text(json.dumps({**self.config,'build':build_stamp()},indent=2),encoding='utf-8')
        self.stop_event=threading.Event();self.lock=threading.RLock();self.log_lock=threading.Lock();self.health_lock=threading.Lock()
        self.storage_stop_requested=threading.Event()
        self.stop_lock=threading.Lock()
        # Raw packets are recorded before offer_frame. Under processing load,
        # keep only the newest pending full scan instead of building latency.
        # The in-flight scan, source timestamps and freshness gates are unchanged.
        self.frames=queue.Queue(maxsize=1);self.stats=Counter();self.errors=deque(maxlen=6)
        self.stats.update({k:0 for k in ('comma_records_unsaved','comma_non_json','comma_payloads_invalidated',
                                       'comma_records_rejected','route_records_rejected','route_records_unsaved',
                                       'comma_context_failures','engine_deadline_overruns',
                                       'comma_link_attempts','comma_link_failures','comma_links_established',   # c099 v30d: link history in /state and final-state.json
                                       'empty_scans')})   # c099 v30d review: scans with no 0.2-200 m return (covered/blocked lidar window)
        self.stats.update({k:0 for k in ('held_restoration_frames','held_restoration_frames_missing_provenance',
            'held_restoration_frames_with_additions','held_restoration_extra_candidate_points')})
        self.telemetry=Telemetry()
        self.engine=Engine(Mount(**self.config['mount']),self.config.get('perception_policy'),forced_experiment=self.forced_experiment)
        self.speaker=Speaker(self.assets/'audio')
        self.threads=[];self.ssh=None;self.offsets=deque(maxlen=100)
        self.state={'status':'not_ready','reasons':['waiting for lidar and comma'],'alerts':[],'tracks':[]}
        self.last_frame_ns=0;self.last_comma_ns=0;self.last_audio=None
        self.last_lidar_packet_ns=0   # c099 v30d: monotonic receipt of the last raw MSOP data packet from sensor_ip (autodrive no-lidar rule, forensics §2)
        self.engine_failure=None;self.processing_thread=None   # c099 v30d: a dead frame thread is reported as an engine error (forensics §3)
        # c099 v30d review: scan clocks. last_offer_ns = an assembled scan was queued (receiver); last_scan_ns = the frame thread took one;
        # last_usable_scan_ns = it had points (decode keeps 0.2-200 m returns; a covered or iced window gives none, and v30d then showed
        # Lidar OK + 'waiting for processed frames'). stats['empty_scans'] counts the scans without points.
        self.last_offer_ns=0;self.last_scan_ns=0;self.last_usable_scan_ns=0;self.processing_started_ns=0
        self.last_unavailable_notice_ns=None   # c099 v30d review: repeat the notice while engine_failure is set
        self.audio_fault=None
        self.speech_lock=threading.RLock()
        self.night_companion=None
        self.night_start_lock=threading.Lock()
        self.coverage=CoverageMonitor(self.config.get('coverage_unavailable_notice_s'));self.last_speech_delivery=None;self.last_status_attempt=None
        self.ground_observations=deque(maxlen=100);self.ground_candidate=None
        self.record_status='starting';self.comma_status='not connected';self.comma_recording_status='recording';self.stopped=False
        self.decision_log=open(self.folder/'decisions.jsonl','w',encoding='utf-8',buffering=1)
        self.context_log=open(self.folder/'frame-contexts.jsonl','w',encoding='utf-8',buffering=1)
        self.comma_log=open(self.folder/'comma.jsonl','w',encoding='utf-8',buffering=1)
        self.comma_rejected_log=open(self.folder/'comma-rejected.jsonl','w',encoding='utf-8',buffering=1)
        self.coverage_log=open(self.folder/'coverage-events.jsonl','w',encoding='utf-8',buffering=1)
        from audible_review import Journal
        self.audible_review=Journal(self.folder,self.config['session_id'],bool(self.config.get('audio_enabled')))
        self.pending_pings={};self.ping_index=0
        self.capture_segments=[]
        self.map_prior=None
        if mode=='live' and self.config.get('map_enabled',True) and self.config.get('timing_verified') and self.config['mount'].get('alignment_verified'):
            try:
                from mapping.live import AtlasPrior
                self.map_prior=AtlasPrior(Path(root).parent/'maps')
            except Exception as e:self.error('Map advisory unavailable: '+str(e))
        self.targets=TargetCalibration()
        self.route_journal=None
        try:
            from mapping.journal import RouteJournal
            self.route_journal=RouteJournal(self.folder,self.config['session_id'],minimum_free_bytes=self.config['minimum_free_gb']*1e9)
            if self.route_journal.error:self.error(self.route_journal.error)
        except (OSError,ValueError) as e:self.error('Route recording unavailable: '+str(e))
        if self.processing_backend=='isolated':
            from inference_worker import EngineProcess
            try:self.engine=EngineProcess(self.engine,live=True,stop_event=self.stop_event)
            except Exception:
                # No child or recorder may leak out of a failed session start.
                self.processing_backend='inline'
                self.stop(analyze=False)
                raise

    def _start_night_companion(self):
        if self.config.get('night_awareness_mode','off') not in ('auto','on') or self.stop_event.is_set():return
        if not self.night_start_lock.acquire(blocking=False):return
        try:
            if self.night_companion is not None:return
            from night_companion import NightCompanion
            self.night_companion=NightCompanion(self);self.night_companion.start()
        except Exception as error:
            self.error('Experimental awareness startup: '+str(error))
            if self.night_companion:
                try:self.night_companion.stop()
                except Exception:pass
            self.night_companion=None
        finally:self.night_start_lock.release()

    def error(self,message):
        # deque.append is atomic; diagnostics must not wait behind a stalled inference lock.
        self.errors.append(str(message))

    def spawn(self,fn,*args,register=True):
        def guarded():
            try:fn(*args)
            except Exception as e:
                self.error(f'{fn.__name__}: {e}')
                with self.lock:self.state={'status':'not_ready','reasons':[f'{fn.__name__} failed: {e}'],'alerts':[],'tracks':[]}
                # c099 v30d: 2026-10-21 forensics §3 - v30 process_frames died on its first track association (ImportError
                # near_edge_motion) and the simple page blamed the lidar cable. Errors, state and reasons above are unchanged.
                if fn.__name__ in ENGINE_THREADS:self.engine_failed(fn.__name__,f'{type(e).__name__}: {e}')
        t=threading.Thread(target=guarded,daemon=True)
        # c099 v30d review: register=False for a helper its owner joins (receive_comma's _clock_ping, joined in _end_ssh). Each finished
        # Thread kept in self.threads holds a Windows handle until the session ends: +1 per comma attempt, ~240 per hour while the comma
        # is unreachable (reviewer exp1_leak: 339 -> 494 process handles over 150 attempts).
        if register:self.threads.append(t)
        t.start()
        return t   # c099 v30d: receive_comma joins its own ping thread; start() keeps the frame thread (existing callers ignore it)

    def engine_failed(self,thread,message):
        """c099 v30d: record the first engine failure (forensics §3) for /state, final-state.json and coverage-events.jsonl.
        Failures during closeout are ignored. CoverageMonitor and speech are not touched: the stale-frame path still speaks
        WARNING SYSTEM UNAVAILABLE after notice_s, exactly as in v30."""
        with getattr(self,'health_lock',threading.Lock()):
            if self.engine_failure is not None or self.stop_event.is_set():return
            at=time.monotonic_ns()
            # c099 v30d review: 'action' names what a driver can do (was 'restart the app'); autodrive restarts the session by itself
            self.engine_failure={'thread':thread,'error':str(message)[:500],'wall_ns':time.time_ns(),'at_ns':at,
                                 'processed_frames':self.stats['processed_frames'],'action':'stop, then start the recording'}
            failure=dict(self.engine_failure)
        try:self.coverage_event({'kind':'engine_failure','receipt_ns':at,**failure})
        except (OSError,ValueError):pass

    def check_engine_thread(self,now_ns=None):
        """c099 v30d: backstop for a frame thread that ends without an exception (spawn's guard reports the ones that raise).
        c099 v30d review: also a frame thread that is alive but takes no scan for ENGINE_STALL_S while the receiver offers them (hung in
        Engine.update or on a lock) - nothing else reported it, and autodrive's raw-packet rule kept that session open with no warnings."""
        t=self.processing_thread
        if t is None or self.stop_event.is_set() or self.engine_failure is not None:return
        if not t.is_alive():
            self.engine_failed('process_frames','process_frames thread ended unexpectedly');return
        now_ns=time.monotonic_ns() if now_ns is None else now_ns
        offered=self.last_offer_ns;taken=max(self.last_scan_ns,self.processing_started_ns)
        # the queue holds one scan and a live thread polls it every 0.2 s, so 'offered within 1 s, none taken for 10 s' is a stall
        if offered and now_ns-offered<=1_000_000_000 and now_ns-taken>=ENGINE_STALL_S*1e9:
            self.engine_failed('process_frames',f'frame thread alive but no lidar scan taken for {(now_ns-taken)/1e9:.0f} s while scans arrive (stalled)')

    def start(self):
        self.processing_started_ns=time.monotonic_ns()   # c099 v30d review: stall baseline before the first scan is taken
        self.processing_thread=self.spawn(self.process_frames)   # c099 v30d: watched by check_engine_thread
        self.spawn(self.watchdog)
        if self.mode=='live':
            self.spawn(self.receive_lidar)
        if self.mode in ('live','comma'):
            if self.config['comma_host']:self.spawn(self.comma_link)   # c099 v30d: retries until Stop (v30: one receive_comma attempt per session)
            else:self.comma_status='Enter comma Wi-Fi IP in Setup'
        if self.mode=='comma':self.record_status='comma telemetry only; lidar not required'
        self._start_night_companion()

    def watchdog(self):
        while not self.stop_event.wait(.2):
            if self.storage_stop_requested.is_set():
                # Finalize from a separate thread so stop can join every worker.
                threading.Thread(target=self.stop,args=(False,),daemon=True).start()
                return
            self.check_engine_thread()   # c099 v30d
            self.check_freshness()

    def coverage_event(self,event):
        if event:
            if not self.log_lock.acquire(timeout=.05):
                self.stats['coverage_events_unsaved']+=1;return
            try:self.coverage_log.write(json.dumps(event,default=jsonable,separators=(',',':'))+'\n')
            finally:self.log_lock.release()

    def _link_event(self,event,**fields):
        # c099 v30d: comma link transitions beside the coverage transitions (2026-10-21: link state lived only in memory / comma_status).
        # Every reader of coverage-events.jsonl filters by kind.
        if self.stop_event.is_set():return
        try:self.coverage_event({'kind':'comma_link','event':event,'receipt_ns':time.monotonic_ns(),'wall_ns':time.time_ns(),**fields})
        except (OSError,ValueError):pass

    def deliver_coverage_notice(self,now_ns):
        if not self.config['audio_enabled']:return {'delivery':'muted'}
        phrase=self.coverage.pending(now_ns)
        if not phrase:return None
        played=self.speak(phrase,now_ns)
        event={'kind':'coverage_audio','phrase':phrase,'receipt_ns':now_ns,**self.last_speech_delivery}
        # A deferred current notice can retry, but repeated busy polls need not flood the journal.
        attempt=(phrase,event['delivery'])
        if attempt!=self.last_status_attempt or played:self.coverage_event(event)
        self.last_status_attempt=attempt
        if played:
            self.coverage.submitted(phrase)
            if phrase=='WARNING SYSTEM UNAVAILABLE':self.last_unavailable_notice_ns=now_ns   # c099 v30d review: engine-failure repeat clock
        return event

    def check_freshness(self,now_ns=None):
        now_ns=time.monotonic_ns() if now_ns is None else now_ns
        if not self.lock.acquire(timeout=.05):
            # Keep the supervisor alive even if a worker holds the context lock.
            # This path can announce only unavailability, never an obstacle.
            self.stats['context_lock_timeouts']+=1
            if self.engine_failure is not None and self.config.get('audio_enabled'):
                last=self.last_unavailable_notice_ns
                if last is None or now_ns-last>=ENGINE_NOTICE_REPEAT_S*1e9:
                    if self.speak('WARNING SYSTEM UNAVAILABLE',now_ns):self.last_unavailable_notice_ns=now_ns
            return
        try:
            stale=not self.last_frame_ns or now_ns-self.last_frame_ns>500_000_000
            if self.audio_fault:
                self.state={**self.state,'status':'not_ready','reasons':list(dict.fromkeys(self.state.get('reasons',[])+['audio output failed']))}
            if stale or self.state.get('status')!='eligible_experimental':
                coverage,transition=self.coverage.observe(self.state,now_ns,stale=stale)
                self.state={**self.state,'warning_coverage':coverage}
                self.coverage_event(transition)
            # c099 v30d review: a dead or stalled engine never regains coverage, so CoverageMonitor never re-armed 'unavailable' and the
            # notice was spoken once per session (v30's auto-stop/auto-start loop re-announced it about every 90 s). Re-arm it every
            # ENGINE_NOTICE_REPEAT_S while engine_failure is set. This only adds announcements; nothing is muted or delayed.
            if (self.engine_failure is not None and 'unavailable' in self.coverage.announced and self.last_unavailable_notice_ns is not None and
                now_ns-self.last_unavailable_notice_ns>=ENGINE_NOTICE_REPEAT_S*1e9):self.coverage.announced.discard('unavailable')
            self.deliver_coverage_notice(now_ns)
        finally:self.lock.release()

    def speak(self,phrase,created_ns=None):
        with self.speech_lock:return self._speak_locked(phrase,created_ns)

    def _speak_locked(self,phrase,created_ns=None):
        self.last_speech_delivery={'delivery':'failed','phrase':phrase}
        if self.audio_fault:return False
        try:
            if not self.speaker.say(phrase,created_ns):
                reason=getattr(self.speaker,'last_rejection',None) or 'failed_or_expired'
                self.last_speech_delivery['delivery']=reason
                self.stats['deferred_status_audio' if reason=='deferred_until_audio_idle' else 'expired_audio_requests']+=1;return False
            self.last_speech_delivery['delivery']='submitted_to_windows'
            interrupted=getattr(self.speaker,'last_interruption',None)
            if interrupted:
                self.coverage_event({'kind':'audio_interruption',**interrupted})
                if interrupted['phrase'] in STATUS_PHRASES:
                    state='limited' if interrupted['phrase']=='WARNING RANGE LIMITED' else 'unavailable'
                    self.coverage.announced.discard(state)
            return True
        except (OSError,RuntimeError) as e:
            self.audio_fault=str(e);self.error('Audio output failed: '+str(e)+'. Recording continues; fix output, then stop and reconnect.');self.stats['audio_failures']+=1
            return False

    def offer_frame(self,frame):
        if not frame:return
        self.last_offer_ns=time.monotonic_ns()   # c099 v30d review: a scan arrived for the frame thread (stall check)
        try:self.frames.put_nowait(frame)
        except queue.Full:
            try:self.frames.get_nowait()
            except queue.Empty:pass
            self.stats['processing_frames_dropped']+=1
            self.frames.put_nowait(frame)

    def receive_lidar(self):
        sockets=[];writer=None;assembler=FrameAssembler();record_start=0;total_bytes=0;index=0;disk_check=0
        try:
            for port in (self.config['msop_port'],self.config['difop_port']):
                s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                if os.name=='nt':s.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
                s.bind((self.config['bind_ip'],port));s.setblocking(False);sockets.append(s)
            self.record_status='recording';last_disk_ok=True
            while not self.stop_event.is_set():
                ready,_,_=select.select(sockets,[],[],.1)
                for s in ready:
                    for _ in range(512):
                        try:p,source=s.recvfrom(65535)
                        except BlockingIOError:break
                        mono=time.monotonic_ns();wall=time.time_ns()
                        if source[0]!=self.config['sensor_ip']:self.stats['other_source_packets']+=1;continue
                        port=s.getsockname()[1];self.stats['udp_packets']+=1
                        try:
                            if mono-disk_check>1e9:
                                disk_check=mono;last_disk_ok=recording_space_available(self.folder,self.config)
                            if self.record_status=='recording':
                                if not last_disk_ok:
                                    self.storage_stop_requested.set()
                                    raise OSError('free-space reserve reached; ending recording')
                                if lidar_budget_exhausted(self.config,total_bytes+(writer.bytes if writer else 0),len(p)+44):
                                    raise OSError('session storage limit reached; recording stopped')
                                if writer is None or mono-record_start>60e9:
                                    if writer:total_bytes+=writer.bytes;writer.close()
                                    writer=PcapWriter(self.folder/f'lidar-{index:05d}.pcap',self.config['destination_ip'])
                                    self.capture_segments.append({'file':f'lidar-{index:05d}.pcap','first_receipt_ns':mono,
                                                                  'last_receipt_ns':mono,'first_wall_ns':wall,'last_wall_ns':wall})
                                    index+=1;record_start=mono
                                writer.write(p,source,port,wall)
                                self.capture_segments[-1].update(last_receipt_ns=mono,last_wall_ns=wall,bytes=writer.bytes)
                                self.stats['recorded_bytes']=total_bytes+writer.bytes
                        except OSError as e:
                            if writer:writer.close();writer=None
                            self.record_status=str(e);self.error(e)
                        if port==self.config['msop_port']:
                            self.last_lidar_packet_ns=mono   # c099 v30d: raw data packet clock (forensics §2); after the sensor_ip filter, data port only
                            for frame in assembler.push(p,mono):self.offer_frame(frame)
                            self.stats['invalid_packets']=assembler.bad_packets
                            self.stats['duplicate_packets']=assembler.duplicate_packets
                            self.stats['incomplete_frames']=assembler.partial_frames
                        else:self.stats['difop_packets']+=1
        finally:
            if writer:writer.close()
            for s in sockets:s.close()

    def _clock_ping(self,proc):
        # c099 v30d: bound to ONE ssh child; v30 re-read self.ssh, so after a reconnect an old ping thread would also write to the new
        # child (double rate, interleaved stdin). ValueError = stdin closed by _end_ssh/_stop (it escaped to spawn's guard before).
        while not self.stop_event.wait(.5):
            if proc is not self.ssh or proc.poll() is not None:break
            self.ping_index+=1;t0=time.monotonic_ns()
            with self.lock:
                self.pending_pings={k:v for k,v in self.pending_pings.items() if t0-v<5e9}
                self.pending_pings[self.ping_index]=t0
            try:
                proc.stdin.write((json.dumps({'kind':'ping','id':self.ping_index})+'\n').encode());proc.stdin.flush()
            except (BrokenPipeError,OSError,ValueError):break

    def _comma_ssh_args(self):
        # c099 v30d: v30 receive_comma's checks and ssh argv moved here unchanged, so comma_link validates the configuration once and
        # does not retry a configuration error.
        host=self.config['comma_host'];python=self.config['comma_python']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',host):raise ValueError('invalid comma IP/hostname')
        if not re.fullmatch(r'[A-Za-z0-9_./-]+',python) or python.startswith('-'):raise ValueError('invalid interpreter path')
        ssh=shutil.which('ssh')
        if not ssh:raise RuntimeError('Windows OpenSSH Client is required')
        source=base64.b64encode((self.assets/'comma_reader.py').read_bytes()+b'\n'+
                               (self.assets/'comma_bridge_v2.py').read_bytes()).decode()
        selection=''
        if python in ('auto','python3'):
            selection='FCW_PYTHON=python3; if [ -x /usr/local/venv/bin/python ]; then FCW_PYTHON=/usr/local/venv/bin/python; fi; '
            python='"$FCW_PYTHON"'
        command=f'cd /data/openpilot && ( {selection}exec nice -n 10 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/openpilot:/data {python} -u -c \'import base64;exec(base64.b64decode("{source}"))\' )'
        args=[ssh,'-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',
              '-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2']
        if self.config['ssh_key']:args+=['-i',os.path.expandvars(os.path.expanduser(self.config['ssh_key']))]
        args+=['comma@'+host,command]
        return args

    def receive_comma(self,attempt=1):
        """One ssh attempt; returns at end of stream. c099 v30d: comma_link repeats it until Stop (v30: one attempt per session)."""
        args=self._comma_ssh_args()
        path=self.folder/'ssh-errors.txt';started=time.monotonic_ns();hello_ns=None;records=0;ping=None;code=None
        if self.stop_event.is_set():return {'code':None,'hello':False,'up_s':0.,'records':0,'detail':'stopped'}
        self.comma_status='connecting' if attempt==1 else f'reconnecting (attempt {attempt})'
        with self.lock:self.pending_pings={}   # an ended child never answers; ping_index stays session-unique
        self.stats['comma_link_attempts']+=1
        # c099 v30d: append, with a timestamped header and footer per attempt (v30: 'w', one bare line in 122812's ssh-errors.txt)
        with path.open('ab') as errorfile:
            errorfile.write(f'--- attempt {attempt} at {datetime.datetime.now().isoformat(timespec="milliseconds")} wall_ns {time.time_ns()} monotonic_ns {started} ---\n'.encode());errorfile.flush()
            mark=errorfile.tell()
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errorfile,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.ssh=proc   # publish, THEN test stop_event: either _stop or this thread ends the child
            progress={'started_ns':started,'receipt_ns':None,'reason':None}
            monitor_done=threading.Event()
            monitor=threading.Thread(target=self._monitor_comma_child,args=(proc,progress,monitor_done),daemon=True,name='comma-application-watchdog')
            monitor.start()
            try:
                if not self.stop_event.is_set():
                    ping=self.spawn(self._clock_ping,proc,register=False)   # c099 v30d review: joined by _end_ssh, not kept in self.threads
                    for line in iter(proc.stdout.readline,b''):
                        if self.stop_event.is_set():break
                        receipt=time.monotonic_ns()
                        try:item=json.loads(line)
                        except (ValueError,UnicodeError):
                            self.reject_comma_line(line,receipt,'invalid_json');continue
                        if not isinstance(item,dict):
                            self.reject_comma_line(line,receipt,'non_object_json');continue
                        if item.get('kind') in ('hello','pong','sample','route_reference','stream_restart'):
                            progress['receipt_ns']=receipt
                        records+=1
                        if item.get('kind')=='hello' and hello_ns is None:
                            hello_ns=receipt;self.stats['comma_links_established']+=1;self._link_event('established',attempt=attempt)
                        self.accept_comma_record(item,receipt)
            finally:
                monitor_done.set()
                code=self._end_ssh(proc,ping)
                monitor.join(timeout=1)
            ended=time.monotonic_ns();up=(ended-hello_ns)/1e9 if hello_ns is not None else 0.
            # ssh's own error ('Unknown error', 'Connection timed out', 'Permission denied', ...) or the bridge's forwarded exception
            try:
                with path.open('rb') as f:f.seek(max(mark,f.seek(0,2)-4096));tail=f.read().decode('utf-8','replace')
            except OSError:tail=''
            detail=progress['reason'] or next((l.strip() for l in reversed(tail.splitlines()) if l.strip()),'')[:200] or f'SSH ended ({code})'
            errorfile.write(f'--- attempt {attempt} ended: exit {code} after {(ended-started)/1e9:.1f} s, {records} records, link up {up:.1f} s ---\n'.encode())
        self.comma_status=f'SSH ended ({code}); see ssh-errors.txt'
        return {'code':code,'hello':hello_ns is not None,'up_s':up,'records':records,'detail':detail}

    def _monitor_comma_child(self,proc,progress,done):
        """Application liveness, independent of blocking stdout.readline and session locks.

        Pongs keep an off-road bridge alive even when driving services are absent.
        Silence recycles only this owned child; comma_link owns bounded retry/backoff.
        """
        while not done.wait(COMMA_MONITOR_POLL_S):
            if proc.poll() is not None:return
            received=progress['receipt_ns'];now=time.monotonic_ns()
            limit=COMMA_START_TIMEOUT_S if received is None else COMMA_APPLICATION_IDLE_S
            expired=now-(received if received is not None else progress['started_ns'])>=limit*1e9
            if self.stop_event.is_set() or proc is not self.ssh or expired:
                if expired and not self.stop_event.is_set():
                    progress['reason']=f'comma application bridge silent for {limit:g} s; transport may still be connected'
                    self.stats['comma_application_timeouts']+=1
                    self.comma_status=progress['reason']
                try:proc.terminate()
                except OSError:pass
                try:proc.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    try:proc.kill()
                    except OSError:pass
                return

    def _end_ssh(self,proc,ping):
        """c099 v30d: reap one ssh child (several per session now): terminate, wait, kill if needed, stop its ping thread, close its pipes.
        The ping thread is joined before stdin is closed, so in the normal path it never writes to a closed pipe."""
        if proc.poll() is None:
            try:proc.terminate()
            except OSError:pass
        try:code=proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();code=proc.wait(timeout=5)
        if ping is not None:ping.join(timeout=2)
        for pipe in (proc.stdin,proc.stdout):
            try:pipe.close()
            except (OSError,ValueError):pass
        return code

    def comma_link(self):
        """c099 v30d: keep the comma link for the whole session (2026-10-21 forensics §5: one attempt per session; 122812 and 115523
        never linked). Backoff 2 s doubling to 15 s on stop_event (Stop stays prompt); one logged error per failure streak; a link
        that was up >= COMMA_LINK_HEALTHY_S re-arms both, so a bridge that crashes right after its hello cannot hot-loop."""
        self._comma_ssh_args()   # a configuration error is not retried: it ends this thread through spawn's guard, as in v30
        attempt=0;delay=COMMA_RETRY_FIRST_S;streak=0
        while not self.stop_event.is_set():
            attempt+=1
            try:outcome=self.receive_comma(attempt)
            except Exception as e:outcome={'code':None,'hello':False,'up_s':0.,'records':0,'detail':(type(e).__name__+': '+str(e))[:200]}
            if self.stop_event.is_set():break
            # Only a link that delivered its hello can re-arm (a failed attempt has up_s 0, which must not count when the
            # threshold is patched to 0).
            if outcome['hello'] and outcome['up_s']>=COMMA_LINK_HEALTHY_S:streak=0;delay=COMMA_RETRY_FIRST_S
            streak+=1;self.stats['comma_link_failures']+=1
            lost=self.stats['comma_links_established']>0
            if streak==1:self.error(f"Comma link {'lost' if lost else 'not established'} ({outcome['detail']}); retrying until Stop")
            # c099 v30d review: every link that delivered its hello logs its loss, whatever the streak (only streak 1 did: a link shorter
            # than COMMA_LINK_HEALTHY_S after a failed attempt left 'established' as the last event while the comma was down; reviewer
            # probe_short_link). self.errors keeps one line per streak; a failed attempt logs only the first failure of its streak.
            if outcome['hello'] or streak==1:
                self._link_event('lost' if outcome['hello'] or lost else 'not_established',attempt=attempt,exit_code=outcome['code'],
                                 detail=outcome['detail'],up_s=round(outcome['up_s'],1))
            self.comma_status=f"reconnecting (attempt {attempt+1} in {delay:.0f} s): {outcome['detail']}"
            if self.stop_event.wait(delay):break
            delay=min(COMMA_RETRY_MAX_S,delay*2)
        self.comma_status=f"stopped; {self.stats['comma_links_established']} link(s) in {attempt} attempt(s)"

    def save_comma_evidence(self,item,rejected=False):
        """Raw capture is independent of normalization and advisory route parsing."""
        try:
            with self.log_lock:
                line=json.dumps(item,separators=(',',':'))+'\n'
                if not recording_space_available(self.folder,self.config,len(line.encode('utf-8'))):
                    self.storage_stop_requested.set()
                    raise OSError('free-space reserve reached; ending recording')
                target=self.comma_rejected_log if rejected else self.comma_log
                target.write(line)
                self.comma_recording_status='recording with prior gaps' if self.stats['comma_records_unsaved'] else 'recording'
                return True
        except (OSError,ValueError,TypeError) as e:
            self.comma_recording_status='interrupted: '+str(e)
            if not self.stats['comma_records_unsaved']:self.error('Raw comma recording interrupted: '+str(e))
        self.stats['comma_records_unsaved']+=1
        return False

    def reject_comma_line(self,line,receipt,reason):
        self.stats['comma_non_json']+=1
        # Base64 preserves invalid UTF-8 and exact input bytes without inventing a sample.
        self.save_comma_evidence({'kind':'rejected_stream_line','reason':reason,
            'laptop_receipt_ns':receipt,'laptop_wall_ns':time.time_ns(),
            'raw_base64':base64.b64encode(line).decode('ascii')},rejected=True)

    def accept_comma_record(self,item,receipt=None):
        """One ingestion boundary; malformed records cannot end a whole stream."""
        receipt=time.monotonic_ns() if receipt is None else receipt
        if not isinstance(item,dict):
            self.reject_comma_line(json.dumps(item).encode('utf-8'),receipt,'non_object_json');return
        item={**item,'laptop_receipt_ns':receipt,'laptop_wall_ns':time.time_ns()}
        # Enrich pongs with the actual send timestamp before preserving the record.
        with self.lock:
            ping_id=item.get('id')
            t0=self.pending_pings.pop(ping_id,None) if item.get('kind')=='pong' and type(ping_id) in (int,str) else None
            if t0 is not None:item['laptop_send_ns']=t0
        self.save_comma_evidence(item)
        try:self.audible_review.ingest(item,receipt)
        except (OSError,ValueError,TypeError) as e:self.error('Camera review index unavailable: '+str(e))
        if item.get('kind')=='hello':
            try:(self.folder/'comma-device.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
            except OSError as e:self.error('Comma device metadata not saved: '+str(e))
        with self.lock:
            try:
                if ingest_telemetry_record(self.telemetry,item,receipt,t0):self.stats['comma_payloads_invalidated']+=1
            except Exception as e:
                self.stats['comma_records_rejected']+=1
                if self.stats['comma_records_rejected']==1:self.error('Comma normalization rejected a record: '+type(e).__name__+': '+str(e))
            try:sync=self.telemetry.clock.estimate(receipt)
            except Exception as e:
                sync=None;self.stats['comma_records_rejected']+=1
                self.error('Comma clock estimate unavailable: '+str(e))
            try:
                self.radar.accept(item,receipt)
                if item.get('kind')=='sample' and item.get('service') in ('liveTracks','radarState'):
                    self.stats['radar_records_received']+=1
            except Exception as e:
                self.radar.reset('radar normalization failure')
                if self.radar_forecaster:self.radar_forecaster.clear()
                self.stats['radar_records_rejected']+=1
                if self.stats['radar_records_rejected']==1:self.error('Radar evidence unavailable: '+type(e).__name__)
            try:
                self.camera.accept(item,receipt)
                if item.get('kind')=='sample' and item.get('service')=='modelV2':self.stats['camera_records_received']+=1
            except Exception as e:
                self.camera.reset('camera normalization failure')
                self.stats['camera_records_rejected']+=1
                if self.stats['camera_records_rejected']==1:self.error('Camera evidence unavailable: '+type(e).__name__)
            if item.get('kind')=='sample':self.stats['comma_samples']+=1
            self.last_comma_ns=receipt;self.comma_status='connected; checking source freshness and clock bounds'
        # Optional small-model inference owns its own lock, never the lidar
        # context lock. Exact original messages have already been recorded.
        if self.radar_forecaster:
            try:
                self.radar_forecaster.identity_record(item)
                if item.get('kind')=='sample' and item.get('service')=='liveTracks':
                    self.radar_forecaster.accept(self.radar.latest(receipt,sync))
            except Exception as e:
                self.radar_forecaster.clear();self.stats['radar_forecast_errors']+=1
                if self.stats['radar_forecast_errors']==1:self.error('Radar forecast unavailable: '+type(e).__name__)
        if self.route_journal:
            try:saved=self.route_journal.ingest(item,sync);problem=self.route_journal.error
            except Exception as e:
                saved=False;problem=f'unreadable {item.get("service")} record ({type(e).__name__}: {e})'
                self.stats['route_records_rejected']+=1
            if not saved:
                self.stats['route_records_unsaved']+=1
                if self.stats['route_records_unsaved']==1:self.error('Compact route recording interrupted: '+str(problem))

    def process_frames(self):
        while not self.stop_event.is_set():
            try:packets,q=self.frames.get(timeout=.2)
            except queue.Empty:continue
            dequeued_ns=time.monotonic_ns()
            self.last_scan_ns=dequeued_ns   # c099 v30d review: the frame thread is taking scans (stall check)
            decode_start=time.perf_counter_ns()
            points,stamps=decode_with_time(packets)
            decode_ms=(time.perf_counter_ns()-decode_start)/1e6
            if not len(stamps):
                self.stats['empty_scans']+=1;continue   # c099 v30d review: counted (no usable point: blocked window); still skipped as in v30
            self.last_usable_scan_ns=dequeued_ns   # c099 v30d review
            now=time.monotonic_ns()
            self.offsets.append((now,q['min_receipt_minus_device_ns']))
            while self.offsets and now-self.offsets[0][0]>10e9:self.offsets.popleft()
            offset=min(s[1] for s in self.offsets)
            last_stamp=stamps.max()
            end_ns=int(last_stamp+offset+self.config['lidar_time_adjustment_s']*1e9)
            context_start=time.perf_counter_ns()
            with self.lock:
                q['fusion_cutoff_ns']=time.monotonic_ns()
                ctx,problem=safe_telemetry_context(self.telemetry,end_ns)
                if problem:
                    self.stats['comma_context_failures']+=1
                    if self.stats['comma_context_failures']==1:self.error('Comma context rejected malformed data: '+problem)
            q['pipeline_timing']={'queue_age_ms':max(0,(dequeued_ns-q['last_receipt_ns'])/1e6),
                                  'context_ms':(time.perf_counter_ns()-context_start)/1e6}
            q.update(arrival_age_s=max(0,(time.monotonic_ns()-q['last_receipt_ns'])/1e9),
                     timing_bound_s=self.config['lidar_timing_bound_s'] if self.config['timing_verified'] else 1.)
            if q['timing_bound_s'] is None:q['timing_bound_s']=1.
            relative=(stamps-last_stamp)/1e9
            try:self.process_observation(points,ctx,q,relative,decode_duration_ms=decode_ms)
            except Exception:
                if self.stop_event.is_set():return
                raise

    def process_observation(self,points,ctx,q,relative=None,decode_duration_ms=None):
        start=time.monotonic_ns()
        if self.audio_fault and self.config['audio_enabled']:
            ctx={**ctx,'reasons':ctx.get('reasons',[])+['audio output failed']}
        if self.forced_experiment:
            ctx=self.forced_experiment.prepare(ctx)
        elif self.empirical_integrity.get('trusted'):
            from empirical_corridor import prepare_context as prepare_empirical_context
            ctx=prepare_empirical_context(ctx,self.config['empirical_calibration'],q)
            ctx['empirical_requires_observed_ground']=self.config['empirical_calibration'].get('requires_observed_ground') is True
            ctx['empirical_geometry_basis']=self.config['empirical_calibration'].get('geometry_basis')
        else:ctx=prepare_context(ctx,self.config,q) if FEATURE_LEVEL>=1 else dict(ctx)
        if self.empirical_activation_withheld and not self.forced_experiment:
            ctx.update(ready=False,reasons=list(dict.fromkeys(ctx.get('reasons',[])+
                ['Data-derived calibration integrity failed: '+r for r in self.empirical_integrity.get('reasons',[])])))
        from limited_experiment import prepare_context as prepare_limited_context
        if not self.forced_experiment:ctx=prepare_limited_context(ctx,self.config.get('limited_experiment'),empirical_trusted=self.empirical_integrity.get('trusted') is True)
        trusted=self.commissioning_integrity.get('trusted') is True
        ctx.update(mount_epoch=self.config.get('mount_epoch'),
                   calibration_id=self.commissioning_integrity.get('calibration_id') if trusted else None,
                   calibration_verified=trusted)
        # Snapshots use the lidar source time and records already received by
        # that time; future arrivals cannot be borrowed to improve a decision.
        with self.lock:
            sync=self.telemetry.clock.estimate(ctx['at_ns'])
            radar=self.radar.latest(ctx['at_ns'],sync) if FEATURE_LEVEL>=2 else None
            camera=self.camera.latest(ctx['at_ns'],sync) if FEATURE_LEVEL>=3 else None
        prior=None
        if FEATURE_LEVEL>=4 and self.map_prior:
            try:prior=self.map_prior.snapshot(ctx)
            except Exception:self.stats['map_snapshot_errors']+=1
        fusion_calibration=(self.config['empirical_calibration'].get('fusion_profile',{}) if self.empirical_integrity.get('trusted') else
                            self.config.get('sensor_fusion',{}) if trusted else {})
        if self.forced_experiment and isinstance(radar,dict) and radar.get('available') and radar.get('service','liveTracks')=='liveTracks':
            from radar_fusion import radar_points_from_record
            publication=radar.get('source_ns')
            ctx['radar_publication_ns']=publication
            ctx['radar_publication_age_s']=radar.get('source_age_s')
            ctx['radar_clock_bound_s']=radar.get('clock_bound_s')
            ctx['radar_points']=[{**p,'publication_ns':publication,'source_mono_ns':publication}
                                 for p in radar_points_from_record(radar)]
        fusion_inputs={'radar_snapshot':radar,'camera_snapshot':camera,
                       'calibration':fusion_calibration,
                       'phase':FEATURE_LEVEL,'map_snapshot':prior}
        engine_start=time.perf_counter_ns()
        result=self.engine.update(points,ctx,q,relative,processing_deadline_ns=start+150_000_000,fusion_inputs=fusion_inputs)
        engine_ms=(time.perf_counter_ns()-engine_start)/1e6
        if 'scenario' in ctx:result['scenario']=ctx['scenario']
        if ctx.get('limited_experiment'):result['limited_experiment']=ctx['limited_experiment']
        q['processing_overrun']='inference exceeded 150 ms' in result['reasons']
        with self.lock:
            self.state=result;self.last_frame_ns=time.monotonic_ns();self.stats['processed_frames']+=1
            ground=result.get('ground') or {}
            if ground.get('source')=='held':
                provenance=ground.get('held_restoration')
                if not isinstance(provenance,dict):provenance={}
                extra=provenance.get('extra_candidate_points')
                if type(provenance.get('schema_version')) is int and provenance['schema_version']==1 and type(extra) is int and extra>=0:
                    self.stats['held_restoration_frames']+=1
                    self.stats['held_restoration_frames_with_additions']+=extra>0
                    self.stats['held_restoration_extra_candidate_points']+=extra
                else:self.stats['held_restoration_frames_missing_provenance']+=1
            self.stats['engine_deadline_overruns']+=int(q['processing_overrun'])
            result['processing_timing']={'schema_version':2,'engine_ms':round(engine_ms,3),
                'decode_ms':round(decode_duration_ms,3) if decode_duration_ms is not None else None,
                'decode_scope':'packet_decode' if decode_duration_ms is not None else 'not_measured',
                'capture_mode':self.mode,'clock':'perf_counter_ns','engine_scope':'Engine.update',
                'engine_deadline_ms':150,'engine_deadline_exceeded':bool(q['processing_overrun']),
                'engine_elapsed_over_150_ms':engine_ms>150,
                'engine_deadline_overruns_session':self.stats['engine_deadline_overruns']}
            result['processing_timing'].update(q.get('pipeline_timing',{}))
            if getattr(self,'processing_backend','inline')=='isolated':
                result['processing_timing']['engine_scope']='isolated Engine.update roundtrip'
                result['processing_timing']['worker']=dict(self.engine.last_metrics)
            q['processing_timing']=result['processing_timing']
            self.targets.observe(result,ctx)
            self.stats['decoded_points']+=len(points)
            result['path_preview']=ctx['path'][::2].round(2).tolist()
            if (ctx.get('stationary_confirmed') and result['ground'] and not result['ground'].get('degraded') and q.get('completeness',0)>=.98 and
                q.get('timestamp_valid') and q.get('arrival_age_s',1)<=.2):
                self.ground_observations.append(result['ground'])
                self.ground_candidate=ground_proposal(self.engine.mount,list(self.ground_observations))
            if self.config['audio_enabled']:
                if result['alerts']:
                    request_ns=time.monotonic_ns()
                    played=self.speak(result['alerts'][0]['phrase'],start)
                    result['audio_delivery']='submitted_to_windows' if played else 'failed_or_expired'
                    try:self.audible_review.warning(result,ctx,played,request_ns,sync)
                    except (OSError,ValueError,TypeError) as e:
                        self.error('Audible warning bookmark could not be saved: '+str(e));self.stats['audible_bookmark_errors']+=1
                    if played:self.last_audio=result['alerts'][0]
                    else:self.engine.reject_delivery(result['alerts'][0]['track_id'],result['at_ns'])
            else:result['audio_delivery']='muted'
            if self.audio_fault:
                result['status']='not_ready';result['reasons'].append('audio output failed')
            coverage_now=time.monotonic_ns()
            result['warning_coverage'],transition=self.coverage.observe(result,coverage_now)
            self.coverage_event(transition)
            result['status_audio_delivery']=self.deliver_coverage_notice(coverage_now)
            if self.audio_fault and result['status']!='not_ready':
                result['status']='not_ready';result['reasons'].append('audio output failed')
                result['warning_coverage'],transition=self.coverage.observe(result,coverage_now)
                self.coverage_event(transition)
            if self.map_prior:
                try:
                    body=transform(points,self.engine.mount)
                    if relative is not None and ctx.get('ready'):body=deskew(body,relative,ctx['speed_mps'],ctx['yaw_rate_rps'])
                    map_context={**ctx,'location':self.route_journal.latest_fix(ctx['at_ns']) if self.route_journal else None}
                    self.map_prior.offer(body,map_context,q,result.get('tracks',[]))
                    # The pinned predecision prior was consumed by Engine.
                    # A newer worker result must not retrospectively relabel it.
                except Exception:
                    result['map_status']='unavailable';result['map_note']='advisory failure; original warning decision retained'
        # These measurements diagnose delays beyond Engine.update. They never
        # change source timestamps, freshness tests, deadlines, or decisions.
        result['processing_timing']['post_engine_ms']=round((time.perf_counter_ns()-engine_start)/1e6-engine_ms,3)
        receipt=q.get('last_receipt_ns')
        result['processing_timing']['receipt_to_log_ms']=max(0,(time.monotonic_ns()-receipt)/1e6) if receipt is not None and self.mode=='live' else None
        log_start=time.perf_counter_ns()
        with self.log_lock:
            self.decision_log.write(json.dumps(result,default=jsonable,separators=(',',':'))+'\n')
            self.context_log.write(json.dumps({'context':ctx,'quality':q,
                'sensor_observations':{'schema_version':1,'radar':radar,'camera':camera,
                    'map_prior':prior.audit() if prior else None}},default=jsonable,separators=(',',':'))+'\n')
        log_ms=(time.perf_counter_ns()-log_start)/1e6
        self.stats['last_frame_log_ms']=round(log_ms,3)
        self.stats['max_frame_log_ms']=max(self.stats['max_frame_log_ms'],round(log_ms,3))
        if self.night_companion:
            try:self.night_companion.offer_lidar(result,ctx,q)
            except Exception as error:self.error('Awareness observation: '+str(error))
        return result

    def supervisor_snapshot(self):
        """Atomic progress fields only: watchdogs must not wait for the inference lock."""
        now=time.monotonic_ns();frame=self.last_frame_ns;packet=self.last_lidar_packet_ns
        device=self.telemetry.messages.get('deviceState',{})
        ignition=device.get('data',{}).get('started') if device and now-device.get('receipt_ns',0)<5e9 else None
        return {'mode':self.mode,'folder':str(self.folder),'running':not self.stopped and not self.stop_event.is_set(),
                'stopped':self.stopped,'closing':self.stop_event.is_set(),
                'frame_age_s':max(0.,(now-frame)/1e9) if frame else None,
                'lidar_packet_age_s':max(0.,(now-packet)/1e9) if packet else None,'ignition_on':ignition,
                'engine_failure':dict(self.engine_failure) if self.engine_failure else None,
                'recording':self.record_status,'comma_recording':self.comma_recording_status,
                'state':{'status':'not_ready','alerts':[],'tracks':[],'reasons':['context snapshot temporarily unavailable']}}

    def snapshot(self):
        # Journal snapshot may wait for its durability flush. Do not hold the
        # fusion/telemetry lock while waiting on this advisory status read.
        route_status=self.route_journal.snapshot() if self.route_journal else {'state':'unavailable','location':{'state':'unavailable','message':'Route journal unavailable; see errors.'}}
        now=time.monotonic_ns()
        build=build_stamp()   # c099 v30d: taken before self.lock, so the build lock never nests under it
        if not self.lock.acquire(timeout=.1):return self.supervisor_snapshot()
        try:
            age=(now-self.last_frame_ns)/1e9 if self.last_frame_ns else None
            packet=self.last_lidar_packet_ns   # c099 v30d
            scan,usable=self.last_scan_ns,self.last_usable_scan_ns   # c099 v30d review
            failure=dict(self.engine_failure) if self.engine_failure else None   # c099 v30d
            state=dict(self.state)
            if self.audio_fault:state={**state,'status':'not_ready','reasons':['audio output failed']+state.get('reasons',[])}
            if age is None or age>.5:state={**state,'status':'not_ready','reasons':['no fresh processed lidar frame']+state.get('reasons',[]),'alerts':[],
                                           'warning_coverage':CoverageMonitor.describe(state,stale=True)}
            elif self.audio_fault:state['warning_coverage']=CoverageMonitor.describe(state)
            services={k:{'valid':bool(v.get('valid')),'age_s':round((now-v['receipt_ns'])/1e9,2)} for k,v in self.telemetry.messages.items()}
            for k in ('liveCalibration','livePose','liveLocationKalman','modelV2'):
                if k in services:
                    data=self.telemetry.messages[k].get('data',{})
                    services[k]['source_health']={name:data[name] for name in
                        ('calStatus','calPerc','inputsOK','sensorsOK','posenetOK','confidence') if name in data}
            device=self.telemetry.messages.get('deviceState',{})
            ignition=device.get('data',{}).get('started') if device and now-device['receipt_ns']<5e9 else None
            radar=self.radar.latest(now,self.telemetry.clock.estimate(now))
            camera=self.camera.latest(now,self.telemetry.clock.estimate(now))
            radar_forecast=self.radar_forecaster.snapshot(radar) if self.radar_forecaster else None
            return {'mode':self.mode,'folder':str(self.folder),'state':state,'stats':dict(self.stats),
                    'radar':radar,'camera':camera,'radar_forecast':radar_forecast,
                    'capabilities':capabilities(),'commissioning':workflow_status(self.config),
                    'commissioning_integrity':self.commissioning_integrity,
                    'empirical_integrity':self.empirical_integrity,
                    'comma_services':services,'ignition_on':ignition,
                    'frame_age_s':age,
                    # c099 v30d: raw lidar packet age (forensics §2), first engine failure (§3) and build stamp (§1)
                    'lidar_packet_age_s':max(0.,(now-packet)/1e9) if packet else None,'engine_failure':failure,'build':build,
                    # c099 v30d review: scans taken by the frame thread, and the last one with usable points (simple page: blocked window)
                    'lidar_scan_age_s':max(0.,(now-scan)/1e9) if scan else None,'lidar_usable_scan_age_s':max(0.,(now-usable)/1e9) if usable else None,
                    'comma_status':self.comma_status,'recording':self.record_status,'comma_recording':self.comma_recording_status,
                    'errors':list(self.errors),'ground_proposal':self.ground_candidate,
                    'route_recording':route_status,
                    'target_calibration':self.targets.snapshot(),
                    'audio_enabled':self.config['audio_enabled'],'audio_fault':self.audio_fault,'last_audio':self.last_audio,
                    'running':not self.stopped and not self.stop_event.is_set(),'stopped':self.stopped,'closing':self.stop_event.is_set()}
        finally:self.lock.release()

    def mark(self,label):
        if label not in ('false_warning','useful_warning','missed_obstacle'):raise ValueError('unknown label')
        with self.log_lock:
            with open(self.folder/'labels.jsonl','a',encoding='utf-8') as f:
                f.write(json.dumps({'label':label,'at_ns':time.monotonic_ns(),'last_audio':self.last_audio})+'\n')

    def autodrive_event(self,record):
        """c099 v30d: append one autodrive supervisor transition to <recording>/autodrive-events.jsonl (2026-10-21 forensics §2:
        autodrive wrote nothing, so every start and stop had to be reconstructed from timing). Never raises and never requests a
        storage stop, so no new path can end a recording. stop_lock is held while stop_event is checked, so nothing is written after
        closeout starts (session_sync copies the folder once final-state.json exists). False: the caller keeps the event."""
        try:
            line=json.dumps(record,default=jsonable,separators=(',',':'))+'\n'
            if not self.stop_lock.acquire(timeout=.5):return False
            try:
                if self.stopped or self.stop_event.is_set():return False
                if not recording_space_available(self.folder,self.config,len(line.encode('utf-8'))):
                    self.stats['autodrive_events_unsaved']+=1;return False
                with open(self.folder/'autodrive-events.jsonl','a',encoding='utf-8') as stream:stream.write(line)
                self.stats['autodrive_events_saved']+=1;return True
            finally:self.stop_lock.release()
        except Exception:
            self.stats['autodrive_events_unsaved']+=1;return False

    def stop(self,analyze=True):
        with self.stop_lock:return self._stop(analyze)

    def _stop(self,analyze=True):
        if self.stopped:return
        self.stop_event.set()
        with self.night_start_lock:
            if self.night_companion:
                try:self.night_companion.stop()
                except Exception as error:self.error('Experimental awareness closeout: '+str(error))
        if self.map_prior:self.map_prior.close()
        ssh=self.ssh   # c099 v30d: comma_link replaces self.ssh per attempt; act on one captured child
        if ssh and ssh.poll() is None:
            # Terminate first: closing a buffered stdin can wait behind a blocked
            # ping writer. receive_comma owns pipe close/reap after stdout unblocks.
            try:ssh.terminate()
            except OSError:pass
        for t in self.threads:t.join(timeout=3)
        if getattr(self,'processing_backend','inline')=='isolated':self.engine.close()
        if self.route_journal:self.route_journal.close()
        try:self.speaker.stop()
        except (OSError,RuntimeError) as e:self.error('Audio stop failed: '+str(e))
        with self.log_lock:
            for f in (self.decision_log,self.context_log,self.comma_log,self.comma_rejected_log,self.coverage_log):f.close()
        self.audible_review.close()
        self.stopped=True
        (self.folder/'raw-capture-index.json').write_text(json.dumps({'segments':self.capture_segments,
             'note':'Host recording intervals; coverage does not prove lossless UDP or valid lidar frames.'},indent=2),encoding='utf-8')
        (self.folder/'final-state.json').write_text(json.dumps(self.snapshot(),default=jsonable,indent=2),encoding='utf-8')
        if self.ground_candidate:(self.folder/'ground-proposal.json').write_text(json.dumps(self.ground_candidate,indent=2),encoding='utf-8')
        (self.folder/'target-calibration.json').write_text(json.dumps({'pairs':self.targets.pairs,**self.targets.snapshot()},indent=2),encoding='utf-8')
        return summarize(self.folder) if analyze else {'status':'recording_closed','folder':str(self.folder)}

def replay_and_review(folder,cancel=None,progress=None):
    """Checkpointed replay; publish complete derived results atomically."""
    from learning.replay_checkpoint import run_replay,source_fingerprint,digest,VERSION
    from hindsight import write_json
    folder=Path(folder)
    fingerprint=source_fingerprint(folder,VERSION)
    working=folder.parent.parent/'learning/replay'/digest(fingerprint)
    report=run_replay(folder,working,cancel,progress=progress)
    report['mount_epoch']=json.loads((folder/'session.json').read_text(encoding='utf-8-sig')).get('mount_epoch')
    from mapping.pipeline import drive_lineage
    lineage=drive_lineage(folder,folder.parent.parent)
    report['drive_lineage']=lineage
    report['physical_drive_group']=lineage['physical_drive_group'] if lineage.get('verified') else None
    for kind,name in (('decision','recomputed-decisions.jsonl'),('context','recomputed-contexts.jsonl')):
        source=Path(report['products'][kind]['path']);temporary=folder/(name+'.learning.tmp')
        if not recording_space_available(folder,{'minimum_free_gb':10},source.stat().st_size):
            from offline_jobs import WorkPaused
            raise WorkPaused('Replay publish is waiting for disk reserve')
        try:
            with source.open('rb') as inp,temporary.open('wb') as out:
                while block:=inp.read(1024*1024):
                    check_work(cancel);out.write(block)
                out.flush();os.fsync(out.fileno())
            check_work(cancel);os.replace(temporary,folder/name)
        finally:
            if temporary.exists():temporary.unlink()
    write_json(folder/'replay-review.json',report)
    return report


def _legacy_replay_and_review(folder,cancel=None):
    """Re-decode UDP and re-normalize recorded comma data at causal frame cutoffs."""
    folder=Path(folder);config=json.loads((folder/'session.json').read_text())
    engine=Engine(Mount(**config['mount']),config.get('perception_policy'));assembler=FrameAssembler();motion=ScanMotion()
    telemetry=Telemetry();telemetry_diagnostics=Counter()
    def telemetry_records():
        path=folder/'comma.jsonl'
        if not path.exists():return
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:record=json.loads(line)
                except ValueError:telemetry_diagnostics['invalid_json_rows']+=1;continue
                if isinstance(record,dict) and type(record.get('laptop_receipt_ns')) is int:yield record
                else:telemetry_diagnostics['invalid_record_rows']+=1
    raw_iter=iter(telemetry_records());next_raw=next(raw_iter,None)
    from recording import frames as recorded_frames
    replay_diagnostics={}
    count=0;legacy_cutoffs=0;lidar_t=[];lidar_yaw=[];comma_t=[];comma_yaw=[]
    with (folder/'recomputed-decisions.jsonl').open('w',encoding='utf-8') as out:
        for points,stamp,original_context,quality in recorded_frames(folder,cancel,strict=True,diagnostics=replay_diagnostics):
            cutoff=quality.get('fusion_cutoff_ns')
            if cutoff is None:
                cutoff=quality['last_receipt_ns'];legacy_cutoffs+=1
            while next_raw is not None and next_raw['laptop_receipt_ns']<=cutoff:
                r=next_raw
                try:
                    if ingest_telemetry_record(telemetry,r,r['laptop_receipt_ns'],r.get('laptop_send_ns')):
                        telemetry_diagnostics['payloads_invalidated']+=1
                except Exception:telemetry_diagnostics['records_rejected']+=1
                next_raw=next(raw_iter,None)
            ctx,problem=safe_telemetry_context(telemetry,original_context['at_ns'])
            if problem:telemetry_diagnostics['context_failures']+=1
            decision=engine.update(points,ctx,quality,(stamp-stamp.max())/1e9)
            decision['processing_timing_scope']='offline_replay; original live engine/decode cost is unknown here'
            out.write(json.dumps(decision,default=jsonable,separators=(',',':'))+'\n');count+=1
            effective_time=int(ctx['at_ns']+np.median(stamp)-stamp.max())
            registration=motion.update(transform(points,engine.mount),effective_time)
            if registration.get('valid'):
                lidar_t.append(registration['at_ns']/1e9);lidar_yaw.append(registration['yaw_rate_rps'])
            if ctx.get('motion_ready',ctx.get('ready')) and ctx.get('motion_source_ns'):
                source_time=ctx['motion_source_ns']/1e9
                if not comma_t or source_time>comma_t[-1]:
                    comma_t.append(source_time);comma_yaw.append(ctx['yaw_rate_rps'])
    timing=timing_alignment(lidar_t,lidar_yaw,comma_t,comma_yaw) if min(len(lidar_t),len(comma_t))>=100 else {'accepted':False,'reason':'not enough independently registered turning data'}
    report={'redecoded_frames':count,'unmatched_frames':replay_diagnostics['unmatched_frames'],
            'replay_version':APP_VERSION,'recording_version':config.get('version','unknown'),
            'telemetry_record_diagnostics':dict(telemetry_diagnostics),
            'renormalized_frames':count,'legacy_conservative_cutoff_frames':legacy_cutoffs,
            'replay_scope':'raw lidar + recorded comma messages; source normalization re-run without future messages',
            'contexts_without_raw_frames':replay_diagnostics['contexts_without_raw_frames'],'timing_proposal':timing,
            'session':str(folder.resolve()),'mount':config['mount'],
            'comma_commit':json.loads((folder/'comma-device.json').read_text()).get('commit') if (folder/'comma-device.json').exists() else None,
            'total_time_adjustment_s':config['lidar_time_adjustment_s']+timing.get('offset_s',0),
            'note':'No calibration or thresholds were automatically activated. Use held-out evidence to validate a proposal.'}
    (folder/'replay-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summarize(folder,recompute_hindsight=False,cancel=cancel)
    return report


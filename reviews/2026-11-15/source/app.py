"""Odyssey FCW commissioning application. Loopback UI, no vehicle actuation."""
import argparse
import copy
import hashlib
from dataclasses import asdict
import ipaddress
import json
import math
import os
for _thread_setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_thread_setting,'1')
from pathlib import Path
import secrets
import re
import socket
import sys
import threading
import runtime_clock as time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from runtime import Session,DEFAULT_CONFIG,APP_VERSION,jsonable,replay_and_review,recording_space_available
from runtime import FORK_LABEL,build_stamp,start_build_stamp   # c099 v30d: build identity (2026-10-21 live forensics §1)
from fcw_engine import Mount
from audio_output import Speaker
from scenarios import run_demo
from calibration import combine_timing_reviews
from hindsight import validate_footprint
from policy import canonical_policy
from session_sync import SessionSync,validate_sync
from hindsight import write_json
from development_progress import load_development_progress,PRESENTATION_REVISION

BASE=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent.parent
ASSETS=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
EXAM_ROOT=BASE/'exam'
_SOURCE_ID=None

def launch_identity():
    """Version text is not a build identity: forks deliberately share APP_VERSION."""
    global _SOURCE_ID
    stamp=build_stamp()
    identity={'fork':FORK_LABEL,'version':APP_VERSION,'pid':os.getpid(),
              'frozen':bool(getattr(sys,'frozen',False)),'executable':str(Path(sys.executable).resolve()),
              'exe_sha256':stamp.get('exe_sha256'),'hash_state':stamp.get('state'),
              'bundle_sha256':stamp.get('bundle_sha256'),'bundle_state':stamp.get('bundle_state')}
    if not identity['frozen']:
        if _SOURCE_ID is None:
            digest=hashlib.sha256()
            for path in sorted(ASSETS.rglob('*.py')):
                if '__pycache__' in path.parts:continue
                digest.update(path.relative_to(ASSETS).as_posix().encode()+b'\0');digest.update(path.read_bytes())
            _SOURCE_ID=digest.hexdigest()
        identity['source_sha256']=_SOURCE_ID
    return identity

def same_running_build(existing,current):
    other=existing.get('identity') or {}
    if existing.get('app')!='OdysseyFCW' or other.get('version')!=current.get('version') or other.get('fork')!=current.get('fork'):return False
    if other.get('frozen') is not current.get('frozen'):return False
    key=('bundle_sha256' if current.get('bundle_sha256') else 'exe_sha256') if current.get('frozen') else 'source_sha256'
    value=current.get(key)
    return isinstance(value,str) and len(value)==64 and other.get(key)==value

class Application:
    def __init__(self,config_path=None,record_root=None):
        self.config_path=Path(config_path or BASE/'config.json')
        self.config=json.loads(json.dumps(DEFAULT_CONFIG))
        if self.config_path.exists():self.config.update(json.loads(self.config_path.read_text(encoding='utf-8-sig')))
        self.config['version']=APP_VERSION
        from autodrive import enforce_always_warn   # Claude fork v19: config always_warn -> forced warnings + speech on at every launch
        if enforce_always_warn(self.config) and self.config_path.exists():write_json(self.config_path,{k:v for k,v in self.config.items() if k!='version'} | {'version':self.config['version']})
        from wifi_connect import WifiConnection
        self.wifi=WifiConnection()
        self.autodrive=None
        start_build_stamp()   # c099 v30d: hash the running exe once at launch, off the frame path (forensics §1); no-op when not frozen
        # Reviewed UI-only snapshot. No Session/Engine code consumes this value.
        self.development_progress=load_development_progress(ASSETS/'development-progress.json')
        self.config['perception_policy']=canonical_policy(self.config.get('perception_policy'))
        if not 1024<=int(self.config['http_port'])<=65535:raise ValueError('Invalid local app port')
        self.record_root=Path(record_root or Path(os.environ.get('LOCALAPPDATA',BASE))/'OdysseyFCW'/'recordings')
        from selfcal import status as selfcal_status
        self.selfcal_state=selfcal_status(self.record_root,self.config)
        self.session=None;self.review_status='';self.busy=False;self.lock=threading.RLock();self.token=secrets.token_hex(24)
        self.timing_proposal=None
        self.map_review=None
        self.review_cancel=threading.Event();self.background_stop=threading.Event();self.shutdown=threading.Event()
        self.standby_pending=False;self.standby=None;self.pending_start=None;self.review_thread=None;self.maintenance_thread=None;self.closing=False
        self.learning=None;self.starting=False;self.start_gate=threading.Lock()
        self.commissioning_busy=False;self.commissioning_thread=None;self.commissioning_proposal=None
        self.commissioning_job={'state':'idle','message':'Measure the installation, then import independently reviewed physical evidence.'}
        self.last_review_folder=None;self.pending_review_count=0
        self.config['sync']=validate_sync(self.config.get('sync',DEFAULT_CONFIG['sync']),self.record_root)
        self.syncer=SessionSync(self.record_root,self.record_root.parent,lambda:self.config['sync'],self.activity,self.lock)
        previous=self.review_folder_path()
        if previous and (previous/'map-review.json').exists():
            try:
                full_map=json.loads((previous/'map-review.json').read_text(encoding='utf-8'))
                self.map_review={k:full_map[k] for k in ('status','reason','revision','backend','coverage_percent','localized_keyframes','eligible_keyframes','new_submaps','promoted_static_voxels','forgotten_voxels') if k in full_map}
                self.last_review_folder=previous
            except (ValueError,OSError):pass

    def activity(self):return self.standby_pending or self.starting or self.busy or self.commissioning_busy or bool(self.session and not self.session.stopped)

    @staticmethod
    def _config_hash(config):
        return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def _invalidate_commissioning(self,cfg):
        from commissioning import profile_template
        from object_fusion import default_fusion_calibration
        cfg['fcw_commissioning']=profile_template(cfg)
        cfg['sensor_fusion']=default_fusion_calibration(cfg.get('mount_epoch'))
        cfg.pop('commissioning_import_receipt',None)
        cfg.pop('empirical_calibration',None)
        cfg.pop('empirical_previous',None)
        self.selfcal_state={'state':'ready_to_record','proposal':None,'active_profile_id':None,
                           'message':'Mounting setup changed. A new ordinary drive can estimate calibration.'}
        self.commissioning_proposal=None
        self.commissioning_job={'state':'idle','message':'Geometry or timing changed. Independent capability evidence must be checked again.'}

    def _refresh_selfcal(self):
        from selfcal import status
        with self.lock:cfg=copy.deepcopy(self.config)
        state=status(self.record_root,cfg)
        with self.lock:
            if self.config.get('mount_epoch')==cfg.get('mount_epoch'):self.selfcal_state=state

    def _apply_selfcal(self,data):
        from selfcal import latest_report
        from selfcal_capture import _fingerprints
        from empirical_calibration import from_report,check_active
        if set(data)!={'id','digest'}:raise ValueError('Choose the current reviewed data-derived calibration')
        with self.lock:
            if self.activity() or self.closing:raise ValueError('Stop recording and wait for analysis before applying calibration')
            found=latest_report(self.record_root,self.config.get('mount_epoch'))
            if not found:raise ValueError('No current-mount calibration report is ready')
            report=found['report']
            if (data['id'],data['digest'])!=(report['id'],report['digest']):raise ValueError('Calibration proposal changed; review the current result')
            active=self.config.get('empirical_calibration') or {}
            if (active.get('id')==report['id'] and active.get('source',{}).get('report_digest')==report['digest'] and
                check_active(self.config,self.record_root).get('trusted')):
                return {'ok':True,'already_applied':True,'profile_id':active['id'],
                        'message':'This calibration is already applied; its rollback copy was preserved.'}
            folder=self.record_root/found['folder']
            if _fingerprints(folder,None)!=report['source_files']:raise ValueError('Source data changed since calibration analysis')
            profile=from_report(report,self.config,folder)
            cfg=copy.deepcopy(self.config)
            keys=('mount','empirical_calibration','fcw_commissioning','commissioning_import_receipt',
                  'sensor_fusion','lidar_time_adjustment_s','lidar_timing_bound_s','timing_verified','calibration_basis')
            previous={key:copy.deepcopy(cfg[key]) for key in keys if key in cfg}
            self._invalidate_commissioning(cfg)
            cfg.update(mount=profile['mount'],empirical_calibration=profile,
                       empirical_previous=previous,timing_verified=False,lidar_timing_bound_s=None,
                       lidar_time_adjustment_s=profile['lidar_time_adjustment_s'],
                       calibration_basis='sensor_derived_empirical')
            artifact=self.record_root.parent/profile['source']['applied_report']
            expected=self.record_root.parent/'calibration'/'empirical'/(profile['id']+'.json')
            if artifact.resolve()!=expected.resolve():raise ValueError('Unexpected calibration artifact destination')
            for parent in (artifact.parent.parent,artifact.parent):
                if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('Calibration artifact directory cannot be linked')
            artifact.parent.mkdir(parents=True,exist_ok=True)
            if artifact.exists():
                if json.loads(artifact.read_text(encoding='utf-8'))!=report:raise ValueError('Existing calibration artifact differs')
            else:write_json(artifact,report)
            proof=check_active(cfg,self.record_root)
            if not proof['trusted']:raise ValueError('; '.join(proof['reasons']))
            backup=self.config_path.with_name('config-before-selfcal-'+profile['id']+'.json')
            if not backup.exists():write_json(backup,self.config)
            write_json(self.config_path,cfg);self.config=cfg;self.session=None;self.timing_proposal=None
        self._refresh_selfcal()
        return {'ok':True,'profile_id':profile['id'],'method':'sensor_derived_empirical',
                'message':'Data-derived calibration applied. Live coverage remains limited to the supported motion and uncertainty.'}

    def _rollback_selfcal(self):
        with self.lock:
            if self.activity() or self.closing:raise ValueError('Stop recording before rolling back calibration')
            previous=self.config.get('empirical_previous')
            if not isinstance(previous,dict):raise ValueError('No previous calibration is available')
            old_epoch=(previous.get('empirical_calibration') or {}).get('mount_epoch')
            if old_epoch and old_epoch!=self.config.get('mount_epoch'):raise ValueError('Previous calibration belongs to a different mounting period')
            cfg=copy.deepcopy(self.config)
            for key in ('mount','empirical_calibration','fcw_commissioning','commissioning_import_receipt',
                        'sensor_fusion','lidar_time_adjustment_s','lidar_timing_bound_s','timing_verified','calibration_basis'):
                if key in previous:cfg[key]=copy.deepcopy(previous[key])
                else:cfg.pop(key,None)
            cfg.pop('empirical_previous',None)
            write_json(self.config_path,cfg);self.config=cfg;self.session=None
        self._refresh_selfcal()
        return {'ok':True,'message':'Previous calibration restored; recordings and proposals retained.'}

    def commissioning_snapshot(self):
        from commissioning import workflow_status
        proposal=self.commissioning_proposal
        return {'workflow':workflow_status(self.config),'job':dict(self.commissioning_job),'busy':self.commissioning_busy,
                'proposal':{'id':proposal['id'],'eligible':proposal['receipt']['eligible'],
                    'reasons':proposal['receipt'].get('reasons',[]),
                    'validation':proposal['receipt'].get('validation'),
                    'summary':proposal['receipt'].get('limitation','Incomplete evidence; cannot apply.'),
                    'source_sessions':proposal['receipt'].get('source_sessions',[])} if proposal else None,
                'measurements':{'mount':copy.deepcopy(self.config['mount']),'vehicle_reference':copy.deepcopy(self.config.get('vehicle_reference',{}))},
                'field_target_validated':False,
                'note':'Preliminary build. The four-second / 125 m target and nuisance rate are not yet validated on physical drives.'}

    def _start_commissioning_job(self,kind,payload):
        """Hash large physical sources outside HTTP/UI and recording locks."""
        with self.lock:
            if self.activity() or self.closing:raise ValueError('Stop collection and wait for background review before commissioning')
            if kind=='apply':
                proposal=self.commissioning_proposal
                if not proposal or payload.get('id')!=proposal['id'] or proposal['receipt'].get('eligible') is not True:
                    raise ValueError('Choose the current eligible reviewed commissioning proposal')
                if self._config_hash(self.config)!=proposal['receipt']['config_before_sha256']:
                    raise ValueError('Settings changed since import; import the evidence again')
                job_payload=copy.deepcopy(proposal)
            else:job_payload=copy.deepcopy(payload)
            before=copy.deepcopy(self.config);job_id=uuid.uuid4().hex
            self.commissioning_busy=True
            self.commissioning_job={'id':job_id,'state':'verifying' if kind=='import' else 'applying',
                'message':'Checking source recordings and evidence hashes. No settings have changed.'}
            if kind=='import':self.commissioning_proposal=None
        def work():
            try:
                if self.learning:self.learning.quiesce()
                from commissioning_store import import_bundle
                if kind=='import':
                    candidate,receipt=import_bundle(before,self.record_root,job_payload)
                    with self.lock:
                        if self.closing:raise ValueError('Application closing; nothing applied')
                        if self._config_hash(self.config)!=self._config_hash(before):raise ValueError('Settings changed during verification; import again')
                        self.commissioning_proposal={'id':job_id,'candidate':candidate,'receipt':receipt,'bundle':job_payload}
                        self.commissioning_job={'id':job_id,'state':'review_ready' if receipt.get('eligible') else 'incomplete',
                            'message':'Evidence checked. Review the proposal, then click Apply.' if receipt.get('eligible') else 'Evidence is incomplete; nothing was enabled.'}
                else:
                    self._apply_commissioning_proposal(before,job_payload,job_id)
            except Exception as error:
                with self.lock:self.commissioning_job={'id':job_id,'state':'failed','message':str(error)}
            finally:
                with self.lock:self.commissioning_busy=False
        self.commissioning_thread=threading.Thread(target=work,daemon=True)
        self.commissioning_thread.start()
        return {'ok':True,'job_id':job_id,'state':self.commissioning_job['state']}

    def _apply_commissioning_proposal(self,before,proposal,job_id):
        from commissioning_store import import_bundle,validate_import_apply
        check=validate_import_apply(before,proposal['candidate'],proposal['receipt'],self.record_root)
        if not check.get('valid'):raise ValueError('; '.join(check.get('reasons',[])))
        # Explicit Apply rehashes the physical sources in this worker. Uploaded
        # paths never become arbitrary reads; import_bundle scopes every source.
        candidate,receipt=import_bundle(before,self.record_root,proposal['bundle'])
        check=validate_import_apply(before,candidate,receipt,self.record_root)
        if not check.get('valid'):raise ValueError('; '.join(check.get('reasons',[])))
        with self.lock:
            if self.closing or self.starting or self.busy or (self.session and not self.session.stopped):
                raise ValueError('Application became active; nothing applied')
            if self._config_hash(self.config)!=receipt['config_before_sha256']:
                raise ValueError('Settings changed during verification; nothing applied')
            # Only bounded, verified new JSON artifacts may be written. Existing
            # recordings and calibration imports are never overwritten.
            for artifact in receipt['artifact_writes']:
                target=Path(artifact['path']);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(artifact['content'].encode('utf-8'));stream.flush();os.fsync(stream.fileno())
            check=validate_import_apply(self.config,candidate,receipt,self.record_root)
            if not check.get('valid'):raise ValueError('; '.join(check.get('reasons',[])))
            backup=self.config_path.with_name('config-before-commissioning-'+job_id+'.json')
            with backup.open('x',encoding='utf-8') as stream:json.dump(before,stream,indent=2,allow_nan=False)
            write_json(self.config_path,candidate);self.config=candidate
            self.commissioning_proposal=None;self.session=None;self.timing_proposal=None
            self.commissioning_job={'id':job_id,'state':'applied','message':'Reviewed commissioning evidence applied. Physical warning range and nuisance performance remain unvalidated.'}

    def review_folder_path(self):
        if self.session and (self.session.folder/'review.html').exists():return self.session.folder
        if self.last_review_folder:return self.last_review_folder
        paths=sorted(self.record_root.glob('*/review.html'),reverse=True) if self.record_root.exists() else []
        return paths[0].parent if paths else None

    def pending_reviews(self):
        from postprocess import session_signature
        pending=[]
        if not self.record_root.exists():return pending
        if not recording_space_available(self.record_root,self.config,16*1024*1024):
            self.review_status='Offline review waiting for disk space above the recording reserve.'
            return pending
        for folder in sorted(self.record_root.iterdir()):
            if not folder.is_dir() or folder.resolve().parent!=self.record_root.resolve() or not (folder/'final-state.json').exists():continue
            try:
                cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
                if cfg.get('synthetic_fixture') or cfg.get('capture_mode') not in ('live','comma','demo'):continue
                progress=json.loads((folder/'offline-status.json').read_text()) if (folder/'offline-status.json').exists() else {}
                if progress.get('state') not in ('complete','complete_with_errors') or progress.get('engine_version')!=APP_VERSION or progress.get('source_signature')!=session_signature(folder):pending.append(folder)
            except (OSError,ValueError):continue
        return pending

    def background_mode(self):
        mode=self.config.get('background_analysis_mode','full')
        return mode if mode in ('full','calibration_only','off') else 'off'

    def start_background(self):
        mode=self.background_mode()
        if mode=='off' or mode=='calibration_only' and not self.config.get('self_calibration_enabled',True):
            self.review_status='Recording and warnings only; heavy offline analysis runs on the Macs.'
            return
        if self.maintenance_thread:return
        if mode=='calibration_only' or self.config.get('learning',{}).get('enabled',True):
            from learning.coordinator import Coordinator
            self.learning=Coordinator(self,ASSETS,APP_VERSION,allowed_stages={'self_calibration'} if mode=='calibration_only' else None)
            if self.config['sync']['enabled']:
                self.syncer.state.update(state='paused',message='File sync is paused while the bounded learning scheduler is enabled.')
        else:self.syncer.start()
        def maintenance():
            while not self.background_stop.is_set():
                if self.learning:
                    try:self.learning.tick()
                    except Exception as error:
                        self.review_status='Offline learning unavailable: '+str(error)
                    self.background_stop.wait(2);continue
                if not self.activity():
                    pending=self.pending_reviews();self.pending_review_count=len(pending)
                    if pending:
                        with self.lock:
                            if self.activity() or self.closing:continue
                            self.busy=True;self.review_cancel.clear();self.review_thread=threading.current_thread()
                        self._review(pending[0])
                self.background_stop.wait(3)
        self.maintenance_thread=threading.Thread(target=maintenance,daemon=True);self.maintenance_thread.start()

    def close_background(self):
        if self.standby:self.standby.close()
        self.closing=True;self.wifi.close();self.background_stop.set();self.review_cancel.set();self.syncer.close()
        if self.learning:self.learning.close()
        for thread in (self.review_thread,self.maintenance_thread,self.commissioning_thread):
            if thread and thread is not threading.current_thread():thread.join(timeout=3)

    def save(self,values):
        with self.lock:
            if self.activity():raise ValueError('Stop collection and wait for review before changing setup')
            cfg=json.loads(json.dumps(self.config))
            for key in ('sensor_ip','destination_ip','bind_ip'):
                if key in values:cfg[key]=str(ipaddress.IPv4Address(values[key]))
            for key in ('comma_host','ssh_key'):
                if key in values:cfg[key]=str(values[key]).strip()
            for key in ('msop_port','difop_port'):
                if key in values:
                    cfg[key]=int(values[key])
                    if not 1024<=cfg[key]<=65535:raise ValueError('UDP ports must be 1024–65535')
            if cfg['msop_port']==cfg['difop_port']:raise ValueError('Use separate MSOP/DIFOP ports')
            if 'record_limit_gb' in values:
                cfg['record_limit_gb']=float(values['record_limit_gb'])
                if not math.isfinite(cfg['record_limit_gb']) or not 0<=cfg['record_limit_gb']<=1000:raise ValueError('Choose 0 for available disk space, or up to 1000 GB')
            if 'mount_height_m' in values:
                height=float(values['mount_height_m'])
                if abs(height-cfg['mount']['xyz'][2])>.001:
                    cfg['mount']['alignment_verified']=False;cfg['mount']['xyz'][2]=height
                    self._invalidate_commissioning(cfg)
            Mount(**cfg['mount']).validate()
            cfg['audio_enabled']=bool(values.get('audio_enabled',cfg['audio_enabled']))
            if 'map_enabled' in values:cfg['map_enabled']=bool(values['map_enabled'])
            if 'hindsight_footprint' in values:
                cfg['hindsight_footprint']=validate_footprint(values['hindsight_footprint'])
            if 'sync' in values:cfg['sync']=validate_sync(values['sync'],self.record_root)
            write_json(self.config_path,cfg);self.config=cfg

    def start(self,mode):
        if mode not in ('live','demo','comma'):raise ValueError('Unknown mode')
        if mode=='comma' and not self.config['comma_host']:raise ValueError('Enter the comma IP and save setup first')
        with self.start_gate:
            with self.lock:
                if self.closing:raise ValueError('Application is closing')
                if self.standby_pending:raise ValueError('Wake the tablet before starting recording')
                if self.commissioning_busy:raise ValueError('Wait for commissioning verification to finish before recording')
                if self.busy:
                    self.pending_start=mode;self.review_cancel.set();self.review_status='Pausing offline work; the requested recording will start automatically.';return
                if self.session and not self.session.stopped:raise ValueError('A collection is already active')
                self.starting=True
            try:
                # Never wait on optional workers while holding the capture lock.
                if self.learning:self.learning.quiesce()
                with self.lock:
                    if self.closing:raise ValueError('Application is closing')
                    if mode=='live' and self.config.get('self_calibration_enabled',True) and not self.config.get('mount_epoch'):
                        self.config['mount_epoch']='automatic-mount-'+uuid.uuid4().hex
                        self._invalidate_commissioning(self.config)
                        write_json(self.config_path,self.config)
                    self.review_cancel.clear()
                    self.session=Session(self.config,ASSETS,self.record_root,mode)
                    self.session.start()
                    if mode=='demo':self.session.spawn(run_demo,self.session)
                    self.review_status=''
            finally:
                with self.lock:self.starting=False

    def stop(self):
        with self.lock:
            if not self.session or self.session.stopped or self.busy:return
            self.busy=True;session=self.session;self.review_status='Finishing logs and automatic review…'
            self.review_cancel.clear()
        def work():
            try:session.stop(analyze=False)
            except Exception as e:
                self.review_status='Could not finish recording: '+str(e);self._finish_review();return
            if self.background_mode()!='full':
                self.review_status='Recording saved; bounded calibration may run while idle, heavy analysis is offloaded to the Macs.' if self.background_mode()=='calibration_only' else 'Recording saved; heavy offline analysis is offloaded to the Macs.'
                self._finish_review()
                if self.learning and not self.activity():self.learning.resume_idle()
                return
            if self.learning:
                self.last_review_folder=session.folder
                self.review_status='Recording closed. Automatic analysis is queued and will pause for the next recording.'
                self._finish_review()
                if not self.activity():self.learning.resume_idle()
            else:self._review(session.folder)
        self.review_thread=threading.Thread(target=work,daemon=True);self.review_thread.start()

    def _finish_review(self):
        with self.lock:
            self.busy=False;wanted=self.pending_start;self.pending_start=None
        if wanted and not self.closing:
            try:self.start(wanted)
            except Exception as e:self.review_status='Could not start recording: '+str(e)

    def offline_completed(self,job):
        """Refresh small review metadata between recordings, never raw data."""
        if job.get('state')=='complete' and job.get('stage')=='self_calibration':
            self._refresh_selfcal();return
        if job.get('state')!='complete' or job.get('stage') not in ('raw_replay','atlas','review'):return
        from learning.pipeline import session_path
        folder=session_path(self.record_root,job['payload']['session'])
        epoch=self.config.get('mount_epoch');map_summary=None
        proposal=self._current_timing_proposal(epoch)
        path=folder/'map-review.json'
        if path.exists() and path.stat().st_size<=1_000_000:
            full=json.loads(path.read_text(encoding='utf-8'))
            map_summary={k:full[k] for k in ('status','reason','revision','backend','coverage_percent','localized_keyframes','eligible_keyframes','new_submaps','promoted_static_voxels','forgotten_voxels') if k in full}
        with self.lock:
            if self.activity():return
            if self.config.get('mount_epoch')!=epoch:return
            self.timing_proposal=proposal
            if (folder/'review.html').is_file() and (not self.last_review_folder or folder.name>=self.last_review_folder.name):
                self.last_review_folder=folder
            if map_summary is not None:self.map_review=map_summary
            self.review_status='Offline '+job['stage'].replace('_',' ')+' completed; original recordings and warning checks retained.'

    def _current_timing_proposal(self,epoch):
        by_drive={}
        for path in sorted(self.record_root.glob('*/replay-review.json'),reverse=True)[:20]:
            try:
                if path.stat().st_size>1_000_000:continue
                report=json.loads(path.read_text(encoding='utf-8'))
                group=report.get('physical_drive_group')
                if epoch and report.get('mount_epoch')==epoch and report.get('drive_lineage',{}).get('verified') is True and isinstance(group,str) and group:
                    by_drive.setdefault(group,report)
            except (OSError,ValueError):pass
        proposal=combine_timing_reviews(list(by_drive.values())) if len(by_drive)>=2 else {
            'accepted':False,'reason':'Timing evidence from two distinct reviewed driving episodes in the current mounting period is required.'}
        return {**proposal,'mount_epoch':epoch,'physical_drive_groups':sorted(by_drive)}

    def _review(self,folder):
        from postprocess import run_offline
        try:
            if not recording_space_available(folder,self.config,16*1024*1024):
                self.review_status='Offline review waiting for disk space above the recording reserve.'
                return
            outcome=run_offline(folder,self.record_root.parent,EXAM_ROOT,lambda text:setattr(self,'review_status',text),self.review_cancel.is_set)
            if outcome['state']=='paused':self.review_status='Offline review paused; it will resume when idle.'
            else:
                self.last_review_folder=Path(folder);map_file=Path(folder)/'map-review.json'
                if map_file.exists():
                    full_map=json.loads(map_file.read_text())
                    self.map_review={k:full_map[k] for k in ('status','reason','revision','backend','coverage_percent','localized_keyframes','eligible_keyframes','new_submaps','promoted_static_voxels','forgotten_voxels') if k in full_map}
                self.timing_proposal=self._current_timing_proposal(self.config.get('mount_epoch'))
                self._refresh_selfcal()
                self.review_status='Review ready. False-alarm and miss evidence saved; settings unchanged.'
                if outcome['state']=='complete_with_errors':self.review_status='Review saved with an offline-stage error; see offline-status.json. Original recording retained.'
        except Exception as e:self.review_status='Review failed: '+str(e)
        finally:self._finish_review()

    def snapshot(self):
        from proposals import review_state
        with self.lock:
            return {'session':self.session.snapshot() if self.session else None,'running':bool(self.session and not self.session.stopped),
                    'verification_fixture':bool(self.config.get('synthetic_fixture')),
                    'night_awareness_mode':self.config.get('night_awareness_mode','off'),
                    'night_awareness_status':self.session.night_companion.snapshot(time.monotonic_ns()) if self.session and self.session.night_companion else None,
                    'forced_mode':isinstance(self.config.get('forced_experiment'),dict) and self.config['forced_experiment'].get('enabled') is True,
                    'busy':self.busy,'review_status':self.review_status,
                    'auto':{'record':bool(self.config.get('auto_record')),'always_warn':bool(self.config.get('always_warn')),'state':self.autodrive.state if self.autodrive else None},
                    'build':build_stamp(),   # c099 v30d: /state names the build with or without a session (forensics §1)
                    'wifi':self.wifi.snapshot(),
                    'standby':self.standby.snapshot() if self.standby else {'pending':False,'state':'awake'},
                    'starting':self.starting,'learning':self.learning.snapshot() if self.learning else {'state':'offloaded_to_compute_hosts' if self.background_mode()!='full' else 'disabled_by_launch_mode'},
                    'background_analysis':{'enabled':self.background_mode()=='full','mode':self.background_mode(),'allowed_stages':['self_calibration'] if self.background_mode()=='calibration_only' else [],'message':'Bounded calibration may run while idle; heavy analysis runs on the Macs.' if self.background_mode()=='calibration_only' else 'Heavy offline analysis runs on the Macs; live warning and calibration-readiness checks remain active.'},
                    'mount_epoch':self.config.get('mount_epoch'),
                    'timing_proposal':self.timing_proposal,
                    'map_review':self.map_review,
                    'commissioning':self.commissioning_snapshot(),
                    'self_calibration':{**copy.deepcopy(self.selfcal_state),'automatic_analysis':bool(self.background_mode()!='off' and self.selfcal_state.get('automatic_analysis',True))},
                    'self_calibration_rollback_available':isinstance(self.config.get('empirical_previous'),dict),
                    'proposal':review_state(self.record_root.parent,APP_VERSION,self.config['perception_policy']),
                    'perception_policy':self.config['perception_policy'],
                    'sync_settings':self.config['sync'],'sync_status':dict(self.syncer.state),'pending_reviews':self.pending_review_count,
                    'config':{k:self.config[k] for k in ('sensor_ip','destination_ip','bind_ip','comma_host','ssh_key','msop_port','difop_port','record_limit_gb','audio_enabled','map_enabled')},
                    'hindsight_footprint':self.config['hindsight_footprint'],
                    'height_m':self.config['mount']['xyz'][2],'height_note':'Initial roof estimate: Honda 69.6 in + half of M1 45 mm; mount thickness negligible. Not a measured optical origin.'}

    def action(self,route,data):
        if route=='/connect-comma-wifi':
            if data:raise ValueError('This action uses only the saved comma Wi-Fi profile')
            if self.closing:raise ValueError('Application is closing')
            return {'ok':True,'wifi':self.wifi.start()}
        elif route=='/configure':self.save(data)
        elif route=='/selfcal/apply':return self._apply_selfcal(data)
        elif route=='/selfcal/rollback':
            if data:raise ValueError('Rollback takes no replacement parameter values')
            return self._rollback_selfcal()
        elif route=='/selfcal/refresh':
            self._refresh_selfcal();return {'ok':True}
        elif route=='/commissioning/measurements':
            from commissioning_store import save_measurements
            with self.lock:
                if self.activity():raise ValueError('Stop collection and wait for verification before saving measurements')
                cfg=save_measurements(self.config,data)
                cfg.pop('empirical_calibration',None);cfg.pop('empirical_previous',None)
                write_json(self.config_path,cfg);self.config=cfg;self.session=None;self.timing_proposal=None
                self.selfcal_state={'state':'ready_to_record','proposal':None,'active_profile_id':None,
                                   'message':'Geometry changed. Collect a new ordinary drive for data-derived setup.'}
                self.commissioning_proposal=None
                self.commissioning_job={'state':'idle','message':'Measurements saved in a new mounting period. Physical verification is still required.'}
        elif route=='/commissioning/import':return self._start_commissioning_job('import',data)
        elif route=='/commissioning/apply':
            if set(data)!={'id'}:raise ValueError('Apply requires only the reviewed proposal identity')
            return self._start_commissioning_job('apply',data)
        elif route=='/new-mount-epoch':
            with self.lock:
                if self.activity():raise ValueError('Stop collection before recording a mounting change')
                cfg=json.loads(json.dumps(self.config))
                cfg['mount_epoch']='reported-remount-'+uuid.uuid4().hex
                cfg['mount']['measured']=False;cfg['mount']['alignment_verified']=False
                cfg['timing_verified']=False;cfg['lidar_timing_bound_s']=None
                self._invalidate_commissioning(cfg)
                write_json(self.config_path,cfg);self.config=cfg
                # A stopped Session can contain old target/ground proposals.
                # Its original evidence remains on disk; none may be applied to
                # the newly reported physical mounting period.
                self.session=None;self.timing_proposal=None
        elif route=='/night-awareness':
            mode=data.get('mode')
            if mode not in ('off','auto','on'):raise ValueError('Choose Off, Auto or On')
            with self.lock:
                cfg=copy.deepcopy(self.config);cfg['night_awareness_mode']=mode
                write_json(self.config_path,cfg);self.config=cfg
                session=self.session
                if session and not session.stopped:session.config['night_awareness_mode']=mode
            if session and not session.stopped and mode!='off' and session.night_companion is None:
                threading.Thread(target=session._start_night_companion,daemon=True,name='start-night-awareness').start()
            return {'night_awareness_mode':mode,'experimental_noisy':True,'geometric_fcw_unchanged':True}
        elif route=='/network-info':
            from wifi_identity import connected_ssids
            return connected_ssids()
        elif route=='/sync-now':
            if self.learning:raise ValueError('File sync is paused until it can share the bounded worker with automatic learning')
            threading.Thread(target=self.syncer.run_once,daemon=True).start()
        elif route=='/sleep':
            if data!={'confirmed':True}:raise ValueError('Confirm Sleep tablet first')
            from standby_control import Standby
            with self.lock:
                if self.standby is None:self.standby=Standby(self)
            self.standby.request()
            return {'ok':True,'message':'Standby requested; FCW will remain open and re-arm on wake.'}
        elif route=='/quit':
            if self.starting:raise ValueError('Recording is starting; wait for startup to finish before quitting')
            if self.session and not self.session.stopped:raise ValueError('Stop recording before quitting')
            from app_lifecycle import mark_manual_quit
            mark_manual_quit(BASE)
            self.shutdown.set()
        elif route in ('/apply-proposal','/rollback-policy'):
            with self.lock:
                if self.activity():raise ValueError('Stop recording and wait for review before changing settings')
                from proposals import apply_proposal,rollback_policy
                cfg=apply_proposal(self.config_path,self.record_root.parent,EXAM_ROOT,data.get('id'),APP_VERSION) if route=='/apply-proposal' else rollback_policy(self.config_path,self.record_root.parent)
                self.config=json.loads(json.dumps(DEFAULT_CONFIG));self.config.update(cfg);self.config['version']=APP_VERSION
        elif route=='/start':self.start(data.get('mode','live'))
        elif route=='/stop':
            if self.autodrive:self.autodrive.manual_stop()   # v19: a hand-pressed Stop holds auto-start until the lidar goes quiet
            self.stop()
        elif route=='/speaker-test':
            if self.session and not self.session.stopped:raise ValueError('Stop collection before testing speakers')
            Speaker(ASSETS/'audio').say('AUDIO TEST')
        elif route=='/forced-off':
            if data:raise ValueError('Disable forced mode takes no parameters')
            with self.lock:
                cfg=copy.deepcopy(self.config);cfg['audio_enabled']=False
                if isinstance(cfg.get('forced_experiment'),dict):cfg['forced_experiment']['enabled']=False
                if self.session:
                    with self.session.lock:
                        self.session.config['audio_enabled']=False;self.session.speaker.stop()
                        if not self.session.stopped:self.session.coverage_event({'kind':'user_disabled_forced_audio','receipt_ns':time.monotonic_ns()})
                write_json(self.config_path,cfg);self.config=cfg
            return {'ok':True,'message':'Audio muted now. Forced mode is disabled for the next connection.'}
        elif route=='/forced-on':   # Claude fork v19: undo /forced-off from the simple page (confirmation in the page)
            if not isinstance(self.config.get('forced_experiment'),dict):raise ValueError('Forced warnings are not set up on this install (no forced_experiment block in config.json)')
            with self.lock:
                cfg=copy.deepcopy(self.config);cfg['audio_enabled']=True
                if isinstance(cfg.get('forced_experiment'),dict):cfg['forced_experiment']['enabled']=True
                running=bool(self.session and not self.session.stopped)
                if self.session:
                    with self.session.lock:self.session.config['audio_enabled']=True
                write_json(self.config_path,cfg);self.config=cfg
            return {'ok':True,'message':'Warnings and speech are back on.'+(' Press Stop drive, then Start drive, to apply them to this drive.' if running else '')}
        elif route=='/audible-label':
            if set(data)!={'session_id','event_id','label'}:raise ValueError('Unexpected warning review fields')
            with self.lock:
                if self.session and not self.session.stopped:raise ValueError('Stop recording before reviewing warnings')
                folder=self.review_folder_path()
                if not folder:raise ValueError('No session selected for review')
                from audible_review import label_event
                return label_event(folder,data['session_id'],data['event_id'],data['label'])
        elif route=='/mark':
            if not self.session:raise ValueError('No collection running')
            self.session.mark(data.get('label'))
        elif route=='/apply-ground':
            with self.lock:
                self._require_current_calibration_session()
                p=self.session.ground_candidate
                if not p or not p.get('accepted'):raise ValueError('No stable parked-ground proposal yet')
                cfg=json.loads(json.dumps(self.config));cfg['mount']=p['mount_proposal']
                Mount(**cfg['mount']).validate()
                cfg['timing_verified']=False;cfg['lidar_timing_bound_s']=None
                self._invalidate_commissioning(cfg)
                write_json(self.config_path,cfg);self.config=cfg;self.timing_proposal=None
        elif route in ('/target-baseline','/target-capture','/target-check'):
            if not self.session or self.session.stopped or self.session.mode!='live':raise ValueError('Start a parked live collection first')
            with self.session.lock:
                if route=='/target-baseline':return self.session.targets.set_baseline(time.monotonic_ns())
                return self.session.targets.capture([float(data['x']),float(data['y'])],time.monotonic_ns(),route=='/target-check')
        elif route=='/apply-alignment':
            with self.lock:
                self._require_current_calibration_session()
                cfg=json.loads(json.dumps(self.config));cfg['mount']=self.session.targets.resulting_mount(self.session.engine.mount)
                Mount(**cfg['mount']).validate()
                cfg['timing_verified']=False;cfg['lidar_timing_bound_s']=None
                self._invalidate_commissioning(cfg)
                write_json(self.config_path,cfg);self.config=cfg;self.timing_proposal=None
        elif route=='/apply-timing':
            with self.lock:
                if self.activity():raise ValueError('Stop collection and wait for review first')
                p=self.timing_proposal
                if not p or not p.get('accepted'):raise ValueError('No consistent independent timing proposal yet')
                if not self.config.get('mount_epoch') or p.get('mount_epoch')!=self.config.get('mount_epoch'):
                    raise ValueError('Timing evidence must match the current known mounting period')
                groups=p.get('physical_drive_groups')
                if not isinstance(groups,list) or not all(isinstance(g,str) and g for g in groups) or len(set(groups))<2:
                    raise ValueError('Timing evidence needs two distinct reviewed physical driving episodes')
                if p['mount']!=self.config['mount']:raise ValueError('Mount changed; timing proposal needs new compatible data')
                cfg=json.loads(json.dumps(self.config))
                cfg['lidar_time_adjustment_s']=p['adjustment_s'];cfg['lidar_timing_bound_s']=p['estimated_error_s'];cfg['timing_verified']=True
                self._invalidate_commissioning(cfg)
                write_json(self.config_path.with_name('timing-evidence.json'),p)
                write_json(self.config_path,cfg);self.config=cfg
        else:raise ValueError('Unknown action')
        return {'ok':True}

    def _require_current_calibration_session(self):
        if self.starting or self.busy or self.commissioning_busy or not self.session or not self.session.stopped:
            raise ValueError('Stop collection and finish closeout before applying calibration')
        epoch=self.config.get('mount_epoch')
        if not epoch or self.session.config.get('mount_epoch')!=epoch:
            raise ValueError('Calibration evidence must be collected in the current known mounting period')
        if self.session.config.get('mount')!=self.config['mount']:
            raise ValueError('Mount setup changed; collect new calibration evidence for this geometry')

def handler(app,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def setup(self):
            self.request.settimeout(5);self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);super().setup()
        def valid_host(self):return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')
        def respond(self,content,mime='application/json',status=200):
            if not isinstance(content,bytes):content=json.dumps(content,default=jsonable,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(content)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError,TimeoutError):pass
        def do_GET(self):
            if not self.valid_host():self.send_error(403);return
            static_page=self.path in ('/','/advanced','/state','/health','/development-progress','/pipeline','/commissioning/template') or self.path.startswith('/development-evidence/')
            folder=app.review_folder_path() if not static_page else None
            if self.path in ('/','/advanced'):   # Claude fork v19: '/' = simple driver page, '/advanced' = the full commissioning dashboard
                page=ASSETS/('simple.html' if self.path=='/' and (ASSETS/'simple.html').is_file() else 'dashboard.html')
                self.respond(page.read_text(encoding='utf-8').replace('__TOKEN__',app.token).replace('__INSTANCE__',str(os.getpid())).encode(),'text/html; charset=utf-8')
            elif self.path=='/health':self.respond({'app':'OdysseyFCW','version':APP_VERSION,'fork':FORK_LABEL,
                'presentation_revision':PRESENTATION_REVISION,'identity':launch_identity()})
            elif self.path=='/state':self.respond(app.snapshot())
            elif self.path=='/commissioning/template':
                from commissioning import profile_template
                with app.lock:template=profile_template(app.config)
                self.respond({'profile':template,'evidence':[]})
            elif self.path=='/development-progress':self.respond(app.development_progress)
            elif self.path=='/pipeline':self.respond((ASSETS/'pipeline.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path.startswith('/development-evidence/'):
                name=self.path[len('/development-evidence/'):]
                allowed={item['file'] for item in app.development_progress.get('evidence',{}).values()}
                if name in allowed:
                    self.respond((ASSETS/'development-evidence'/name).read_bytes(),'text/plain; charset=utf-8')
                else:self.send_error(404)
            elif self.path=='/audible-review' and folder:
                from audible_review import page
                self.respond(page(folder,app.token).encode(),'text/html; charset=utf-8')
            elif folder and re.fullmatch(r'/audible-images/[0-9a-f]{32}-(before|at|after)\.jpg',self.path):
                image=folder/self.path.lstrip('/')
                if image.is_file() and not image.is_symlink() and image.resolve().is_relative_to(folder.resolve()):self.respond(image.read_bytes(),'image/jpeg')
                else:self.send_error(404)
            elif self.path in ('/audible-warning-events.jsonl','/audible-review.json') and folder:
                path=folder/self.path.lstrip('/')
                if path.is_file():self.respond(path.read_bytes())
                else:self.send_error(404)
            elif self.path=='/review' and folder and (folder/'review.html').exists():
                self.respond((folder/'review.html').read_bytes(),'text/html; charset=utf-8')
            elif folder and (self.path in ('/auto-labels.jsonl','/coverage-events.jsonl','/comma-rejected.jsonl','/hindsight-review.json','/map-review.json','/route-review.json','/route-gps.geojson','/atlas-route.json','/route-trajectory.jsonl','/offline-status.json','/proposal-review.json','/proposal-exam.json') or re.fullmatch(r'/auto-events/(?:reaction|pass)-[0-9a-f]{16}\.json',self.path)):
                path=folder/self.path.lstrip('/')
                if path.is_file():self.respond(path.read_bytes(),'application/json; charset=utf-8')
                else:self.send_error(404)
            else:self.send_error(404)
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                limit=2*1024*1024 if self.path=='/commissioning/import' else 8192
                if not 0<length<=limit:raise ValueError('Invalid request size')
                # Consume the bounded body before rejecting authentication. On
                # Windows, closing with a body still arriving can reset TCP and
                # discard the 403 response. The socket retains its 5 s timeout.
                payload=self.rfile.read(length)
                if not self.valid_host() or self.headers.get('X-FCW-Token')!=app.token:
                    self.respond({'error':'Forbidden'},status=403);return
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                    self.respond({'error':'Forbidden'},status=403);return
                if self.path=='/commissioning/import':
                    from commissioning_store import _json
                    data=_json(payload)
                else:data=json.loads(payload)
                if not isinstance(data,dict):raise ValueError('Expected a JSON object')
                result=app.action(self.path,data);self.respond(result)
            except (ValueError,TypeError,KeyError,OSError) as e:self.respond({'error':str(e)},status=400)
    return Handler

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config');parser.add_argument('--record-root');parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--no-background',action='store_true');parser.add_argument('--self-test-sync',action='store_true')
    parser.add_argument('--self-test-mapping',action='store_true');parser.add_argument('--mapping-test-output')
    parser.add_argument('--duration',type=float);parser.add_argument('--review-session')
    parser.add_argument('--audible-review-session',help='Automatically extract bookmarked road-camera stills for a completed session')
    parser.add_argument('--hindsight-session')
    parser.add_argument('--self-test-imports',action='store_true',help='Claude fork 2026-10-21: import every engine module (packaged build check) and exit')
    parser.add_argument('--self-calibrate-session',help='Analyze a completed ordinary-drive recording without applying settings')
    parser.add_argument('--learning-worker')
    parser.add_argument('--run-exam');parser.add_argument('--exam-candidate');parser.add_argument('--exam-output')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--demo',action='store_true');mode.add_argument('--comma-only',action='store_true')
    parser.add_argument('--self-test-inference-worker',action='store_true')
    parser.add_argument('--self-test-recorded-replay',metavar='CONFIG_JSON')
    parser.add_argument('--self-test-camera-worker',metavar='CONFIG_JSON')
    parser.add_argument('--self-test-worker-owner-exit')
    args=parser.parse_args()
    if getattr(sys,'frozen',False) and ASSETS.resolve()==(BASE/'_internal').resolve():
        verification=start_build_stamp()
        if verification is not None:verification.join(30)
        stamp=build_stamp()
        if stamp.get('state')!='complete' or stamp.get('bundle_state')!='complete':
            raise RuntimeError('FCW bundle integrity failed: '+str(stamp.get('state')))
    if args.self_test_recorded_replay:
        from recorded_replay_selftest import run
        print(json.dumps(run(args.self_test_recorded_replay)),flush=True);return
    if args.self_test_camera_worker:
        from camera_worker_selftest import run
        result=run(args.self_test_camera_worker,ASSETS)
        print(json.dumps(result),flush=True)
        raise SystemExit(0 if result['status']=='passed' else 1)
    if args.self_test_inference_worker or args.self_test_worker_owner_exit:
        from inference_worker_selftest import run
        print(json.dumps(run(args.self_test_worker_owner_exit)),flush=True);return
    if args.self_test_imports:
        # every module the engine may load by name at run time (v26 / v30 shipped without near_edge_motion)
        import importlib
        names=__import__('json').loads(__import__('pathlib').Path(__file__).with_name('engine-modules.json').read_text()) if __import__('pathlib').Path(__file__).with_name('engine-modules.json').exists() else []
        bad=[]
        for n in names:
            try:importlib.import_module(n)
            except Exception as e:bad.append(f'{n}: {type(e).__name__}: {e}')
        print(__import__('json').dumps({'modules':len(names),'failed':bad}));raise SystemExit(1 if bad or not names else 0)
    if args.self_calibrate_session:
        from selfcal import run_session
        folder=Path(args.self_calibrate_session)
        result=run_session(folder,folder.parent.parent,progress=lambda p:print(json.dumps(p),flush=True))
        print(json.dumps(result,indent=2));return
    if args.learning_worker:
        from learning.pipeline import run_worker
        raise SystemExit(run_worker(args.learning_worker))
    if args.self_test_sync:
        from sync_selftest import run
        print(json.dumps(run()));return
    if args.self_test_mapping:
        from mapping.selftest import run
        print(json.dumps(run(args.mapping_test_output)));return
    if args.run_exam:
        from exam import run_exam
        config_path=Path(args.config or BASE/'config.json')
        cfg=json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.exists() else DEFAULT_CONFIG
        baseline=canonical_policy(cfg.get('perception_policy'))
        candidate=json.loads(Path(args.exam_candidate).read_text()) if args.exam_candidate else baseline
        report=run_exam(args.run_exam,args.record_root or Path(os.environ.get('LOCALAPPDATA',BASE))/'OdysseyFCW/recordings',baseline,candidate,lambda s:print(s,flush=True))
        if args.exam_output:write_json(Path(args.exam_output),report)
        print(json.dumps({'status':report['status'],'eligible':report['eligible'],'exam_digest':report['exam_digest']}));return
    if args.hindsight_session:
        if not (Path(args.hindsight_session)/'final-state.json').is_file():raise ValueError('Finish recording before hindsight analysis')
        from evaluation import summarize
        report=summarize(args.hindsight_session)
        print(json.dumps(report['hindsight'],default=jsonable))
        if report['hindsight']['status']!='complete':raise RuntimeError('Hindsight analysis failed; see report')
        return
    if args.audible_review_session:
        from audible_review import build_images
        print(json.dumps(build_images(Path(args.audible_review_session))));return
    if args.review_session:
        from postprocess import run_offline
        print(json.dumps(run_offline(args.review_session,Path(args.review_session).parent.parent,EXAM_ROOT),indent=2));return
    from app_lifecycle import clear_manual_quit
    if not args.no_browser and not args.duration:clear_manual_quit(BASE)
    app=Application(args.config,args.record_root);port=app.config['http_port']
    class Server(ThreadingHTTPServer):
        allow_reuse_address=False
        def server_bind(self):
            if os.name=='nt':self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            super().server_bind()
    try:server=Server(('127.0.0.1',port),handler(app,port))
    except OSError:
        if not args.duration and not args.no_browser:
            import urllib.request
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f'http://127.0.0.1:{port}/health',timeout=1) as response:existing=json.loads(response.read())
                if existing.get('app')=='OdysseyFCW':
                    # A newly unpacked duplicate may still be hashing its own executable.
                    # Give the existing/current identity workers time to finish; pending is not a different build.
                    end=time.monotonic()+10
                    while not same_running_build(existing,launch_identity()) and time.monotonic()<end:
                        current=launch_identity();other=existing.get('identity') or {}
                        if current.get('hash_state')=='complete' and other.get('hash_state')=='complete':break
                        time.sleep(.1)
                        with opener.open(f'http://127.0.0.1:{port}/health',timeout=1) as response:existing=json.loads(response.read())
                    if not same_running_build(existing,launch_identity()):raise RuntimeError('A different FCW build is running, or startup verification has not finished. Open the existing driver window; do not launch more copies.')
                    webbrowser.open(f'http://127.0.0.1:{port}');return
            except (OSError,ValueError):pass
        raise
    server.timeout=.2
    if args.demo:app.start('demo')
    elif args.comma_only:app.start('comma')
    url=f'http://127.0.0.1:{port}'
    print('ODYSSEY FCW — commissioning build\n'+url+'\nCtrl+C saves the recording and quits. '+('Pending reviews resume next launch.' if app.background_mode()=='full' else 'Heavy offline analysis runs on the Macs.'),flush=True)
    if not args.no_browser:webbrowser.open(url)
    if not args.no_background:app.start_background()
    if not (args.demo or args.comma_only or args.duration):   # v19: hands-free recording when config auto_record is set
        from autodrive import AutoDrive
        app.autodrive=AutoDrive(app);app.autodrive.start()
    started=time.monotonic()
    try:
        while not app.shutdown.is_set() and (not args.duration or time.monotonic()-started<args.duration):server.handle_request()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        app.close_background()
        if app.session and not app.session.stopped:
            app.session.stop(analyze=False)
            write_json(app.session.folder/'offline-status.json',{'schema_version':1,'state':'pending','reason':('Saved on app shutdown; automatically resumed on next launch' if app.background_mode()=='full' else 'Saved on app shutdown; heavy analysis is offloaded to the Macs')})


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:main()
    except Exception as e:
        print('Startup failed:',e,flush=True)
        if not any(flag in sys.argv for flag in ('--self-test-recorded-replay','--self-test-inference-worker','--self-test-camera-worker','--self-test-worker-owner-exit','--duration','--review-session','--audible-review-session','--hindsight-session','--self-calibrate-session','--run-exam','--self-test-sync','--self-test-mapping','--learning-worker')):
            if os.name=='nt':
                import ctypes
                ctypes.windll.user32.MessageBoxW(None,str(e),'Odyssey FCW could not start',0x10)
        raise

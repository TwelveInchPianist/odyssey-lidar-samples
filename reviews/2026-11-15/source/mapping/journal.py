"""Compact append-only route evidence, independent of bulk lidar/comma budgets.

Rotated JSONL parts, periodic flush/fsync, explicit discontinuities and validity.
Readers preserve complete records after a crash and report corrupt/truncated
records. They never repair by overwriting original evidence.
"""
from collections import Counter
import json, os, shutil, threading
import runtime_clock as time
from pathlib import Path
from .contracts import canonical_json
from .location import decode_fix,FixContinuity,LOCATION_SERVICES,best_fix,reason_text

MOTION_SERVICES=('carState','livePose','liveLocationKalman','liveCalibration')
ROUTE_SERVICES=set(MOTION_SERVICES+LOCATION_SERVICES)
INTERVAL_NS={'carState':50_000_000,'livePose':50_000_000,'liveLocationKalman':100_000_000,'liveCalibration':500_000_000,'gnssMeasurements':100_000_000}

def journal_parts(folder):return sorted(Path(folder).glob('route-journal-[0-9][0-9][0-9][0-9][0-9].jsonl'))

def read_journal(folder,diagnostics=None,cancel=None):
    diagnostics=Counter() if diagnostics is None else diagnostics
    previous=-1
    for path in journal_parts(folder):
        with path.open('rb') as f:
            for line_number,line in enumerate(f,1):
                if cancel and cancel():
                    from offline_jobs import WorkPaused
                    raise WorkPaused()
                if not line.endswith(b'\n'):
                    diagnostics['truncated_records']+=1;continue
                try:
                    event=json.loads(line)
                    if event.get('schema_version')!=1 or type(event.get('seq')) is not int:raise ValueError()
                except (ValueError,TypeError):diagnostics['invalid_records']+=1;continue
                if event['seq']<=previous:diagnostics['non_increasing_sequence']+=1;continue
                if previous>=0 and event['seq']!=previous+1:diagnostics['missing_sequence_records']+=event['seq']-previous-1
                previous=event['seq'];yield {'part':path.name,'line':line_number},event

class RouteJournal:
    def __init__(self,folder,session_id,minimum_free_bytes=3_000_000_000,rotate_bytes=16_000_000,fsync_interval_s=1.,disk_usage=shutil.disk_usage):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.session_id=session_id;self.minimum_free_bytes=minimum_free_bytes;self.rotate_bytes=rotate_bytes
        self.fsync_interval_s=fsync_interval_s;self.disk_usage=disk_usage;self.lock=threading.RLock()
        self.file=None;self.part=-1;self.seq=0;self.epoch=0;self.last_source={};self.last_kept={};self.continuity=FixContinuity()
        self.last_sync=time.monotonic();self.last_space=-1e9;self.space_ok=False;self.stats=Counter();self.closed=False;self.error=None;self.locations={}
        self.location_status={'state':'waiting','message':'Waiting for a published GPS/location fix; no position is assumed.'};self.hello_receipt_ns=None
        parts=journal_parts(self.folder)
        if parts:
            # Never append behind a potentially torn tail. Start a new part;
            # sequence continuity is recovered by streaming existing records.
            diagnostics=Counter()
            for _,event in read_journal(self.folder,diagnostics):
                if event.get('session_id')!=session_id:raise ValueError('Cannot resume another session in this route journal')
                self.seq=max(self.seq,event['seq']+1);self.epoch=max(self.epoch,event.get('epoch',0))
            self.part=int(parts[-1].stem.rsplit('-',1)[1]);self.epoch+=1
            self.stats.update(diagnostics)
        self._append('journal_open',{'resumed':bool(parts),'session_id':session_id},time.monotonic_ns())

    def _space(self):
        now=time.monotonic()
        if now-self.last_space>=1:
            self.last_space=now
            self.space_ok=self.disk_usage(self.folder).free>self.minimum_free_bytes
        if not self.space_ok:raise OSError('Route journal paused: disk reserve reached')

    def _append(self,kind,data,receipt_ns):
        if self.closed:return False
        try:
            self._space()
            if self.file is None or self.file.tell()>=self.rotate_bytes:
                if self.file:self.file.flush();os.fsync(self.file.fileno());self.file.close()
                self.part+=1
                self.file=(self.folder/f'route-journal-{self.part:05d}.jsonl').open('xb',buffering=64*1024)
            record={'schema_version':1,'session_id':self.session_id,'seq':self.seq,'epoch':self.epoch,
                    'kind':kind,'receipt_ns':receipt_ns,'data':data}
            payload=(canonical_json(record)+'\n').encode()
            self.file.write(payload);self.seq+=1;self.stats['records_written']+=1;self.stats['bytes_written']+=len(payload);self.error=None
            if time.monotonic()-self.last_sync>=self.fsync_interval_s:self._flush()
            return True
        except (OSError,ValueError,TypeError) as e:
            self.error=str(e);self.stats['records_unsaved']+=1
            if self.file:
                try:self.file.close()
                except OSError:pass
                self.file=None
            return False

    def _flush(self):
        if self.file:self.file.flush();os.fsync(self.file.fileno())
        self.last_sync=time.monotonic()
        target=self.folder/'route-journal-status.json';temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.snapshot(),indent=2,allow_nan=False),encoding='utf-8');os.replace(temporary,target)

    def ingest(self,item,sync=None):
        with self.lock:
            if self.closed:return False
            receipt=item.get('laptop_receipt_ns',time.monotonic_ns());kind=item.get('kind')
            if kind=='hello':
                self.epoch+=1;self.last_source.clear();self.last_kept.clear();self.continuity.reset();self.locations.clear()
                self.hello_receipt_ns=receipt
                if not set(item.get('services',[]))&set(LOCATION_SERVICES):self.location_status={'state':'unavailable','message':'No location service advertised. Local lidar/motion mapping can still operate.'}
                else:self.location_status={'state':'waiting','message':'Location service registered; waiting for a usable fix.'}
                return self._append('stream_restart',{'commit':item.get('commit'),'services':item.get('services',[])},receipt)
            if kind=='pong':return self._append('clock_exchange',item,receipt)
            if kind=='route_reference':return self._append('comma_recording_reference',item,receipt)
            service=item.get('service')
            if kind!='sample' or service not in ROUTE_SERVICES:return True
            stamp=item.get('logMonoTime');previous=self.last_source.get(service)
            if type(stamp) is not int or stamp<=0:
                return self._append('invalid_source_time',{'service':service,'source_ns':stamp},receipt)
            if previous is not None and stamp<previous:
                self.epoch+=1;self.last_kept.clear();self.continuity.reset();self.locations.clear()
                self._append('source_clock_discontinuity',{'service':service,'previous_ns':previous,'source_ns':stamp},receipt)
            if previous==stamp:
                self.stats['duplicate_source_messages']+=1;return True
            self.last_source[service]=stamp
            if stamp-self.last_kept.get(service,-10**20)<INTERVAL_NS.get(service,0):
                self.stats['deliberately_downsampled_messages']+=1;return True
            self.last_kept[service]=stamp
            # Retain exact selected source fields, flags and original clocks for
            # reprocessing; the normalized fix is an additional interpretation.
            data={k:item.get(k) for k in ('service','valid','logMonoTime','comma_mono_ns','comma_wall_ns','laptop_receipt_ns','laptop_wall_ns','measurement_mono_ns','measurement_time_bound_s','data')}
            fix=decode_fix(item,sync)
            if fix is not None:
                fix=self.continuity.accept(fix);data['location']=fix
                self.locations[service]=fix
                if fix['usable_for_fusion']:
                    self.location_status={'state':'usable','provider':service,'horizontal_accuracy_m':fix['horizontal_accuracy_m'],
                                          'at_ns':fix['at_ns'],'message':'Location fixes are being recorded with reported accuracy.'}
                elif fix.get('usable_for_trace'):
                    self.location_status={'state':'coarse','provider':service,'horizontal_accuracy_m':fix['horizontal_accuracy_m'],'at_ns':fix['at_ns'],
                                          'message':'GPS route recorded. Accuracy or measurement timing is unverified; position is a search hint, not a precise map constraint.'}
                elif fix['valid']:
                    self.location_status={'state':'clock_unverified','provider':service,'message':'Location is recorded, but clock alignment is not verified.'}
                else:self.location_status={'state':'invalid','provider':service,'message':'Location unavailable: '+reason_text(fix['reasons'])+'. Local mapping can continue.'}
            return self._append('sample',data,receipt)

    def snapshot(self):
        with self.lock:
            location=dict(self.location_status)
            if not self.closed:
                now=time.monotonic_ns();current=best_fix(self.locations.values(),now,allow_trace=True)
                if current and current.get('usable_for_fusion'):
                    location={'state':'usable','provider':current['provider'],'horizontal_accuracy_m':current['horizontal_accuracy_m'],
                              'at_ns':current['at_ns'],'message':'Location fixes are being recorded with reported accuracy.'}
                elif current:location.update(state='coarse',message='Approximate GPS route recorded; location quality does not establish a precise map pose.')
                elif location.get('state') in ('usable','coarse'):location.update(state='stale',message='No recent usable location fix; route segments will not be joined across gaps.')
                elif location.get('state')=='waiting' and self.hello_receipt_ns and now-self.hello_receipt_ns>10_000_000_000:location.update(state='unavailable',message='No location fix received. Local lidar/motion mapping can continue.')
            return {'schema_version':1,'session_id':self.session_id,'state':'closed' if self.closed else 'error' if self.error else 'recording',
                    'stats':dict(self.stats),'last_sequence':self.seq-1,'epoch':self.epoch,'error':self.error,'location':location,
                    'durability':'append-only parts; flush/fsync at most once per second during activity; tail loss remains possible on power failure'}

    def latest_fix(self,at_ns):
        with self.lock:
            fix=best_fix(self.locations.values(),at_ns,allow_hints=True)
            return json.loads(json.dumps(fix)) if fix is not None else None

    def close(self):
        with self.lock:
            if self.closed:return
            self._append('journal_close',{'stats':dict(self.stats)},time.monotonic_ns())
            self.closed=True
            try:self._flush()
            except OSError as e:self.error=str(e)
            if self.file:
                try:self.file.close()
                except OSError:pass
                self.file=None

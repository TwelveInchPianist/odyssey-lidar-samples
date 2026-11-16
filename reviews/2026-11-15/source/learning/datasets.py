"""Immutable OEM-radar measurement datasets, not collision ground truth.

No model fitting, FCW mutation, or vehicle access occurs here. Sources retain
transport provenance. Trip iteration deduplicates measurements using bounded
SQLite staging, preserves fault/epoch boundaries, and rejects disagreements.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import runtime_clock as time
import uuid

from radar import normalize_radar_record, ERROR_FIELDS, MAX_POINTS

VERSION = 'radar-datasets-v1'
RESERVE_BYTES = 10_000_000_000
CLOSEOUT_MARGIN = 16*1024*1024
CHUNK = 1024*1024
MAX_LINE_BYTES = 2*1024*1024
DRIVING_REVIEW_VERSION = 'can-movement-episodes-v1'
PHYSICAL_GROUP_GAP_NS = 1_800_000_000_000  # adjacent restarts remain one episode


class DatasetPaused(Exception): pass
class RadarDataConflict(ValueError): pass


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _identity(value): return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _check(cancel):
    if cancel and cancel(): raise DatasetPaused('Radar dataset work paused; source remains incomplete')


def _sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _moving_intervals(value):
    if not isinstance(value,list) or not 0<len(value)<=10_000:return False
    previous=0
    for pair in value:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair) or not 0<pair[0]<pair[1] or pair[0]<previous:return False
        previous=pair[1]
    return True


def _route(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,159}',value):
        raise ValueError('Invalid source route identity')
    return value


def _regular(path):
    path=Path(path)
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)(): raise ValueError('Dataset files must not be links')
    if not path.is_file(): raise ValueError('Dataset source file missing')
    return path


def _digest(path,cancel=None):
    h=hashlib.sha256()
    with _regular(path).open('rb') as f:
        while chunk:=f.read(CHUNK): _check(cancel); h.update(chunk)
    return h.hexdigest()


def _bounded_lines(path,cancel=None):
    with _regular(path).open('rb') as f:
        while True:
            _check(cancel); line=f.readline(MAX_LINE_BYTES+1)
            if not line: return
            if len(line)>MAX_LINE_BYTES or not line.endswith(b'\n'): raise ValueError('Oversized or truncated dataset record')
            yield line


def _normalized(row):
    """Validate an already-normalized raw record without upgrading bad evidence."""
    if not isinstance(row,dict) or row.get('kind')!='raw_oem_tracks' or row.get('frame_origin')!='front_bumper_flu':
        raise ValueError('Dataset requires normalized raw OEM radar in native bumper axes')
    if type(row.get('valid')) is not bool or type(row.get('available')) is not bool:
        raise ValueError('Normalized radar validity is missing')
    if not isinstance(row.get('points'),list) or len(row['points'])>MAX_POINTS: raise ValueError('Invalid radar point list')
    epoch=row.get('epoch')
    if type(epoch) not in (str,int) or isinstance(epoch,bool) or len(str(epoch))>256: raise ValueError('Radar epoch missing or excessive')
    native=row.get('native_valid',row['valid'])
    result=normalize_radar_record({'kind':'sample','service':'liveTracks','valid':native,
        'logMonoTime':row.get('source_ns'),'laptop_receipt_ns':row.get('receipt_ns'),
        'epoch':epoch,'data':{'errors':row.get('errors'),'points':row['points']}})
    if row['valid'] and not result['valid']: raise ValueError('Normalized source claims validity for invalid native measurement')
    if not row['valid'] or not row['available']:
        reasons=row.get('reasons')
        if not isinstance(reasons,list) or any(not isinstance(x,str) or len(x)>1000 for x in reasons):
            raise ValueError('Invalid radar rejection evidence')
        result.update(valid=False,available=False,points=[],reasons=reasons or ['source marked unavailable'])
    result['epoch']=epoch
    return result


class DatasetStore:
    def __init__(self,data_root):
        self.root=Path(data_root).resolve()/'learning/radar-datasets'
        self.root.mkdir(parents=True,exist_ok=True); self.objects=self.root/'objects'; self.objects.mkdir(exist_ok=True)
        self.scratch=self.root/'working'; self.scratch.mkdir(exist_ok=True); self.db_path=self.root/'registry.sqlite'
        self.check_space()
        with self.connect() as db:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,1,2,3): raise ValueError('Unsupported radar dataset registry version')
            db.executescript('''
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,payload TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,state TEXT NOT NULL,payload TEXT NOT NULL,updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS identity_reviews(sequence INTEGER PRIMARY KEY AUTOINCREMENT,source_id TEXT NOT NULL,review_id TEXT NOT NULL,payload TEXT NOT NULL,created REAL NOT NULL,UNIQUE(source_id,review_id));
CREATE TABLE IF NOT EXISTS driving_reviews(sequence INTEGER PRIMARY KEY AUTOINCREMENT,source_id TEXT NOT NULL,review_id TEXT NOT NULL,payload TEXT NOT NULL,created REAL NOT NULL,UNIQUE(source_id,review_id));
PRAGMA user_version=3;
''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=10)
        db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
        try:
            with db: yield db
        finally: db.close()

    def check_space(self,next_bytes=0):
        if shutil.disk_usage(self.root).free<RESERVE_BYTES+CLOSEOUT_MARGIN+next_bytes:
            raise DatasetPaused('Radar dataset work would consume the recording disk reserve')

    def job(self,job_id,state,payload):
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,payload=excluded.payload,updated=excluded.updated',
                       (job_id,state,_canonical(payload),time.time()))

    def _external(self,payload):
        result=dict(payload); sha=result.get('sha256'); relative=Path(result.pop('object_file'))
        if not _sha(sha) or relative!=Path('objects')/sha[:2]/(sha+'.jsonl'): raise ValueError('Invalid dataset object path')
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.objects.resolve()): raise ValueError('Dataset object escapes store')
        result['path']=str(path); result['registry_root']=str(self.root)
        with self.connect() as db:
            row=db.execute('SELECT review_id,payload FROM identity_reviews WHERE source_id=? ORDER BY sequence DESC LIMIT 1',(result['id'],)).fetchone()
        review={'verified':False,'openpilot_commit':None,'car_fingerprint':None,'radar_unavailable':None,
                'source_hashes':self._source_hashes(result),'issues':['Actual source/car identity has not been verified']}
        if row:
            review=json.loads(row[1])
            if _identity(review)!=row[0]:raise ValueError('Source identity review integrity failed')
        result['provenance']={**result['provenance'],'identity_review':review}
        with self.connect() as db:
            driving=db.execute('SELECT review_id,payload FROM driving_reviews WHERE source_id=? ORDER BY sequence DESC LIMIT 1',(result['id'],)).fetchone()
        driving_review={'verified':False,'physical_drive_group':None,'is_driving':False,
                        'source_hashes':self._source_hashes(result),'issues':['Physical driving episode/movement independence not verified']}
        if driving:
            driving_review=json.loads(driving[1])
            if _identity(driving_review)!=driving[0]:raise ValueError('Driving review integrity failed')
        result['provenance']['driving_review']=driving_review
        identity_ok=bool(result['independent_trip_eligible'] and review.get('verified') is True and
            not review.get('issues') and review.get('git_dirty') is False and review.get('car_fingerprint')=='HONDA_ODYSSEY' and review.get('radar_unavailable') is False and
            re.fullmatch(r'[0-9a-f]{40}',str(review.get('openpilot_commit',''))) and review.get('source_hashes')==self._source_hashes(result))
        result['qualification_eligible']=bool(identity_ok and driving_review.get('verified') is True and driving_review.get('is_driving') is True and
            isinstance(driving_review.get('physical_drive_group'),str) and driving_review['physical_drive_group'] and not driving_review.get('issues') and
            driving_review.get('source_hashes')==self._source_hashes(result) and
            _moving_intervals(driving_review.get('movement',{}).get('moving_intervals_source_ns')))
        return result

    @staticmethod
    def _source_hashes(source):
        p=source['provenance']
        return p.get('native_source_hashes',{}) if p.get('kind')=='native_rlog_import' else {'comma.jsonl':p['input_sha256']} if p.get('kind')=='laptop_comma_jsonl' else {}

    def attach_identity_review(self,source_id,review):
        """Append source-bound compatibility evidence; original source stays immutable."""
        with self.connect() as db:row=db.execute('SELECT payload FROM sources WHERE id=?',(source_id,)).fetchone()
        if row is None:raise ValueError('Unknown radar source for identity review')
        source=json.loads(row[0]);expected=self._source_hashes(source)
        if isinstance(review,dict) and 'review_id' in review and review['review_id']!=_identity({k:v for k,v in review.items() if k!='review_id'}):
            raise ValueError('External identity review hash mismatch')
        if isinstance(review,dict) and 'git_dirty' not in review:
            # Older bounded-prefix reviews explicitly retained each native
            # initData.dirty value. Promote only that direct evidence, never the
            # prior "verified" label alone, and retain the original review ID.
            review=json.loads(_canonical(review));evidence=review.get('prefix_evidence',[])
            complete=(isinstance(evidence,list) and evidence and
                      {str(e.get('segment')) for e in evidence if isinstance(e,dict)}==set(expected))
            clean=bool(complete and all(isinstance(e,dict) and isinstance(e.get('initData'),dict) and e['initData'].get('dirty') is False for e in evidence))
            review['git_dirty']=False if clean else None
            if not clean:
                review['verified']=False;review['issues']=list(review.get('issues',[]))+['Clean source tree not established by original evidence']
            original=review.pop('review_id',None)
            if original:review['original_review_id']=original
            review['review_id']=_identity(review)
        if (not isinstance(review,dict) or type(review.get('verified')) is not bool or review.get('source_hashes')!=expected or
            not isinstance(review.get('issues'),list) or any(not isinstance(x,str) for x in review['issues'])):
            raise ValueError('Identity review is missing or bound to different source bytes')
        if review.get('source_route_id',source['source_route_id'])!=source['source_route_id']:
            raise ValueError('Identity review route mismatch')
        if review['verified'] and (review['issues'] or not re.fullmatch(r'[0-9a-f]{40}',str(review.get('openpilot_commit',''))) or
            review.get('car_fingerprint')!='HONDA_ODYSSEY' or review.get('radar_unavailable') is not False or review.get('git_dirty') is not False):
            raise ValueError('Verified identity review has an unsupported/incomplete actual profile')
        # The table hash covers the complete supplied review, including any
        # original external review_id, without editing its provenance.
        key=_identity(review)
        with self.connect() as db:db.execute('INSERT OR IGNORE INTO identity_reviews(source_id,review_id,payload,created) VALUES(?,?,?,?)',
                                             (source_id,key,_canonical(review),time.time()))
        return self._external(source)

    def attach_driving_review(self,source_id,review):
        """Attach observed movement/physical-episode grouping, not route-count proof."""
        with self.connect() as db:row=db.execute('SELECT payload FROM sources WHERE id=?',(source_id,)).fetchone()
        if row is None:raise ValueError('Unknown radar source for driving review')
        source=json.loads(row[0]);group=review.get('physical_drive_group') if isinstance(review,dict) else None
        if (not isinstance(review,dict) or type(review.get('verified')) is not bool or type(review.get('is_driving')) is not bool or
            review.get('source_hashes')!=self._source_hashes(source) or not isinstance(review.get('issues'),list) or
            any(not isinstance(x,str) for x in review['issues']) or
            group is not None and (not isinstance(group,str) or not 0<len(group)<=256)):
            raise ValueError('Driving review missing or bound to different source bytes')
        if review['verified'] and (review['issues'] or review['is_driving'] and group is None):
            raise ValueError('Verified driving review lacks a supported physical group')
        if review['verified'] and review['is_driving'] and not _moving_intervals(review.get('movement',{}).get('moving_intervals_source_ns')):
            raise ValueError('Verified driving source requires measured moving source-time intervals')
        key=_identity(review)
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO driving_reviews(source_id,review_id,payload,created) VALUES(?,?,?,?)',
                       (source_id,key,_canonical(review),time.time()))
            # New admission and every resulting group merge publish together;
            # a concurrent reader cannot count provisional separate groups.
            self._regroup_driving_reviews(db)
        return self._external(source)

    def _regroup_driving_reviews(self,db):
        """Conservatively group overlapping/adjacent captures, never count routes.

        A parked capture may bridge adjacent restarts but never becomes an
        eligible driving source. Revisions are append-only and metadata-only.
        Sources without verified UTC bounds cannot establish new independence.
        """
        rows=db.execute('SELECT s.id,s.payload,d.payload FROM sources s JOIN driving_reviews d ON d.source_id=s.id WHERE d.sequence=(SELECT MAX(x.sequence) FROM driving_reviews x WHERE x.source_id=s.id)').fetchall()
        entries=[]
        for sid,source_text,review_text in rows:
            source=json.loads(source_text);review=json.loads(review_text);m=review.get('movement',{})
            start,end=m.get('start_wall_ns'),m.get('end_wall_ns')
            if review.get('verified') is True and type(start) is int and type(end) is int and 0<start<=end:
                entries.append({'id':sid,'source':source,'review':review,'start':start,'end':end})
        entries.sort(key=lambda e:(e['start'],e['source']['source_route_id']));components=[]
        for entry in entries:
            # Ten seconds is a conservative grouping tolerance only; it is not
            # a statement of verified sensor measurement timing.
            if not components or entry['start']>max(e['end'] for e in components[-1])+PHYSICAL_GROUP_GAP_NS+10_000_000_000:
                components.append([entry])
            else:components[-1].append(entry)
        # Duplicated capture chunks of one source route always remain one group,
        # even when a long recording outage lies between its chunks.
        changed=True
        while changed:
            changed=False
            for i in range(len(components)):
                left={e['source']['source_route_id'] for e in components[i]}
                match=next((j for j in range(i+1,len(components)) if left & {e['source']['source_route_id'] for e in components[j]}),None)
                if match is not None:
                    components[i].extend(components.pop(match));changed=True;break
        for component in components:
            ordered=sorted(component,key=lambda e:(e['start'],e['source']['source_route_id']))
            moving=[e for e in ordered if e['review']['is_driving']]
            anchor=(moving or ordered)[0]['source']['source_route_id']
            group=('physical-drive:' if moving else 'parked-episode:')+anchor
            routes=sorted({e['source']['source_route_id'] for e in component})
            for entry in component:
                updated={**entry['review'],'physical_drive_group':group,
                         'grouping_version':DRIVING_REVIEW_VERSION,'group_member_routes':routes,
                         'grouping_basis':'Verified capture UTC intervals within 30 minutes plus 10-second grouping tolerance, or same route, are one conservative episode; parked routes never count as driving holdouts.'}
                if updated==entry['review']:continue
                db.execute('INSERT OR IGNORE INTO driving_reviews(source_id,review_id,payload,created) VALUES(?,?,?,?)',
                           (entry['id'],_identity(updated),_canonical(updated),time.time()))

    def sources(self):
        with self.connect() as db: rows=db.execute('SELECT id,payload FROM sources ORDER BY created,id').fetchall()
        output=[]
        for key,encoded in rows:
            value=json.loads(encoded)
            if value.get('id')!=key or key!=_identity({k:v for k,v in value.items() if k!='id'}): raise ValueError('Dataset registry manifest integrity failed')
            output.append(self._external(value))
        return output

    def _commit(self,writer,trip,route,provenance,eligible,cancel=None):
        writer.close(); _check(cancel); self.check_space()
        if not writer.records: raise ValueError('Cannot register an empty valid radar source')
        sha=writer.hash.hexdigest(); relative=Path('objects')/sha[:2]/(sha+'.jsonl'); target=self.root/relative
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if _digest(target,cancel)!=sha: raise ValueError('Immutable radar object corrupted')
            writer.path.unlink()
        else: os.replace(writer.path,target)
        value={'schema_version':1,'version':VERSION,'object_file':str(relative),'sha256':sha,'bytes':target.stat().st_size,
               'physical_trip_id':trip,'source_route_id':route,'source_start_ns':writer.first,'source_end_ns':writer.last,
               'records':writer.records,'valid_records':writer.valid,'invalid_records':writer.records-writer.valid,
               # This flag qualifies trip identity, not observation quality.
               # Fault-only known sources must participate in conflict checks.
               'provenance':provenance,'independent_trip_eligible':bool(eligible),
               'evidence_scope':'OEM radar measurement forecasts only; not collision truth','fusion_eligible':False}
        value['id']=_identity(value); encoded=_canonical(value)
        with self.connect() as db:
            old=db.execute('SELECT payload FROM sources WHERE id=?',(value['id'],)).fetchone()
            if old and old[0]!=encoded: raise ValueError('Immutable source manifest conflict')
            db.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?)',(value['id'],encoded,time.time()))
        return self._external(value)


class _Writer:
    def __init__(self,store):
        self.store=store; fd,name=tempfile.mkstemp(prefix='radar-',suffix='.partial',dir=store.scratch)
        self.path=Path(name); self.file=os.fdopen(fd,'wb'); self.hash=hashlib.sha256(); self.records=0; self.valid=0; self.first=None; self.last=None

    def add(self,row):
        line=(_canonical(row)+'\n').encode(); self.store.check_space(len(line)); self.file.write(line); self.hash.update(line)
        self.records+=1; self.valid+=row['valid'] is True; stamp=row.get('source_ns')
        if type(stamp) is int and stamp>0:
            self.first=stamp if self.first is None else min(self.first,stamp); self.last=stamp if self.last is None else max(self.last,stamp)

    def close(self):
        if not self.file.closed: self.file.flush(); os.fsync(self.file.fileno()); self.file.close()

    def cleanup(self):
        self.close()
        if self.path.exists(): self.path.unlink()


class _MovementEvidence:
    """Bounded CAN movement summaries for automatic post-drive admission."""
    def __init__(self):
        self.count=0;self.valid_count=0;self.first=None;self.end=None;self.start_wall=None;self.end_wall=None
        self.last=None;self.valid_s=0.;self.distance=0.;self.moving_s=0.;self.stationary_s=0.;self.run=0.;self.longest=0.;self.maximum=0.
        self.offset_min=None;self.offset_max=None;self.intervals=[];self.regressions=0;self.interval_limit=False

    def observe(self,item,epoch):
        self.count+=1;d=item.get('data');t=item.get('logMonoTime')
        if not isinstance(d,dict) or type(t) is not int or not 0<t<2**63:
            self.last=None;self.run=0.;return
        speed=d.get('vEgo');gear=d.get('gearShifter')
        try:finite=type(speed) in (int,float) and math.isfinite(speed) and -.05<=speed<70
        except OverflowError:finite=False
        good=bool(item.get('valid') is True and d.get('canValid') is True and d.get('canTimeout') is False and finite)
        wall=None;remote=item.get('comma_mono_ns');utc=item.get('comma_wall_ns')
        if type(remote) is int and type(utc) is int and utc>1_500_000_000_000_000_000 and abs(t-remote)<=5_000_000_000:
            wall=utc+t-remote
        elif type(item.get('laptop_wall_ns')) is int and item['laptop_wall_ns']>1_500_000_000_000_000_000:
            wall=item['laptop_wall_ns']
        self.first=t if self.first is None else min(self.first,t);self.end=t if self.end is None else max(self.end,t)
        if wall is not None:
            self.start_wall=wall if self.start_wall is None else min(self.start_wall,wall);self.end_wall=wall if self.end_wall is None else max(self.end_wall,wall)
            offset=wall-t;self.offset_min=offset if self.offset_min is None else min(self.offset_min,offset);self.offset_max=offset if self.offset_max is None else max(self.offset_max,offset)
        if not good:self.last=None;self.run=0.;return
        speed=max(0.,float(speed));self.valid_count+=1;self.maximum=max(self.maximum,speed)
        current=(t,speed,gear,epoch)
        if self.last is not None:
            old=self.last;dt=(t-old[0])/1e9
            if dt<=0:self.regressions+=1
            if old[3]==epoch and 0<dt<=.5:
                self.valid_s+=dt
                if old[1]<.1 and speed<.1:self.stationary_s+=dt
                if old[2] in ('drive','sport','low','eco') and gear in ('drive','sport','low','eco'):
                    self.distance+=(old[1]+speed)/2*dt
                    if min(old[1],speed)>=2:
                        self.moving_s+=dt;self.run+=dt;self.longest=max(self.longest,self.run)
                        if self.intervals and self.intervals[-1][1]==old[0]:self.intervals[-1][1]=t
                        elif len(self.intervals)<10_000:self.intervals.append([old[0],t])
                        else:self.interval_limit=True
                    else:self.run=0.
                else:self.run=0.
            else:self.run=0.
        self.last=current

    def review(self,route,hashes,known):
        duration=(self.end-self.first)/1e9 if self.first is not None and self.end is not None else 0.
        coverage=min(1.,self.valid_s/duration) if duration>0 else 0.;issues=[]
        if not known:issues.append('Route identity unavailable')
        if duration<30 or coverage<.8:issues.append('Insufficient valid CAN duration/coverage for driving admission')
        if self.start_wall is None or self.end_wall is None:issues.append('Capture UTC bounds unavailable')
        spread=(self.offset_max-self.offset_min)/1e9 if self.offset_min is not None else None
        if spread is None or spread>5:issues.append('Source-to-wall timing discontinuity prevents physical-episode grouping')
        if self.regressions:issues.append('Source time regressed during movement review')
        if self.interval_limit:issues.append('Movement interval budget exceeded')
        driving=bool(self.moving_s>=30 and self.distance>=250 and self.longest>=10)
        movement={'samples':self.count,'valid_samples':self.valid_count,'duration_s':duration,'valid_speed_interval_s':self.valid_s,
                  'valid_interval_coverage':coverage,'start_wall_ns':self.start_wall,'end_wall_ns':self.end_wall,
                  'wall_clock_offset_spread_s':spread,'forward_above2_mps_s':self.moving_s,
                  'longest_continuous_forward_above2_mps_s':self.longest,'valid_forward_distance_proxy_m':self.distance,
                  'max_sampled_speed_mps':self.maximum,'stationary_below_point1_mps_s':self.stationary_s,
                  'moving_intervals_source_ns':self.intervals}
        return {'verified':not issues,'physical_drive_group':('physical-drive:' if driving else 'parked-episode:')+route if not issues else None,
                'is_driving':driving,'source_hashes':hashes,'source_route_id':route,'issues':issues,'movement':movement,
                'review_version':DRIVING_REVIEW_VERSION,'grouping_basis':'Conservative temporal grouping is assigned against the dataset catalog.',
                'admission_rule':'Valid CAN coverage>=80% over>=30s; driving requires>=30s forward at>=2m/s,>=250m valid speed-integrated travel and>=10s continuous motion. Only moving-window anchors support independent-drive qualification.',
                'limitations':'CAN distance is a proxy. UTC is for episode grouping only, not sensor synchronization. Parked logs and route restart count cannot establish driving independence.'}


def _native_manifest(folder):
    folder=Path(folder).resolve(); manifest_path=_regular(folder/'manifest.json')
    if manifest_path.stat().st_size>4*1024*1024: raise ValueError('Native manifest too large')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    route=_route(manifest.get('source_route_id'))
    if manifest.get('physical_trip_id')!='comma:'+route or manifest.get('complete') is not True or manifest.get('state')!='complete':
        raise ValueError('Native route import is not complete or has conflicting trip identity')
    aggregate=manifest.get('aggregate',{}); stage=manifest.get('stage',{})
    if (aggregate.get('file')!='radar.jsonl' or not _sha(aggregate.get('sha256')) or
        type(aggregate.get('bytes')) is not int or aggregate['bytes']<=0): raise ValueError('Invalid native aggregate identity')
    if (stage.get('version') not in ('native-radar-import-v1','native-radar-import-v2','native-radar-import-v3') or
        not _sha(stage.get('normalizer_sha256')) or not _sha(stage.get('importer_sha256')) or
        not isinstance(stage.get('schema_sha256'),dict) or not stage['schema_sha256'] or
        any(not _sha(x) for x in stage['schema_sha256'].values()) or
        any(v is not None and not re.fullmatch(r'[0-9a-f]{40}',str(v)) for v in (stage.get('openpilot_commit'),stage.get('opendbc_commit')))):
        raise ValueError('Native schema/code provenance missing')
    if stage['version']!='native-radar-import-v3' and any(stage.get(k) is None for k in ('openpilot_commit','opendbc_commit')):
        raise ValueError('Legacy native decoder provenance missing')
    segments=manifest.get('segments'); expected=manifest.get('expected_present_segments')
    if (not isinstance(segments,dict) or not segments or not isinstance(expected,list) or
        any(type(n) is not int or n<0 for n in expected) or len(expected)!=len(set(expected)) or
        set(segments)!={str(n) for n in expected}): raise ValueError('Native segment completeness invalid')
    for key,value in segments.items():
        if (not isinstance(value,dict) or value.get('segment')!=int(key) or value.get('state')!='complete' or
            value.get('stage')!=stage or not _sha(value.get('source_sha256')) or not _sha(value.get('output_sha256'))):
            raise ValueError('Native segment provenance invalid')
    return manifest,route,folder/'radar.jsonl',hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def register_native_import(import_folder,data_root,cancel=None):
    manifest,route,path,manifest_sha=_native_manifest(import_folder); store=DatasetStore(data_root)
    job_id=_identity({'stage':VERSION,'native_aggregate':manifest['aggregate']['sha256'],'route':route})
    progress={'kind':'native_import','input':str(path),'bytes_processed':0,'total_bytes':path.stat().st_size}
    store.job(job_id,'running',progress); writer=_Writer(store); actual=hashlib.sha256(); fused=0
    try:
        for line in _bounded_lines(path,cancel):
            actual.update(line); progress['bytes_processed']+=len(line); row=json.loads(line)
            if row.get('source_route_id')!=route or row.get('physical_trip_id')!='comma:'+route:
                raise ValueError('Native row trip identity conflicts with manifest')
            if row.get('kind')=='fused_leads': fused+=1; continue
            record=_normalized(row)
            record.update(physical_trip_id='comma:'+route,source_route_id=route,source_format='native_rlog',
                          source_ref=row.get('source_ref'),source_epoch_kind=row.get('source_epoch_kind'))
            writer.add(record)
            if writer.records%1000==0: store.job(job_id,'running',progress)
        if actual.hexdigest()!=manifest['aggregate']['sha256'] or progress['bytes_processed']!=manifest['aggregate']['bytes']:
            raise ValueError('Native aggregate content hash or byte length mismatch')
        provenance={'kind':'native_rlog_import','dataset_stage':VERSION,'aggregate_sha256':actual.hexdigest(),
                    'manifest_sha256':manifest_sha,'native_stage':manifest['stage'],
                    'native_source_hashes':{k:v['source_sha256'] for k,v in manifest['segments'].items()},
                    'fused_rows_excluded':fused,'metadata_note':'v1 fused publisher PID was upstream card; fused rows are excluded here.'}
        result=store._commit(writer,'comma:'+route,route,provenance,True,cancel)
        identity_file=Path(import_folder)/'identity-review.json'
        if identity_file.is_file():
            if identity_file.stat().st_size>4*1024*1024:raise ValueError('Native identity review too large')
            identity_review=json.loads(identity_file.read_text(encoding='utf-8'))
            if manifest['stage'].get('version')=='native-radar-import-v3' and manifest['stage'].get('identity_review_id')!=identity_review.get('review_id'):
                raise ValueError('Native import stage identity review mismatch')
            result=store.attach_identity_review(result['id'],identity_review)
        driving_file=Path(import_folder)/'driving-review.json'
        if driving_file.is_file():
            if driving_file.stat().st_size>4*1024*1024:raise ValueError('Native driving review too large')
            result=store.attach_driving_review(result['id'],json.loads(driving_file.read_text(encoding='utf-8')))
        store.job(job_id,'complete',{**progress,'source_id':result['id'],'records':result['records']}); return result
    except (DatasetPaused,KeyboardInterrupt):
        store.job(job_id,'paused',{**progress,'restart_policy':'Atomic source import restarts from source beginning; no incomplete source registered'}); raise
    except Exception as e:
        store.job(job_id,'failed',{**progress,'error':type(e).__name__+': '+str(e)}); raise
    finally: writer.cleanup()


def extract_session_radar(folder,data_root,cancel=None,force_scan=False):
    folder=Path(folder).resolve(); store=DatasetStore(data_root); source=folder/'comma.jsonl'
    cfg_path=_regular(folder/'session.json'); config=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    session=config.get('session_id')
    if not isinstance(session,str) or not session: session='legacy:'+hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    session_token=hashlib.sha256(session.encode()).hexdigest()
    if not source.is_file(): return {'status':'unavailable','sources':[],'reason':'No original Comma stream','evidence_complete':False}
    source=_regular(source); stat=source.stat()
    job_id=_identity({'stage':VERSION,'session':session,'input_size':stat.st_size,'input_mtime_ns':stat.st_mtime_ns})
    progress={'kind':'session_extract','session_id':session,'input':str(source),'bytes_processed':0,'total_bytes':stat.st_size}
    device=folder/'comma-device.json'
    if not force_scan and device.is_file() and device.stat().st_size<1024*1024:
        header=json.loads(device.read_text(encoding='utf-8-sig')); advertised=header.get('services')
        if (isinstance(advertised,list) and all(isinstance(x,str) for x in advertised) and 'liveTracks' not in advertised and
            'radar_contract_version' not in header and 'radar_services_registered' not in header):
            result={'status':'unavailable_metadata_only','sources':[],'reason':'Legacy bridge did not advertise raw radar; original stream was not scanned',
                    'evidence_complete':False,'advertisement_sha256':hashlib.sha256(device.read_bytes()).hexdigest(),
                    'raw_log_stat_only':{'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},'independent_trip_eligible':False}
            store.job(job_id,'unavailable_metadata_only',result); return result
    store.job(job_id,'running',progress); writers={}; actual=hashlib.sha256(); route=None; epoch=0; samples=0; unknown=0; fused=0
    identity_epochs={0:{'commit':None,'git_dirty':None,'params':[]}};group_epochs={};movement_evidence={}
    def writer_for(key):
        if key not in writers:
            if len(writers)>=64: raise ValueError('Too many route groups in one recording')
            writers[key]=_Writer(store)
        return writers[key]
    try:
        for line in _bounded_lines(source,cancel):
            actual.update(line); progress['bytes_processed']+=len(line); item=json.loads(line)
            if not isinstance(item,dict): raise ValueError('Malformed original Comma record')
            kind=item.get('kind')
            if kind=='hello':
                route=None;epoch+=1;identity_epochs[epoch]={'commit':item.get('commit'),'git_dirty':item.get('git_dirty'),'params':[]};continue
            if kind=='route_reference':
                new=_route(item.get('route'))
                if new!=route:
                    previous=identity_epochs[epoch];epoch+=1;route=new
                    identity_epochs[epoch]={'commit':previous['commit'],'git_dirty':previous['git_dirty'],'params':[]}
                continue
            if kind=='sample' and item.get('service')=='carParams':
                d=item.get('data')
                if item.get('valid') is True and isinstance(d,dict):
                    identity_epochs[epoch]['params'].append({'car_fingerprint':d.get('carFingerprint'),'radar_unavailable':d.get('radarUnavailable')})
                continue
            if kind=='sample' and item.get('service')=='carState':
                group=route or 'unattributed:'+session_token
                movement_evidence.setdefault(group,_MovementEvidence()).observe(item,epoch)
                continue
            if item.get('service')=='radarState': fused+=1; continue
            if item.get('service')!='liveTracks': continue
            record=normalize_radar_record(item); record['epoch']=f'{session_token}:{epoch}'
            if route is None: unknown+=1
            group=route or 'unattributed:'+session_token; trip='comma:'+route if route else 'unknown:'+session_token
            group_epochs.setdefault(group,set()).add(epoch)
            record.update(physical_trip_id=trip,source_route_id=group,source_format='laptop_comma_jsonl',
                          source_epoch_kind='laptop stream/route boundary')
            writer_for(group).add(record); samples+=1
            if samples%1000==0: store.job(job_id,'running',progress)
        final=source.stat()
        if (final.st_size,final.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('Original Comma source changed during extraction')
        manifests=[]
        for group,writer in writers.items():
            known=not group.startswith('unattributed:'); trip='comma:'+group if known else 'unknown:'+session_token
            provenance={'kind':'laptop_comma_jsonl','dataset_stage':VERSION,'input_sha256':actual.hexdigest(),
                        'session_id':session,'source_file':'comma.jsonl','fused_rows_excluded':fused,
                        'route_identity_basis':'observed route_reference before record' if known else 'unknown; no retrospective attribution'}
            registered=store._commit(writer,trip,group,provenance,known,cancel)
            used=[identity_epochs[e] for e in sorted(group_epochs[group])];commits={e['commit'] for e in used if isinstance(e['commit'],str)}
            params=[p for e in used for p in e['params']];fingerprints={p['car_fingerprint'] for p in params if isinstance(p['car_fingerprint'],str)}
            unavailable={p['radar_unavailable'] for p in params if type(p['radar_unavailable']) is bool}
            commit=next(iter(commits)) if len(commits)==1 else None;fingerprint=next(iter(fingerprints)) if len(fingerprints)==1 else None
            radar_unavailable=next(iter(unavailable)) if len(unavailable)==1 else None;issues=[]
            if not known:issues.append('Route identity unavailable')
            if any(not isinstance(e['commit'],str) or not re.fullmatch(r'[0-9a-f]{40}',e['commit']) for e in used) or commit is None:issues.append('Actual hello commit missing or mixed')
            if any(e['git_dirty'] is not False for e in used):issues.append('Clean source tree not confirmed for every used epoch')
            if any(not e['params'] for e in used):issues.append('Valid current CarParams missing from a used source epoch')
            if fingerprint!='HONDA_ODYSSEY' or any(p['car_fingerprint']!=fingerprint for p in params):issues.append('Actual car fingerprint unsupported or mixed')
            if radar_unavailable is not False or any(p['radar_unavailable'] is not False for p in params):issues.append('Raw factory radar availability unconfirmed or mixed')
            review={'verified':not issues,'openpilot_commit':commit,'car_fingerprint':fingerprint,'radar_unavailable':radar_unavailable,
                    'source_hashes':{'comma.jsonl':actual.hexdigest()},'source_route_id':group,'issues':issues,
                    'git_dirty':False if all(e['git_dirty'] is False for e in used) else None,
                    'basis':'Captured hello commit and valid current CarParams in each used stream/route epoch; no VIN fields retained'}
            registered=store.attach_identity_review(registered['id'],review)
            driving=movement_evidence.get(group,_MovementEvidence()).review(group,{'comma.jsonl':actual.hexdigest()},known)
            manifests.append(store.attach_driving_review(registered['id'],driving))
        # A later source may conservatively merge physical episodes; return the
        # latest metadata for every source rather than stale pre-merge views.
        latest={s['id']:s for s in store.sources()};manifests=[latest[s['id']] for s in manifests]
        result={'status':'complete' if samples else 'unavailable','sources':manifests,'records':samples,
                'unknown_route_records':unknown,'fused_records_excluded':fused,'input_sha256':actual.hexdigest(),
                'evidence_complete':True,'reason':None if samples else 'No raw radar records in inspected stream'}
        store.job(job_id,result['status'],{**progress,'source_ids':[s['id'] for s in manifests],'records':samples}); return result
    except (DatasetPaused,KeyboardInterrupt):
        store.job(job_id,'paused',{**progress,'restart_policy':'Restart atomic source extraction; no incomplete writer is registered'}); raise
    except Exception as e:
        store.job(job_id,'failed',{**progress,'error':type(e).__name__+': '+str(e)}); raise
    finally:
        for writer in writers.values(): writer.cleanup()


def _measurement(record):
    return {'native_valid':record.get('native_valid'),'valid':record['valid'],'available':record['available'],
            'errors':record['errors'],
            'points':sorted([{k:p.get(k) for k in ('trackId','dRel','yRel','vRel','aRel','yvRel','measured')} for p in record['points']],key=lambda p:p['trackId'])}


def iter_trip_records(sources,cancel=None):
    """Yield one source-time record per trip stamp, or reject any disagreement.

    Input sources must belong to one physical trip. No payload is yielded until
    all hashes and conflicting duplicates are checked. Epoch boundaries from any
    source are conservative resets; receipt times remain in original sources and
    are not arbitrarily chosen for the canonical source-time training stream.
    """
    sources=list(sources)
    if not sources: return
    if any(s.get('independent_trip_eligible') is not True for s in sources):
        raise ValueError('Unattributed or invalid source is not eligible for an independent-trip corpus')
    trips={s.get('physical_trip_id') for s in sources}; routes={s.get('source_route_id') for s in sources}
    if len(trips)!=1 or len(routes)!=1 or None in trips or None in routes: raise ValueError('Iterate one physical trip/source route at a time')
    trip=next(iter(trips)); route=next(iter(routes)); scratch=Path(sources[0]['registry_root'])/'working'; scratch.mkdir(exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='radar-merge-',suffix='.sqlite',dir=scratch); os.close(fd); path=Path(name)
    db=sqlite3.connect(path)
    try:
        db.execute('PRAGMA cache_size=-2048'); db.execute('PRAGMA temp_store=FILE'); db.execute('PRAGMA journal_mode=OFF')
        db.executescript('CREATE TABLE records(stamp INTEGER PRIMARY KEY,digest TEXT NOT NULL,payload TEXT NOT NULL,source_id TEXT NOT NULL);CREATE TABLE boundaries(stamp INTEGER PRIMARY KEY,reason TEXT NOT NULL);')
        for manifest in sources:
            if not _sha(manifest.get('id')) or not _sha(manifest.get('sha256')): raise ValueError('Invalid registered source identity')
            actual=hashlib.sha256(); count=0; last_stamp=None; last_epoch=None
            for line in _bounded_lines(manifest['path'],cancel):
                actual.update(line); count+=1; row=json.loads(line)
                if row.get('physical_trip_id')!=trip or row.get('source_route_id')!=route: raise ValueError('Stored row route/trip disagrees with registry')
                record=_normalized(row); stamp=record['source_ns']; epoch=str(record['epoch'])
                if type(stamp) is not int or stamp<=0:
                    db.execute('INSERT OR IGNORE INTO boundaries VALUES(?,?)',((last_stamp+1) if last_stamp else 0,'unknown source timestamp'))
                    last_epoch=None; continue
                if last_stamp is not None and stamp<=last_stamp and stamp!=last_stamp:
                    raise ValueError('Source clock regressed within immutable dataset; cannot sort across an unknown restart')
                if epoch!=last_epoch:
                    db.execute('INSERT OR IGNORE INTO boundaries VALUES(?,?)',(stamp,'source epoch boundary'))
                measurement=_measurement(record); payload=_canonical(measurement); sha=hashlib.sha256(payload.encode()).hexdigest()
                previous=db.execute('SELECT digest,source_id FROM records WHERE stamp=?',(stamp,)).fetchone()
                if previous and previous[0]!=sha:
                    raise RadarDataConflict(f'Conflicting radar payload for trip {trip} at {stamp}: {previous[1]} vs {manifest["id"]}')
                db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)',(stamp,sha,payload,manifest['id']))
                last_stamp=stamp; last_epoch=epoch
                if count%1000==0:
                    _check(cancel)
                    if shutil.disk_usage(scratch).free<RESERVE_BYTES+CLOSEOUT_MARGIN: raise DatasetPaused('Radar merge disk reserve reached')
                    db.commit()
            if actual.hexdigest()!=manifest['sha256'] or count!=manifest['records']: raise ValueError('Registered source content integrity failed')
        db.commit(); _check(cancel)
        boundary=iter(db.execute('SELECT stamp,reason FROM boundaries ORDER BY stamp')); pending=next(boundary,None); epoch=0
        for stamp,_,payload,_ in db.execute('SELECT stamp,digest,payload,source_id FROM records ORDER BY stamp'):
            _check(cancel)
            while pending is not None and pending[0]<=stamp:
                epoch+=1; pending=next(boundary,None)
            record=json.loads(payload)
            reasons=[] if record['valid'] and record['available'] else ['native radar fault or unavailable measurement preserved from registered source']
            yield {**record,'schema_version':1,'service':'liveTracks','kind':'raw_oem_tracks',
                   'source_ns':stamp,'receipt_ns':None,'frame_origin':'front_bumper_flu',
                   'epoch':f'{trip}:canonical:{epoch}','physical_trip_id':trip,'source_route_id':route,
                   'reasons':reasons,'diagnostic_only':True,'fusion_eligible':False,'timing_verified':False,
                   'measurement_time_basis':'source_publication_only; original receipts retained in registered sources'}
    finally:
        db.close()
        if path.exists(): path.unlink()

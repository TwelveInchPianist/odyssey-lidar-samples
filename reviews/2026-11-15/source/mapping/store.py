"""Persistent immutable evidence with separately published map/pose revisions.

SQLite metadata and content-addressed local-frame clouds. No recent-frame limit;
geographic indexing is a coarse candidate filter, never a localization verdict.
Original sessions and old map databases are never rewritten by this store.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib, json, math, os, sqlite3, uuid
import runtime_clock as time
from pathlib import Path
import numpy as np
from .contracts import Keyframe,Constraint,canonical_json,identity,checked_pose
from .geodesy import horizontal_distance

SCHEMA_VERSION=2

def cloud_digest(points):
    p=np.ascontiguousarray(points,dtype='<f4')
    if p.ndim!=2 or p.shape[1]!=3 or not 1<=len(p)<=200_000 or not np.isfinite(p).all():raise ValueError('A finite Nx3 local-frame cloud is required')
    return hashlib.sha256(b'odyssey-local-cloud-v1\0'+str(p.shape).encode()+p.tobytes()).hexdigest(),p

class AtlasStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.cloud_root=self.root/'observations';self.cloud_root.mkdir(exist_ok=True)
        self.path=self.root/'atlas.sqlite'
        with self.connect() as db:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,SCHEMA_VERSION):raise ValueError('Unsupported atlas schema; keep original and migrate explicitly')
            db.executescript('''
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS keyframes(rowid INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,session TEXT NOT NULL,segment TEXT NOT NULL,stamp INTEGER NOT NULL,calibration TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,FOREIGN KEY(session) REFERENCES sessions(id));
CREATE INDEX IF NOT EXISTS keyframe_session_time ON keyframes(session,segment,stamp);
CREATE TABLE IF NOT EXISTS descriptors(keyframe TEXT NOT NULL,version TEXT NOT NULL,vector BLOB NOT NULL,dimensions INT NOT NULL,PRIMARY KEY(keyframe,version),FOREIGN KEY(keyframe) REFERENCES keyframes(id));
CREATE TABLE IF NOT EXISTS locations(keyframe TEXT PRIMARY KEY,latitude REAL NOT NULL,longitude REAL NOT NULL,accuracy REAL NOT NULL,FOREIGN KEY(keyframe) REFERENCES keyframes(id));
CREATE INDEX IF NOT EXISTS location_lat_lon ON locations(latitude,longitude);
CREATE VIRTUAL TABLE IF NOT EXISTS geo_index USING rtree(rowid,min_lon,max_lon,min_lat,max_lat);
CREATE TABLE IF NOT EXISTS constraints(id TEXT PRIMARY KEY,source TEXT NOT NULL,target TEXT NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,FOREIGN KEY(source) REFERENCES keyframes(id),FOREIGN KEY(target) REFERENCES keyframes(id));
CREATE INDEX IF NOT EXISTS constraint_endpoints ON constraints(source,target);
CREATE TABLE IF NOT EXISTS decisions(sequence INTEGER PRIMARY KEY AUTOINCREMENT,constraint_id TEXT NOT NULL,state TEXT NOT NULL,reason TEXT NOT NULL,diagnostics TEXT NOT NULL,created REAL NOT NULL,FOREIGN KEY(constraint_id) REFERENCES constraints(id));
CREATE INDEX IF NOT EXISTS decision_latest ON decisions(constraint_id,sequence);
CREATE TABLE IF NOT EXISTS revisions(id TEXT PRIMARY KEY,parent TEXT,source_generation INT NOT NULL,algorithm TEXT NOT NULL,manifest TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS poses(revision TEXT NOT NULL,keyframe TEXT NOT NULL,component TEXT NOT NULL,pose TEXT NOT NULL,uncertainty TEXT NOT NULL,PRIMARY KEY(revision,keyframe),FOREIGN KEY(revision) REFERENCES revisions(id),FOREIGN KEY(keyframe) REFERENCES keyframes(id));
CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS products(revision TEXT NOT NULL,kind TEXT NOT NULL,path TEXT NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(revision,kind),FOREIGN KEY(revision) REFERENCES revisions(id));
CREATE TABLE IF NOT EXISTS extractions(id TEXT PRIMARY KEY,session TEXT NOT NULL,state TEXT NOT NULL,manifest TEXT NOT NULL,created REAL NOT NULL,FOREIGN KEY(session) REFERENCES sessions(id));
CREATE TABLE IF NOT EXISTS extraction_members(extraction TEXT NOT NULL,keyframe TEXT NOT NULL,PRIMARY KEY(extraction,keyframe),FOREIGN KEY(extraction) REFERENCES extractions(id),FOREIGN KEY(keyframe) REFERENCES keyframes(id));
CREATE TABLE IF NOT EXISTS active_extractions(session TEXT PRIMARY KEY,extraction TEXT NOT NULL,FOREIGN KEY(session) REFERENCES sessions(id),FOREIGN KEY(extraction) REFERENCES extractions(id));
CREATE TABLE IF NOT EXISTS session_locations(session TEXT NOT NULL,uri TEXT NOT NULL,last_seen REAL NOT NULL,PRIMARY KEY(session,uri),FOREIGN KEY(session) REFERENCES sessions(id));
''')
            db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',('generation','0'))
            db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',('descriptor_generation','0'))
            db.execute('INSERT OR REPLACE INTO geo_index SELECT k.rowid,l.longitude,l.longitude,l.latitude,l.latitude FROM locations l JOIN keyframes k ON k.id=l.keyframe WHERE k.rowid NOT IN (SELECT rowid FROM geo_index)')
            db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    @staticmethod
    def _advance(db):db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='generation'")

    def generation(self):
        with self.connect() as db:return int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])

    def descriptor_generation(self):
        with self.connect() as db:return int(db.execute("SELECT value FROM metadata WHERE key='descriptor_generation'").fetchone()[0])

    def put_cloud(self,points):
        digest,p=cloud_digest(points);folder=self.cloud_root/digest[:2];folder.mkdir(exist_ok=True);target=folder/(digest+'.npz')
        if target.exists():self.load_cloud(digest);return digest
        temporary=folder/(digest+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('xb') as f:
                np.savez_compressed(f,points=p);f.flush();os.fsync(f.fileno())
            # A completed write can replace an equivalent concurrent write; the
            # identity derives from point content, not ZIP timestamps.
            os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        return digest

    def load_cloud(self,digest):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid cloud identity')
        path=self.cloud_root/digest[:2]/(digest+'.npz')
        with np.load(path,allow_pickle=False) as z:p=z['points']
        if cloud_digest(p)[0]!=digest:raise ValueError('Observation cloud integrity mismatch')
        return p

    def register_session(self,session_id,payload):
        if not session_id:raise ValueError('Missing session identity')
        encoded=canonical_json(payload);digest=identity(payload)
        with self.connect() as db:
            old=db.execute('SELECT digest FROM sessions WHERE id=?',(session_id,)).fetchone()
            if old:
                if old[0]!=digest:raise ValueError('Session identity already exists with different immutable metadata')
                return False
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(session_id,encoded,digest,time.time()));self._advance(db)
        return True

    def locate_session(self,session_id,folder):
        # Storage location is mutable catalog information, not sensor identity.
        uri=Path(folder).resolve().as_uri()
        with self.connect() as db:db.execute('INSERT INTO session_locations VALUES(?,?,?) ON CONFLICT(session,uri) DO UPDATE SET last_seen=excluded.last_seen',(session_id,uri,time.time()))

    def add_keyframe(self,frame,extraction=None):
        frame.validate();self.load_cloud(frame.cloud_hash)
        payload=asdict(frame);encoded=canonical_json(payload);digest=identity(payload)
        with self.connect() as db:
            if extraction:
                row=db.execute('SELECT session,state FROM extractions WHERE id=?',(extraction,)).fetchone()
                if row is None or row[0]!=frame.session_id:raise ValueError('Extraction/session mismatch')
            old=db.execute('SELECT digest FROM keyframes WHERE id=?',(frame.id,)).fetchone()
            if old:
                if old[0]!=digest:raise ValueError('Immutable keyframe conflict: use a new calibration/algorithm evidence identity')
                if extraction:db.execute('INSERT OR IGNORE INTO extraction_members VALUES(?,?)',(extraction,frame.id))
                return False
            db.execute('INSERT INTO keyframes(id,session,segment,stamp,calibration,payload,digest) VALUES(?,?,?,?,?,?,?)',
                       (frame.id,frame.session_id,frame.segment_id,frame.at_ns,frame.calibration_id,encoded,digest))
            loc=frame.location or {};position=loc.get('position') or {}
            if loc.get('usable_for_fusion') or loc.get('usable_for_index'):
                lat=position.get('latitude_deg');lon=position.get('longitude_deg');accuracy=loc.get('horizontal_accuracy_m')
                # Index padding is a retrieval allowance, not a fabricated
                # receiver accuracy. The immutable fix retains None if unknown.
                if accuracy is None:accuracy=100.
                if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (lat,lon,accuracy)) or not(-90<=lat<=90 and -180<=lon<=180 and 0<accuracy<=100):raise ValueError('Invalid keyframe location')
                db.execute('INSERT INTO locations VALUES(?,?,?,?)',(frame.id,lat,lon,accuracy))
                rowid=db.execute('SELECT rowid FROM keyframes WHERE id=?',(frame.id,)).fetchone()[0]
                db.execute('INSERT INTO geo_index VALUES(?,?,?,?,?)',(rowid,lon,lon,lat,lat))
            if extraction:db.execute('INSERT INTO extraction_members VALUES(?,?)',(extraction,frame.id))
            self._advance(db)
        return True

    def keyframe(self,identifier):
        with self.connect() as db:row=db.execute('SELECT payload,digest FROM keyframes WHERE id=?',(identifier,)).fetchone()
        if row is None:raise KeyError(identifier)
        payload=json.loads(row[0])
        if identity(payload)!=row[1] or payload.get('id')!=identifier:raise ValueError('Keyframe metadata integrity mismatch')
        return Keyframe(**payload)

    def keyframes(self,session=None):
        with self.connect() as db:
            cursor=db.execute('SELECT payload,digest FROM keyframes'+(' WHERE session=?' if session else '')+' ORDER BY session,segment,stamp',(session,) if session else ())
            # Streaming avoids a hard history cutoff or loading point clouds.
            for row in cursor:
                payload=json.loads(row[0])
                if identity(payload)!=row[1]:raise ValueError('Keyframe metadata integrity mismatch')
                yield Keyframe(**payload)

    def begin_extraction(self,session,manifest):
        identifier=identity({'session':session,'manifest':manifest})
        with self.connect() as db:db.execute('INSERT OR IGNORE INTO extractions VALUES(?,?,?,?,?)',(identifier,session,'building',canonical_json(manifest),time.time()))
        return identifier

    def extraction_state(self,identifier):
        with self.connect() as db:row=db.execute('SELECT state FROM extractions WHERE id=?',(identifier,)).fetchone()
        return row[0] if row else None

    def complete_extraction(self,identifier):
        with self.connect() as db:
            row=db.execute('SELECT session,state FROM extractions WHERE id=?',(identifier,)).fetchone()
            if not row:raise KeyError(identifier)
            if not db.execute('SELECT 1 FROM extraction_members WHERE extraction=?',(identifier,)).fetchone():raise ValueError('An empty extraction cannot replace existing map evidence')
            active=db.execute('SELECT extraction FROM active_extractions WHERE session=?',(row[0],)).fetchone()
            if row[1]=='complete' and active and active[0]==identifier:return False
            db.execute("UPDATE extractions SET state='complete' WHERE id=?",(identifier,))
            db.execute('INSERT INTO active_extractions VALUES(?,?) ON CONFLICT(session) DO UPDATE SET extraction=excluded.extraction',(row[0],identifier));self._advance(db)
        return True

    def active_keyframes(self):
        with self.connect() as db:
            query='''SELECT k.payload,k.digest FROM keyframes k WHERE NOT EXISTS(SELECT 1 FROM extraction_members m WHERE m.keyframe=k.id)
OR EXISTS(SELECT 1 FROM extraction_members m JOIN active_extractions a ON a.extraction=m.extraction WHERE m.keyframe=k.id) ORDER BY k.session,k.segment,k.stamp'''
            for row in db.execute(query):
                payload=json.loads(row[0])
                if identity(payload)!=row[1]:raise ValueError('Keyframe metadata integrity mismatch')
                yield Keyframe(**payload)

    def extraction_keyframes(self,identifier):
        with self.connect() as db:
            rows=db.execute('SELECT k.payload,k.digest FROM keyframes k JOIN extraction_members m ON m.keyframe=k.id WHERE m.extraction=? ORDER BY k.segment,k.stamp',(identifier,)).fetchall()
        result=[]
        for row in rows:
            payload=json.loads(row[0])
            if identity(payload)!=row[1]:raise ValueError('Keyframe metadata integrity mismatch')
            result.append(Keyframe(**payload))
        return result

    def add_descriptor(self,keyframe,version,vector):
        a=np.asarray(vector,dtype='<f4').reshape(-1)
        if not version or not 1<=len(a)<=8192 or not np.isfinite(a).all():raise ValueError('Invalid place descriptor')
        with self.connect() as db:
            old=db.execute('SELECT vector FROM descriptors WHERE keyframe=? AND version=?',(keyframe,version)).fetchone()
            if old:
                if old[0]!=a.tobytes():raise ValueError('Descriptor version conflict; increment version for a new algorithm')
                return False
            db.execute('INSERT INTO descriptors VALUES(?,?,?,?)',(keyframe,version,a.tobytes(),len(a)));self._advance(db)
            db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='descriptor_generation'")
        return True

    def descriptors(self,version):
        with self.connect() as db:
            for key,session,vector,dimensions in db.execute('SELECT d.keyframe,k.session,d.vector,d.dimensions FROM descriptors d JOIN keyframes k ON k.id=d.keyframe WHERE d.version=? ORDER BY d.keyframe',(version,)):
                a=np.frombuffer(vector,dtype='<f4').copy()
                if len(a)!=dimensions:raise ValueError('Corrupt descriptor')
                yield key,session,a

    def nearby(self,latitude,longitude,radius_m=150.,exclude_session=None,limit=128):
        if not all(math.isfinite(v) for v in (latitude,longitude,radius_m)) or not(-90<=latitude<=90 and -180<=longitude<=180 and 0<radius_m<=100_000):raise ValueError('Invalid geographic query')
        # Expand for stored receiver uncertainty; exact great-circle distance is
        # checked afterward. Longitude query wraps around the antimeridian.
        degrees=(radius_m+300)/110000.;latlo=max(-90,latitude-degrees);lathi=min(90,latitude+degrees)
        lonspan=min(180.,degrees/max(.001,math.cos(math.radians(latitude))))
        if lonspan>=180:ranges=[(-180,180)]
        elif longitude-lonspan<-180:ranges=[(-180,longitude+lonspan),(longitude-lonspan+360,180)]
        elif longitude+lonspan>180:ranges=[(longitude-lonspan,180),(-180,longitude+lonspan-360)]
        else:ranges=[(longitude-lonspan,longitude+lonspan)]
        with self.connect() as db:
            rows={}
            for lo,hi in ranges:
                for row in db.execute('SELECT l.keyframe,l.latitude,l.longitude,l.accuracy,k.session FROM geo_index g JOIN keyframes k ON k.rowid=g.rowid JOIN locations l ON l.keyframe=k.id WHERE g.max_lat>=? AND g.min_lat<=? AND g.max_lon>=? AND g.min_lon<=?',(latlo,lathi,lo,hi)):
                    rows[row[0]]=row
        result=[]
        for key,lat,lon,accuracy,session in rows.values():
            if session==exclude_session:continue
            if abs((lon-longitude+180)%360-180)>lonspan:continue
            distance=horizontal_distance((latitude,longitude),(lat,lon))
            if distance<=radius_m+3*accuracy:result.append({'id':key,'session_id':session,'distance_m':distance,'search_padding_m':accuracy})
        return sorted(result,key=lambda r:r['distance_m'])[:limit]

    def add_constraint(self,constraint):
        constraint.validate();payload=asdict(constraint);encoded=canonical_json(payload);digest=identity(payload)
        with self.connect() as db:
            old=db.execute('SELECT digest FROM constraints WHERE id=?',(constraint.id,)).fetchone()
            if old:
                if old[0]!=digest:raise ValueError('Immutable constraint conflict')
                return False
            db.execute('INSERT INTO constraints VALUES(?,?,?,?,?,?)',(constraint.id,constraint.source_id,constraint.target_id,constraint.kind,encoded,digest))
            db.execute('INSERT INTO decisions(constraint_id,state,reason,diagnostics,created) VALUES(?,?,?,?,?)',
                       (constraint.id,constraint.state,'initial measurement',canonical_json(constraint.provenance),time.time()))
            self._advance(db)
        return True

    def decide_constraint(self,identifier,state,reason,diagnostics):
        if state not in ('accepted','rejected','candidate') or not reason:raise ValueError('Explicit constraint decision and reason required')
        with self.connect() as db:
            db.execute('INSERT INTO decisions(constraint_id,state,reason,diagnostics,created) VALUES(?,?,?,?,?)',(identifier,state,reason,canonical_json(diagnostics),time.time()));self._advance(db)

    def constraints(self,accepted_only=True):
        with self.connect() as db:
            rows=db.execute('''SELECT c.payload,d.state,d.reason,d.diagnostics,c.digest FROM constraints c JOIN decisions d ON d.sequence=(SELECT MAX(sequence) FROM decisions WHERE constraint_id=c.id)''').fetchall()
        result=[]
        for payload,state,reason,diagnostics,digest in rows:
            if accepted_only and state!='accepted':continue
            item=json.loads(payload)
            if identity(item)!=digest:raise ValueError('Constraint metadata integrity mismatch')
            item.update(state=state,decision_reason=reason,decision_diagnostics=json.loads(diagnostics));result.append(item)
        return result

    def event(self,kind,payload):
        with self.connect() as db:db.execute('INSERT INTO events(kind,payload,created) VALUES(?,?,?)',(kind,canonical_json(payload),time.time()))

    def current_revision(self):
        with self.connect() as db:
            row=db.execute("SELECT value FROM metadata WHERE key='active_revision'").fetchone()
        return row[0] if row else None

    def publish_revision(self,algorithm,source_generation,poses,manifest,activate=True):
        if not algorithm or not poses:raise ValueError('A named algorithm and nonempty checked poses are required')
        if manifest.get('validated') is not True or manifest.get('unresolved_constraints',1)!=0:raise ValueError('Unvalidated graph cannot be published')
        for key,value in poses.items():
            checked_pose(value['map_from_body'])
            if not value.get('component'):raise ValueError('Pose coordinate component must be explicit')
            canonical_json(value.get('uncertainty',{}))
        poses={key:{'component':value['component'],'map_from_body':checked_pose(value['map_from_body']).tolist(),'uncertainty':value.get('uncertainty',{})} for key,value in poses.items()}
        with self.connect() as db:
            # BEGIN IMMEDIATE establishes one evidence-generation publication.
            db.execute('BEGIN IMMEDIATE')
            actual=int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])
            if actual!=source_generation:raise ValueError('Evidence changed during optimization; solve a fresh snapshot')
            parent_row=db.execute("SELECT value FROM metadata WHERE key='active_revision'").fetchone();parent=parent_row[0] if parent_row else None
            known={row[0] for row in db.execute('''SELECT k.id FROM keyframes k WHERE NOT EXISTS(SELECT 1 FROM extraction_members m WHERE m.keyframe=k.id)
OR EXISTS(SELECT 1 FROM extraction_members m JOIN active_extractions a ON a.extraction=m.extraction WHERE m.keyframe=k.id)''')}
            if set(poses)!=known:raise ValueError('A revision must account for all keyframes, using separate components where needed')
            payload={**manifest,'algorithm':algorithm,'source_generation':source_generation,'parent':parent,'poses_digest':identity(poses)}
            revision=identity(payload)
            db.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?,?,?,?)',(revision,parent,source_generation,algorithm,canonical_json(payload),time.time()))
            for key,value in poses.items():db.execute('INSERT OR IGNORE INTO poses VALUES(?,?,?,?,?)',(revision,key,value['component'],canonical_json(value['map_from_body']),canonical_json(value.get('uncertainty',{}))))
            if activate:db.execute("INSERT INTO metadata VALUES('active_revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(revision,))
        return revision

    def apply_decisions(self,source_generation,decisions):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            actual=int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])
            if actual!=source_generation:raise ValueError('Evidence changed before graph decisions were committed')
            for decision in decisions:
                db.execute('INSERT INTO decisions(constraint_id,state,reason,diagnostics,created) VALUES(?,?,?,?,?)',
                           (decision['id'],'rejected',decision['reason'],canonical_json(decision),time.time()))
            if decisions:self._advance(db)
            return int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])

    def revision_manifest(self,revision=None):
        revision=revision or self.current_revision()
        with self.connect() as db:row=db.execute('SELECT manifest FROM revisions WHERE id=?',(revision,)).fetchone()
        if row is None:raise KeyError(revision)
        return json.loads(row[0])

    def register_product(self,revision,kind,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():raise ValueError('Revision product must be a local atlas file')
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        relative=path.relative_to(self.root.resolve()).as_posix()
        with self.connect() as db:
            old=db.execute('SELECT path,digest FROM products WHERE revision=? AND kind=?',(revision,kind)).fetchone()
            if old:
                if old!=(relative,digest):raise ValueError('Immutable map product conflict')
                return digest
            db.execute('INSERT INTO products VALUES(?,?,?,?)',(revision,kind,relative,digest))
        return digest

    def product(self,revision,kind,verify=True):
        with self.connect() as db:row=db.execute('SELECT path,digest FROM products WHERE revision=? AND kind=?',(revision,kind)).fetchone()
        if row is None:raise KeyError((revision,kind))
        path=(self.root/row[0]).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Product path escaped atlas')
        if verify:
            with path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=row[1]:raise ValueError('Map product integrity mismatch')
        return path

    def revision_poses(self,revision=None):
        revision=revision or self.current_revision()
        if revision is None:return {}
        with self.connect() as db:rows=db.execute('SELECT keyframe,component,pose,uncertainty FROM poses WHERE revision=?',(revision,)).fetchall()
        result={key:{'component':component,'map_from_body':json.loads(pose),'uncertainty':json.loads(uncertainty)} for key,component,pose,uncertainty in rows}
        if identity(result)!=self.revision_manifest(revision).get('poses_digest'):raise ValueError('Pose revision integrity mismatch')
        return result

    def activate_revision(self,revision,reason,require_map=False,require_current_generation=False):
        if not reason:raise ValueError('Rollback reason required')
        if require_map:
            self.product(revision,'static_map');self.product(revision,'map_report');self.product(revision,'spatial_index')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT source_generation FROM revisions WHERE id=?',(revision,)).fetchone()
            if row is None:raise KeyError(revision)
            if require_current_generation and row[0]!=int(db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0]):raise ValueError('Evidence changed before map activation')
            db.execute("INSERT INTO metadata VALUES('active_revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(revision,))
            db.execute('INSERT INTO events(kind,payload,created) VALUES(?,?,?)',('revision_activated',canonical_json({'revision':revision,'reason':reason}),time.time()))

    def summary(self):
        with self.connect() as db:
            counts={name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ('sessions','keyframes','constraints','revisions','locations')}
        return {**counts,'schema_version':SCHEMA_VERSION,'generation':self.generation(),'active_revision':self.current_revision()}

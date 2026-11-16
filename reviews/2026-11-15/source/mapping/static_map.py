"""Rebuild versioned static evidence from original local clouds and checked poses.

Each verified physical drive contributes at most one positive/negative visit per cell. A pose
correction reprojects the originals; it does not incrementally smear old voxels.
Only completed products may be activated with their matching pose revision.
"""
from collections import defaultdict
import json, math, os, uuid, sqlite3
import runtime_clock as time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from offline_jobs import check_work
from route_map import visible_absences

VERSION='reprojected-static-evidence-v4-physical-drives'
VOXEL=.4
COVERAGE=2.
EXPIRY_S=90*86400

class StaticMap:
    version=VERSION
    def __init__(self,store):self.store=store
    @staticmethod
    def _drive(frame,allow_synthetic):
        lineage=frame.source_ref.get('drive_lineage',{})
        group=lineage.get('physical_drive_group');epoch=lineage.get('mount_epoch')
        if (lineage.get('schema_version')!=1 or lineage.get('verified') is not True or
            lineage.get('persistence_eligible') is not True or lineage.get('mount_epoch_known') is not True or
            not isinstance(group,str) or not 0<len(group)<=256 or
            not isinstance(epoch,str) or not 0<len(epoch)<=200 or epoch.startswith('legacy-unknown:')):return None
        if lineage.get('basis')=='synthetic_fixture':
            return group if allow_synthetic and lineage.get('synthetic_allowed') is True else None
        evidence=lineage.get('evidence',{})
        if (lineage.get('basis')!='reviewed_physical_drive' or not isinstance(evidence,dict) or
            not evidence.get('catalog_revision') or not evidence.get('catalog_input_signature') or
            not evidence.get('driving_reviews') or not lineage.get('source_route_ids')):return None
        return group

    def _compatible_product(self,revision,allow_synthetic=False,verify=True):
        # The report is tiny and always verified; skipping a large spatial-index
        # rehash must not skip the semantic-version/physical-evidence boundary.
        report=json.loads(self.store.product(revision,'map_report',verify=True).read_text(encoding='utf-8'))
        if report.get('map_backend')!=self.version:
            raise ValueError('Legacy static map needs a new physical-drive-aware revision')
        if report.get('synthetic_evidence_allowed') and not allow_synthetic:
            raise ValueError('Synthetic map evidence cannot be read as a physical-drive map')
        return report

    def rebuild(self,revision_id,cancel=None,now=None,allow_synthetic=False):
        now=time.time() if now is None else now
        try:
            old=self._compatible_product(revision_id,allow_synthetic);self.store.product(revision_id,'static_map');self.store.product(revision_id,'spatial_index')
            if old.get('synthetic_evidence_allowed') is not bool(allow_synthetic):
                raise ValueError('Map evidence mode changed; publish a new revision')
            return old
        except KeyError:pass
        try:existing_product=self.store.product(revision_id,'static_map')
        except KeyError:existing_product=None
        poses=self.store.revision_poses(revision_id);frames=[self.store.keyframe(key) for key in poses]
        if not poses or set(poses)!={f.id for f in frames}:raise ValueError('Static rebuild requires poses for every immutable keyframe')
        by_drive=defaultdict(list);unknown_sessions=set();unknown_frames=0
        for frame in frames:
            group=self._drive(frame,allow_synthetic)
            if group is None:unknown_sessions.add(frame.session_id);unknown_frames+=1
            else:by_drive[group].append(frame)
        def stamp(frame):
            value=frame.quality.get('capture_wall_ns',0)
            return value/1e9 if type(value) in (int,float) and math.isfinite(value) and 0<value/1e9<=now+86400 else 0.
        ordered=sorted(by_drive,key=lambda s:min((stamp(f) for f in by_drive[s] if stamp(f)>0),default=math.inf))
        state={};coverage={};uncertain_time_sessions=set()
        for drive in ordered:
            check_work(cancel)
            uncertain_time_sessions.update(f.session_id for f in by_drive[drive] if not stamp(f))
            visit=sorted((f for f in by_drive[drive] if stamp(f)),key=lambda f:f.at_ns)
            if not visit:continue
            times=[stamp(f) for f in visit];when=max(times)
            if when<=0 or when>now+86400:
                uncertain_time_sessions.update(f.session_id for f in visit)
                # Geometry is still retained for localization, but an unknown
                # date must not establish recent static-object persistence.
                continue
            counts={};covered=set();absent=set()
            active_by_component=defaultdict(list)
            for (component,x,y,z),item in state.items():
                if item['seen']>=2 and item['missed']<3 and when-item['last']<EXPIRY_S:active_by_component[component].append((x,y,z))
            trees={component:(np.asarray(keys,float)+.5)*VOXEL for component,keys in active_by_component.items()}
            trees={component:(p,cKDTree(p)) for component,p in trees.items()}
            for frame in visit:
                check_work(cancel);entry=poses[frame.id];component=entry['component'];T=np.asarray(entry['map_from_body']);R,t=T[:3,:3],T[:3,3]
                body=self.store.load_cloud(frame.cloud_hash);world=body@R.T+t
                keys=set(map(tuple,np.floor(world/VOXEL).astype(np.int64)))
                # Do not accumulate the three-view support across unrelated
                # local clocks, discontinuities, calibration or mount changes.
                lineage=frame.source_ref['drive_lineage']
                segment=(frame.session_id,frame.segment_id,frame.calibration_id,lineage['mount_epoch'])
                for xyz in keys:
                    key=((component,*xyz),segment);row=counts.setdefault(key,{'n':0,'first':frame.at_ns,'last':frame.at_ns,'origin':t.copy(),'baseline':0.})
                    row['n']+=1;row['last']=frame.at_ns;row['baseline']=max(row['baseline'],float(np.linalg.norm(t-row['origin'])))
                visibility=frame.source_ref.get('visibility_cloud_hash')
                if not visibility:continue
                points=self.store.load_cloud(visibility);origin=np.asarray(frame.source_ref.get('sensor_origin_body',[0,0,0]),float)
                rays=origin+(points[::max(1,math.ceil(len(points)/400)),None,:]-origin)*np.linspace(.1,1.,16)[None,:,None]
                mapped=rays.reshape(-1,3)@R.T+t
                covered.update((component,*xyz) for xyz in set(map(tuple,np.floor(mapped/COVERAGE).astype(np.int64))))
                if component in trees:
                    existing,tree=trees[component];near=existing[tree.query_ball_point(t,65)]
                    absent.update((component,*key) for key in visible_absences(near,R,t,points,keys,origin))
            positive={key[0] for key,row in counts.items() if row['n']>=3 and row['last']-row['first']>=300_000_000 and row['baseline']>=2.}
            for key in positive:
                row=state.setdefault(key,{'seen':0,'missed':0,'last':when})
                if row['missed']>=3 or when-row['last']>=EXPIRY_S:row['seen']=0
                row.update(seen=row['seen']+1,missed=0,last=when)
            for key in absent-{key[0] for key in counts}:
                if key in state:state[key]['missed']+=1
            for key in covered:
                row=coverage.setdefault(key,{'seen':0,'last':when});row['seen']+=1;row['last']=when
        check_work(cancel)
        components=sorted({p['component'] for p in poses.values()});index={name:i for i,name in enumerate(components)}
        keys=sorted(state);cover_keys=sorted(coverage)
        points=np.array([key[1:] for key in keys],dtype=np.float32).reshape(-1,3);points=(points+.5)*VOXEL
        count=np.array([state[key]['seen'] for key in keys],np.int32);missed=np.array([state[key]['missed'] for key in keys],np.int32)
        last=np.array([state[key]['last'] for key in keys],np.float64)
        active=(count>=2)&(missed<3)&(now-last<EXPIRY_S)
        report={'schema_version':2,'status':'complete','map_backend':self.version,'revision':revision_id,'sessions':len({f.session_id for f in frames}),'keyframes':len(frames),
                'verified_physical_drive_groups':len(by_drive),'unknown_lineage_sessions':len(unknown_sessions),
                'unknown_lineage_keyframes':unknown_frames,'synthetic_evidence_allowed':bool(allow_synthetic),
                'components':len(components),'promoted_static_voxels':int(active.sum()),'pending_voxels':int((count<2).sum()),
                'forgotten_voxels':int(((missed>=3)|(now-last>=EXPIRY_S)).sum()),'coverage_cells':sum(coverage[k]['seen']>=2 and now-coverage[k]['last']<EXPIRY_S for k in cover_keys),
                'unknown_capture_date_sessions':len(uncertain_time_sessions),'voxel_m':VOXEL,'coverage_cell_m':COVERAGE,'created_at_s':now,'warning_override':False,
                'limitations':'Static evidence requires three views spanning >=0.3s and >=2m within one continuous capture/calibration/mount segment, then two verified distinct physical drives. Multiple sessions, restarts or remounts in one drive count once. Unknown drive/mount lineage retains original geometry but contributes no persistence, coverage or disappearance visit. Clear-view absence is quantized ray evidence, not proof of empty road. No semantic labels or warning veto.'}
        directory=self.store.root/'revisions'/revision_id;directory.mkdir(parents=True,exist_ok=True)
        map_path=directory/'static-map.npz';temporary=directory/('static-'+uuid.uuid4().hex+'.tmp')
        arrays=dict(map_backend=np.array(self.version),points=points,component_index=np.array([index[k[0]] for k in keys],np.int32),components=np.array(components),
                                seen=count,missed=missed,last=last,coverage=np.array([k[1:] for k in cover_keys],np.int32).reshape(-1,3),
                                coverage_component=np.array([index[k[0]] for k in cover_keys],np.int32),coverage_seen=np.array([coverage[k]['seen'] for k in cover_keys],np.int32),
                                coverage_last=np.array([coverage[k]['last'] for k in cover_keys],np.float64))
        if existing_product:
            with np.load(existing_product,allow_pickle=False) as saved:
                if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):raise ValueError('Existing immutable map product differs from rebuilt evidence')
        else:
            with temporary.open('xb') as f:
                np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            check_work(cancel);os.replace(temporary,map_path)
        spatial_path=directory/'spatial-map.sqlite'
        try:self.store.product(revision_id,'spatial_index')
        except KeyError:
            spatial_tmp=directory/('spatial-'+uuid.uuid4().hex+'.tmp')
            db=sqlite3.connect(spatial_tmp)
            try:
                db.executescript('CREATE TABLE components(id INT PRIMARY KEY,name TEXT UNIQUE);CREATE TABLE voxels(component INT,x INT,y INT,z INT,seen INT,missed INT,last REAL,PRIMARY KEY(component,x,y,z));CREATE TABLE coverage(component INT,x INT,y INT,z INT,seen INT,last REAL,PRIMARY KEY(component,x,y,z));')
                db.executemany('INSERT INTO components VALUES(?,?)',enumerate(components))
                for start in range(0,len(keys),2000):
                    check_work(cancel)
                    db.executemany('INSERT INTO voxels VALUES(?,?,?,?,?,?,?)',[(index[k[0]],*map(int,k[1:]),state[k]['seen'],state[k]['missed'],state[k]['last']) for k in keys[start:start+2000]])
                for start in range(0,len(cover_keys),2000):
                    check_work(cancel)
                    db.executemany('INSERT INTO coverage VALUES(?,?,?,?,?,?)',[(index[k[0]],*map(int,k[1:]),coverage[k]['seen'],coverage[k]['last']) for k in cover_keys[start:start+2000]])
                db.commit()
            finally:db.close()
            check_work(cancel);os.replace(spatial_tmp,spatial_path)
        report_path=directory/'map-report.json';report_tmp=directory/'map-report.json.tmp'
        report_tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');os.replace(report_tmp,report_path)
        self.store.register_product(revision_id,'static_map',map_path);self.store.register_product(revision_id,'spatial_index',spatial_path);self.store.register_product(revision_id,'map_report',report_path)
        return report

    def local_snapshot(self,revision,component,position,radius_m=80,now=None,verify=True,allow_synthetic=False):
        """Bounded spatial query for live use; never loads the entire atlas cloud."""
        now=time.time() if now is None else now;position=np.asarray(position,float)
        if position.shape!=(3,) or not np.isfinite(position).all() or not 0<radius_m<=200:raise ValueError('Invalid local map query')
        self._compatible_product(revision,allow_synthetic,verify=verify)
        path=self.store.product(revision,'spatial_index',verify=verify)
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            row=db.execute('SELECT id FROM components WHERE name=?',(component,)).fetchone()
            if row is None:return {'points':np.empty((0,3)),'coverage':set(),'revision':revision,'component':component}
            def bounds(scale):return [int(math.floor((position[axis]+sign*radius_m)/scale)) for axis in range(3) for sign in (-1,1)]
            cells=db.execute('SELECT x,y,z FROM voxels WHERE component=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ? AND seen>=2 AND missed<3 AND last>?',(row[0],*bounds(VOXEL),now-EXPIRY_S)).fetchall()
            cover=db.execute('SELECT x,y,z FROM coverage WHERE component=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ? AND seen>=2 AND last>?',(row[0],*bounds(COVERAGE),now-EXPIRY_S)).fetchall()
        finally:db.close()
        return {'points':(np.asarray(cells,float).reshape(-1,3)+.5)*VOXEL,'coverage':set(cover),'revision':revision,'component':component}

    def snapshot(self,revision=None,now=None,allow_synthetic=False):
        revision=revision or self.store.current_revision();now=time.time() if now is None else now
        if revision is None:return {'revision':None,'components':{}}
        self._compatible_product(revision,allow_synthetic)
        with np.load(self.store.product(revision,'static_map'),allow_pickle=False) as data:
            active=(data['seen']>=2)&(data['missed']<3)&(now-data['last']<EXPIRY_S)
            coverage=(data['coverage_seen']>=2)&(now-data['coverage_last']<EXPIRY_S)
            result={}
            for i,component in enumerate(data['components'].tolist()):
                points=data['points'][active&(data['component_index']==i)].copy()
                cover=data['coverage'][coverage&(data['coverage_component']==i)]
                result[component]={'points':points,'coverage':set(map(tuple,cover.tolist()))}
        return {'revision':revision,'components':result}

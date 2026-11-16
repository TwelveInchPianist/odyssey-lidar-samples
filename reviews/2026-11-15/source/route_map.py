"""Phase 2: conservative local 3D submaps and advisory map evidence.

No map result can cancel an alert, relax readiness, or lower priority for a
corridor object. Geometric localization can be unavailable or ambiguous.
"""
import hashlib, io, json, math, queue, sqlite3, threading
import runtime_clock as time
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from scipy.spatial import cKDTree
from fcw_engine import Ground,Mount,transform,ego_delta
from recording import frames
from offline_jobs import check_work

SCHEMA=1
VOXEL=.4
COVERAGE=2.

def rotation_z(a):
    c,s=math.cos(a),math.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def downsample(p,voxel=.45,limit=1600):
    p=np.asarray(p,dtype=float)[:,:3];p=p[np.isfinite(p).all(1)]
    if not len(p):return p
    _,idx=np.unique(np.floor(p/voxel).astype(np.int64),axis=0,return_index=True)
    p=p[np.sort(idx)]
    return p[::max(1,math.ceil(len(p)/limit))]

def features(points):
    p=np.asarray(points)[:,:3]
    p=p[(p[:,0]>4)&(p[:,0]<65)&(abs(p[:,1])<30)&(p[:,2]>.4)&(p[:,2]<5)]
    return downsample(p)

def icp(source,target,seed=None):
    source=downsample(source);target=downsample(target)
    if min(len(source),len(target))<80:return None
    spread=np.linalg.svd(source-source.mean(0),compute_uv=False)
    if spread[-1]/max(spread[0],1e-9)<.025:return None
    R,t=(np.eye(3),np.zeros(3)) if seed is None else (seed[0].copy(),seed[1].copy())
    tree=cKDTree(target)
    for _ in range(16):
        moved=source@R.T+t;distance,index=tree.query(moved)
        good=distance<=min(2.,max(.08,float(np.quantile(distance,.7))))
        if good.sum()<60:return None
        a,b=source[good],target[index[good]];ac=a-a.mean(0);bc=b-b.mean(0)
        U,_,Vt=np.linalg.svd(ac.T@bc);r=Vt.T@U.T
        if np.linalg.det(r)<0:Vt[-1]*=-1;r=Vt.T@U.T
        tr=b.mean(0)-a.mean(0)@r.T
        change=np.linalg.norm(R-r)+np.linalg.norm(t-tr);R,t=r,tr
        if change<.0003:break
    distances,_=tree.query(source@R.T+t);inliers=distances<.35
    overlap=float(inliers.mean());rms=float(np.sqrt(np.mean(distances[inliers]**2))) if inliers.any() else 99.
    if overlap<.65 or rms>.22:return None
    return {'R':R,'t':t,'rmse_m':rms,'overlap':overlap}

def bootstrap(source,anchors):
    """Match only a few similarly shaped anchors; reject distinct near-equal fits."""
    if len(source)<80:return None
    candidates=[]
    shape=np.percentile(np.linalg.norm(source-source.mean(0),axis=1),[25,50,75])
    ranked=sorted(anchors,key=lambda a:np.linalg.norm(shape-np.percentile(np.linalg.norm(a['points']-a['points'].mean(0),axis=1),[25,50,75])))[:4]
    for anchor in ranked:
        target=anchor['points']
        for angle in np.arange(0,2*math.pi,math.pi/6):
            R=rotation_z(angle);t=target.mean(0)-source.mean(0)@R.T
            fit=icp(source,target,(R,t))
            if fit:
                fit['submap']=anchor['submap'];fit['score']=fit['rmse_m']+.25*(1-fit['overlap'])
                # Multiple seeds converging on one solution are not rival places.
                if not any(v['submap']==fit['submap'] and np.linalg.norm(v['t']-fit['t'])<1 and np.linalg.norm(v['R']-fit['R'])<.15 for v in candidates):candidates.append(fit)
    if not candidates:return None
    candidates.sort(key=lambda c:c['score'])
    if len(candidates)>1 and candidates[1]['score']<candidates[0]['score']*1.3+.02:return None
    return candidates[0]

class MapStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.path=self.root/'route-map.sqlite'
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS visits(id TEXT PRIMARY KEY, created REAL);
CREATE TABLE IF NOT EXISTS submaps(id TEXT PRIMARY KEY, anchor BLOB, created REAL);
CREATE TABLE IF NOT EXISTS keyframes(id TEXT PRIMARY KEY,map TEXT,cloud BLOB,created REAL);
CREATE TABLE IF NOT EXISTS voxels(map TEXT,x INT,y INT,z INT,seen INT,missed INT,last REAL,PRIMARY KEY(map,x,y,z));
CREATE TABLE IF NOT EXISTS coverage(map TEXT,x INT,y INT,z INT,seen INT,PRIMARY KEY(map,x,y,z));''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=20)
        try:
            with db:yield db
        finally:db.close()
    def anchors(self):
        with self.connect() as db:
            rows=db.execute('SELECT map,cloud FROM keyframes ORDER BY created DESC LIMIT 128').fetchall()
            if not rows:rows=db.execute('SELECT id,anchor FROM submaps ORDER BY created DESC LIMIT 128').fetchall()
        return [{'submap':name,'points':np.load(io.BytesIO(blob),allow_pickle=False)} for name,blob in rows]
    def has_visit(self,session):
        with self.connect() as db:return db.execute('SELECT 1 FROM visits WHERE id=?',(session,)).fetchone() is not None
    def commit(self,session,observations,anchors,coverage,absent=None,now=None,keyframes=None,cancel=None):
        now=time.time() if now is None else now;absent=absent or {}
        with self.connect() as db:
            if db.execute('SELECT 1 FROM visits WHERE id=?',(session,)).fetchone():return False
            for key,points in anchors.items():
                buf=io.BytesIO();np.save(buf,np.asarray(points,dtype=np.float32),allow_pickle=False)
                db.execute('INSERT OR IGNORE INTO submaps VALUES(?,?,?)',(key,buf.getvalue(),now))
            for key,name,points in keyframes or []:
                buf=io.BytesIO();np.save(buf,np.asarray(points,dtype=np.float32),allow_pickle=False)
                db.execute('INSERT OR IGNORE INTO keyframes VALUES(?,?,?,?)',(key,name,buf.getvalue(),now))
            for name,keys in observations.items():
                check_work(cancel)
                for x,y,z in keys:
                    db.execute('INSERT INTO voxels VALUES(?,?,?,?,1,0,?) ON CONFLICT(map,x,y,z) DO UPDATE SET seen=CASE WHEN missed>=3 OR excluded.last-last>=7776000 THEN 1 ELSE seen+1 END,missed=0,last=excluded.last',(name,int(x),int(y),int(z),now))
            for name,keys in coverage.items():
                check_work(cancel)
                for x,y,z in keys:
                    db.execute('INSERT INTO coverage VALUES(?,?,?,?,1) ON CONFLICT(map,x,y,z) DO UPDATE SET seen=seen+1',(name,int(x),int(y),int(z)))
            for name,keys in absent.items():
                for x,y,z in keys:
                    if (x,y,z) not in observations.get(name,set()):db.execute('UPDATE voxels SET missed=missed+1 WHERE map=? AND x=? AND y=? AND z=?',(name,int(x),int(y),int(z)))
            db.execute('INSERT INTO visits VALUES(?,?)',(session,now))
        return True
    def snapshot(self,name,now=None):
        now=time.time() if now is None else now
        with self.connect() as db:
            rows=db.execute('SELECT x,y,z,seen,missed,last FROM voxels WHERE map=?',(name,)).fetchall()
            cover=db.execute('SELECT x,y,z FROM coverage WHERE map=? AND seen>=2',(name,)).fetchall()
        active=np.array([(x,y,z) for x,y,z,n,miss,last in rows if n>=2 and miss<3 and now-last<90*86400],dtype=float).reshape(-1,3)
        return {'points':(active+.5)*VOXEL,'coverage':set(tuple(v) for v in cover),
                'pending':sum(n<2 for _,_,_,n,_,_ in rows),'forgotten':sum(miss>=3 or now-last>=90*86400 for _,_,_,_,miss,last in rows)}

def visible_absences(stored_world,R,t,points_sensor,observed_keys,sensor_origin=None):
    """Negative evidence only where a measured ray reaches BEHIND a map voxel."""
    if not len(stored_world):return set()
    origin=np.zeros(3) if sensor_origin is None else np.asarray(sensor_origin)
    local=(stored_world-t)@R-origin
    obs=np.asarray(points_sensor)[:,:3]-origin;obs=obs[np.isfinite(obs).all(1)]
    # Work in body coordinates; narrow central vertical sector avoids guessing
    # coverage at the MEMS field-of-view edges after mounting tilt.
    def bins(p):return np.floor(np.column_stack([np.arctan2(p[:,1],p[:,0]),np.arctan2(p[:,2],np.hypot(p[:,0],p[:,1]))])/math.radians(.5)).astype(int)
    ob=bins(obs);ranges=np.linalg.norm(obs,axis=1);nearest={}
    for key,dist in zip(map(tuple,ob),ranges):nearest[key]=min(nearest.get(key,math.inf),float(dist))
    cells=bins(local);result=set()
    for world,p,key in zip(stored_world,local,map(tuple,cells)):
        distance=float(np.linalg.norm(p));voxel=tuple(np.floor(world/VOXEL).astype(int))
        if voxel in observed_keys:continue
        if p[0]>5 and distance<55 and abs(math.atan2(p[1],p[0]))<math.radians(50) and nearest.get(key,0)>distance+1.:
            result.add(voxel)
    return result

def update_map(folder,map_root,cancel=None):
    folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    report={'schema_version':SCHEMA,'status':'unavailable','localized_keyframes':0,'eligible_keyframes':0,
            'coverage_percent':0.,'new_submaps':0,'promoted_static_voxels':0,'forgotten_voxels':0,
            'warning_override':False,'coordinate_system':'local 3D submaps; no assumed global GPS alignment'}
    def finish():
        (folder/'map-review.json').write_text(json.dumps(report,indent=2));return report
    if not (folder/'final-state.json').exists():report['reason']='completed session required';return finish()
    if not cfg.get('map_enabled',True):report['reason']='map disabled';return finish()
    if not (cfg.get('capture_mode')=='live' and cfg.get('timing_verified') and cfg.get('mount',{}).get('measured') and cfg.get('mount',{}).get('alignment_verified')):
        report['reason']='verified mounting, timing and live data required';return finish()
    store=MapStore(map_root);session=cfg.get('session_id') or hashlib.sha256((folder/'session.json').read_bytes()+folder.name.encode()).hexdigest()
    if store.has_visit(session):
        prior=folder/'map-review.json'
        if prior.exists():
            saved=json.loads(prior.read_text())
            if saved.get('status')=='complete':report=saved;report['already_processed']=True;return finish()
        report.update(status='already_processed',reason='same session cannot count as a repeated drive');return finish()
    mount=Mount(**cfg['mount']);ground=Ground();anchors=store.anchors();new_anchors={};counts={};cover={};absent={};keyframes=[]
    previous=None;last_t=0;name=None;R=np.eye(3);t=np.zeros(3);motion_exclusions={}
    path=folder/'decisions.jsonl'
    if path.exists():
        with path.open() as f:
            for line in f:
                try:r=json.loads(line)
                except ValueError:continue
                boxes=[tr['bounds_m'] for tr in r.get('tracks',[]) if 'bounds_m' in tr and (np.linalg.norm(tr.get('velocity',[0,0]))>1 or tr.get('association_ambiguous'))]
                if boxes:motion_exclusions[r['at_ns']]=boxes
    for raw,stamps,ctx,q in frames(folder,cancel):
        check_work(cancel)
        now=ctx['at_ns'];dt=(now-last_t)/1e9
        if dt<1.:continue
        if not(ctx.get('motion_ready') and q.get('timestamp_valid') and q.get('completeness',0)>=.98 and q.get('timing_bound_s',1)<=.05):previous=None;continue
        p=transform(raw,mount);g=ground.fit(p)
        if g is None:previous=None;continue
        heights=p[:,2]-(p[:,0]*g['abc'][0]+p[:,1]*g['abc'][1]+g['abc'][2])
        usable=p[(heights>.4)&(heights<5)]
        for box in motion_exclusions.get(now,[]):usable=usable[~((usable>=np.array(box['lo'])-.3)&(usable<=np.array(box['hi'])+.3)).all(1)]
        feat=features(usable)
        if len(feat)<80:previous=None;continue
        report['eligible_keyframes']+=1
        if previous is None or dt>2.5:
            fit=bootstrap(feat,anchors)
            if fit:name=fit['submap'];R,t=fit['R'],fit['t'];report['localized_keyframes']+=1
            else:
                name=session[:12]+'-'+str(now);R=np.eye(3);t=np.zeros(3);new_anchors[name]=feat;report['new_submaps']+=1
        else:
            movement,angle=ego_delta(ctx['speed_mps'],ctx['yaw_rate_rps'],dt)
            seed_R=R@rotation_z(angle);seed_t=t+R@np.r_[movement,0.]
            fit=icp(feat,previous,(seed_R,seed_t))
            if fit is None or np.linalg.norm(fit['t']-seed_t)>1.5 or np.linalg.norm(fit['R']-seed_R)>.15:
                previous=None;last_t=now;continue
            R,t=fit['R'],fit['t']
            if name not in new_anchors:report['localized_keyframes']+=1
        world=feat@R.T+t;previous=world;last_t=now
        keys=set(map(tuple,np.floor(world/VOXEL).astype(int)));table=counts.setdefault(name,{})
        for key in keys:
            item=table.setdefault(key,[0,now,now,t.copy()]);item[0]+=1;item[2]=now
        # Recorded rays establish coverage; no-return areas are never marked clear.
        origin=np.asarray(mount.xyz)
        rays=(origin+(p[::max(1,len(p)//800),None,:]-origin)*np.linspace(.1,1.,20)[None,:,None]).reshape(-1,3)
        rays=rays@R.T+t;cover.setdefault(name,set()).update(map(tuple,np.floor(rays/COVERAGE).astype(int)))
        existing=store.snapshot(name)['points']
        absent.setdefault(name,set()).update(visible_absences(existing,R,t,p,keys,origin))
        if report['eligible_keyframes']%15==1:keyframes.append((session+'-'+str(now),name,world))
    observations={name:{key for key,v in table.items() if v[0]>=3 and v[2]-v[1]>=2_000_000_000} for name,table in counts.items()}
    if not any(observations.values()):report['reason']='no confidently registered repeated observations';return finish()
    absent={name:keys-set(counts.get(name,{})) for name,keys in absent.items()}
    check_work(cancel)
    store.commit(session,observations,new_anchors,cover,absent,keyframes=keyframes,cancel=cancel)
    snapshots=[store.snapshot(name) for name in observations]
    report.update(status='complete',coverage_percent=100*report['localized_keyframes']/max(1,report['eligible_keyframes']),
                  promoted_static_voxels=sum(len(s['points']) for s in snapshots),pending_voxels=sum(s['pending'] for s in snapshots),
                  forgotten_voxels=sum(s['forgotten'] for s in snapshots),submaps=len(observations),
                  reason='Static evidence needs at least two distinct drives, three registered observations per drive; disappearance needs three clear-view visits or 90-day expiry.')
    return finish()

def in_corridor(track,context,mount):
    path=np.asarray(context.get('path',[]),float)
    if path.ndim!=2 or len(path)<2 or not np.isfinite(path).all():return True
    p=np.asarray(track['position']);delta=path[1:]-path[:-1];length=np.sum(delta**2,axis=1)
    fraction=np.clip(np.sum((p-path[:-1])*delta,axis=1)/np.maximum(length,1e-9),0,1)
    distance=float(np.min(np.linalg.norm(p-(path[:-1]+fraction[:,None]*delta),axis=1)))
    bounds=track.get('bounds_m',{});radius=.5
    if 'lo' in bounds and 'hi' in bounds:radius=max(radius,float(np.linalg.norm(np.array(bounds['hi'][:2])-bounds['lo'][:2])/2))
    return distance<=mount.width_m/2+.25+radius or p[0]>np.max(path[:,0])

class MapPrior:
    """Background localization; stale/ambiguous results produce no map evidence."""
    def __init__(self,root):
        self.store=MapStore(root);self.anchors=self.store.anchors();self.pending=queue.Queue(maxsize=1)
        self.stop_event=threading.Event();self.result=None;self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();self.last_offer=0
    def offer(self,points,context):
        if not self.anchors or not context.get('motion_ready') or context['at_ns']-self.last_offer<500_000_000:return
        self.last_offer=context['at_ns'];item=(np.array(points,copy=True),{k:context[k] for k in ('at_ns','speed_mps','yaw_rate_rps')})
        try:self.pending.put_nowait(item)
        except queue.Full:pass
    def _run(self):
        while not self.stop_event.is_set():
            try:p,ctx=self.pending.get(timeout=.2)
            except queue.Empty:continue
            try:
                p=features(p)
                prior=self.result;fit=None
                if prior and 0<(ctx['at_ns']-prior['at_ns'])/1e9<3:
                    dt=(ctx['at_ns']-prior['at_ns'])/1e9;move,angle=ego_delta(ctx['speed_mps'],ctx['yaw_rate_rps'],dt)
                    seed_t=prior['t']+prior['R']@np.r_[move,0.]
                    reference=prior['reference'];reference=reference[np.linalg.norm(reference[:,:2]-seed_t[:2],axis=1)<80]
                    fit=icp(p,reference,(prior['R']@rotation_z(angle),seed_t))
                    if fit:fit['submap']=prior['submap']
                if fit is None:fit=bootstrap(p,self.anchors)
                if fit:
                    snap=self.store.snapshot(fit['submap']);reference=snap['points'] if len(snap['points'])>=80 else next(a['points'] for a in self.anchors if a['submap']==fit['submap'])
                    self.result={**fit,**ctx,**snap,'tree':cKDTree(snap['points']) if len(snap['points']) else None,'reference':reference}
                else:self.result=None
            except Exception:self.result=None
    def annotate(self,result,context,mount):
        prior=self.result;dt=(context['at_ns']-prior['at_ns'])/1e9 if prior else math.inf
        valid=bool(prior and context.get('motion_ready') and 0<=dt<=.75 and abs(context['speed_mps']-prior['speed_mps'])<2 and abs(context['yaw_rate_rps']-prior['yaw_rate_rps'])<.1)
        result['map_status']='localized_advisory' if valid else 'unknown'
        if not valid:return
        move,angle=ego_delta(context['speed_mps'],context['yaw_rate_rps'],dt);R=prior['R']@rotation_z(angle);t=prior['t']+prior['R']@np.r_[move,0.]
        for track in result.get('tracks',[]):
            bounds=track.get('bounds_m',{});z=float(np.mean([bounds.get('lo',[0,0,1])[2],bounds.get('hi',[0,0,1])[2]]))
            world=R@np.r_[track['position'],z]+t;covered=tuple(np.floor(world/COVERAGE).astype(int)) in prior['coverage']
            dist=float(prior['tree'].query(world)[0]) if prior['tree'] is not None else math.inf
            inside=in_corridor(track,context,mount)
            kind='persistent_static_structure' if dist<.8 else 'new_relative_to_map' if covered else 'unknown'
            track['map_evidence']={'kind':kind,'in_corridor_or_unknown':inside,
                                   'priority':'lower_roadside' if kind=='persistent_static_structure' and not inside else 'raised_evidence' if kind=='new_relative_to_map' else 'unchanged',
                                   'warning_override':False}
        by_id={tr['id']:tr.get('map_evidence') for tr in result.get('tracks',[])}
        for a in result.get('candidates',[])+result.get('alerts',[]):
            if by_id.get(a['track_id']):a['map_evidence']={**by_id[a['track_id']],'priority':'unchanged' if by_id[a['track_id']]['kind']=='persistent_static_structure' else by_id[a['track_id']]['priority']}
    def close(self):self.stop_event.set();self.thread.join(timeout=2)

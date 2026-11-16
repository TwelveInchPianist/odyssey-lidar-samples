"""Background localization with immutable, nonblocking FCW map-prior snapshots.

The predecision API supplies bounded advisory evidence; it never removes a
candidate, cancels an alert, or allows mapped structure to override a corridor.
"""
import math, queue, threading
import runtime_clock as time
from collections import deque
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from fcw_engine import Ground,ego_delta
from route_map import features,rotation_z,in_corridor
from .store import AtlasStore
from .places import PlaceMatcher,SequenceRecognizer
from .static_map import StaticMap,COVERAGE
from .graph import inverse
from .backends import BackendProfile

MAX_POSE_AGE_S=.75
MAX_LOCAL_POINTS=250_000

def _epoch(value):
    if isinstance(value,dict):value=value.get('id')
    return value if isinstance(value,str) and value.strip() else None

@dataclass(frozen=True)
class MapPriorSnapshot:
    """One decision's immutable view. Query performs no filesystem/network work.

    Position/yaw bounds are conservative geometric consistency envelopes, not
    statistically calibrated confidence intervals. Unknown evidence has zero
    score and may never be interpreted as an empty driving corridor.
    """
    status:str='unknown'
    reason:str='no_confirmed_snapshot'
    revision:str|None=None
    component:str|None=None
    at_ns:int|None=None
    pose_age_s:float|None=None
    mount_epoch:str|None=None
    calibration_id:str|None=None
    position_bound_m:float|None=None
    yaw_bound_rad:float|None=None
    confirmations:int=0
    _transform:object=field(default=None,repr=False,compare=False)
    _tree:object=field(default=None,repr=False,compare=False)
    _coverage:frozenset=field(default_factory=frozenset,repr=False)
    _origin:object=field(default=None,repr=False,compare=False)
    _clock:object=field(default=time.monotonic,repr=False,compare=False)
    _expires_monotonic:float=field(default=0.,repr=False,compare=False)

    def audit(self):
        expired=self.status=='available' and self._clock()>self._expires_monotonic
        return {'schema_version':1,'status':'unknown' if expired else self.status,
                'reason':'snapshot_expired' if expired else self.reason,'revision':self.revision,
                'component':self.component,'at_ns':self.at_ns,'pose_age_s':self.pose_age_s,
                'mount_epoch':self.mount_epoch,'calibration_id':self.calibration_id,
                'position_bound_m':self.position_bound_m,'yaw_bound_rad':self.yaw_bound_rad,
                'uncertainty_kind':'geometric_envelope_not_calibrated_probability',
                'confirmations':self.confirmations,'coverage_cells':len(self._coverage),
                'warning_override':False,'stage':'predecision'}

    def query(self,track,in_corridor_or_unknown=True):
        result={'kind':'unknown','priority':'unchanged','evidence_delta':0.,
                'in_corridor_or_unknown':in_corridor_or_unknown is not False,
                'revision':self.revision,'coverage':False,'warning_override':False,
                'stage':'predecision','reason':self.reason}
        if self.status!='available':return result
        if self._clock()>self._expires_monotonic:
            result['reason']='snapshot_expired';return result
        try:
            xy=np.asarray(track['position'],float)
            bounds=track.get('bounds_m',{})
            z=float(np.mean([bounds.get('lo',[0,0,1])[2],bounds.get('hi',[0,0,1])[2]]))
            if xy.shape!=(2,) or not np.isfinite(np.r_[xy,z]).all():raise ValueError('invalid position')
            local=np.r_[xy,z];world=self._transform[:3,:3]@local+self._transform[:3,3]
            envelope=self.position_bound_m+np.linalg.norm(local)*math.sin(self.yaw_bound_rad)
            result['position_envelope_m']=float(envelope)
            if envelope>.8:
                result['reason']='position_uncertainty_too_large';return result
            if np.linalg.norm(world-self._origin)+envelope>80:
                result['reason']='outside_local_snapshot';return result
            # Demand covered space over the whole uncertainty envelope, rather
            # than declaring an edge-of-coverage location "new".
            corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
            covered=all(tuple(cell) in self._coverage for cell in np.floor((world+corners*envelope)/COVERAGE).astype(int))
            result['coverage']=covered
            distance=float(self._tree.query(world)[0]) if self._tree is not None else math.inf
            result['nearest_static_m']=distance if math.isfinite(distance) else None
            # Absence of the entire local map is not evidence of clear space.
            if self._tree is None or not covered:
                result['reason']='insufficient_coverage';return result
            if distance+envelope<.8:kind='persistent_static_structure'
            elif distance-envelope>.8:kind='new_relative_to_map'
            else:
                result['reason']='map_distance_ambiguous';return result
            result.update(kind=kind,reason='confirmed_own_map_geometry')
            if in_corridor_or_unknown is False:
                result.update(priority='lower_roadside' if kind=='persistent_static_structure' else 'raised_evidence',
                              evidence_delta=-.1 if kind=='persistent_static_structure' else .1)
            return result
        except (ValueError,TypeError,KeyError,IndexError,OverflowError):
            result['reason']='invalid_track_position';return result

class AtlasPrior:
    def __init__(self,root,profile=None):
        self.store=AtlasStore(Path(root)/'atlas-v2');self.profile=profile or BackendProfile();self.maps=self.profile.map_factory(self.store);self.matcher=PlaceMatcher(self.store,index=self.profile.index_factory(self.store),registration=self.profile.registration_factory())
        self.recognizer=SequenceRecognizer();self.lock=threading.RLock();self.pending=queue.Queue(maxsize=1)
        self.stop_event=threading.Event();self.last_frame=None;self.last_offer=None;self.odom=np.eye(4);self.segment=0
        self.result=None;self.revision=None;self.tracking=None;self.last_global=None;self.last_state='waiting_for_confirmed_map';self.last_processed_ns=None
        self.lineage=(None,None);self.alignment_history=deque(maxlen=8)
        self.verified_product=None;self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def offer(self,points,context,quality=None,tracks=None):
        quality=quality or {};at=context['at_ns']
        with self.lock:
            lineage=(_epoch(context.get('mount_epoch')),_epoch(context.get('calibration_id')))
            dt=(at-self.last_frame)/1e9 if self.last_frame is not None else None
            reliable=bool(context.get('motion_ready') and quality.get('timestamp_valid') is True and quality.get('completeness',0)>=.98 and quality.get('timing_bound_s',1)<=.05)
            if not reliable or dt is None or not 0<dt<=.25 or lineage!=self.lineage:
                self.segment+=1;self.odom=np.eye(4);self.result=None;self.last_offer=None
            elif reliable:
                movement,angle=ego_delta(context['speed_mps'],context['yaw_rate_rps'],dt)
                self.odom[:3,3]+=self.odom[:3,:3]@np.r_[movement,0.];self.odom[:3,:3]=self.odom[:3,:3]@rotation_z(angle)
            self.last_frame=at;self.lineage=lineage
            if not reliable:return
            if self.last_offer is not None and at-self.last_offer<500_000_000:return
            self.last_offer=at
            boxes=[t['bounds_m'] for t in tracks or [] if 'bounds_m' in t and (t.get('association_ambiguous') or t.get('hits',3)<3 or np.linalg.norm(t.get('velocity',[0,0]))>1)]
            item=(np.array(points,copy=True),dict(context),self.odom.copy(),self.segment,boxes)
        try:self.pending.put_nowait(item)
        except queue.Full:
            try:self.pending.get_nowait()
            except queue.Empty:pass
            try:self.pending.put_nowait(item)
            except queue.Full:pass
    def _run(self):
        extractor=self.profile.feature_factory()
        while not self.stop_event.is_set():
            try:points,ctx,odom,segment,boxes=self.pending.get(timeout=.2)
            except queue.Empty:continue
            try:
                revision=self.store.current_revision()
                if not revision:
                    with self.lock:self.last_state='no_published_own_map';self.result=None;self.revision=None
                    continue
                if revision!=self.revision:
                    with self.lock:self.revision=revision;self.result=None
                    self.recognizer.reset();self.tracking=None;self.last_global=None;self.alignment_history.clear()
                    path=self.store.product(revision,'spatial_index');self.verified_product=(path,path.stat().st_size,path.stat().st_mtime_ns)
                expanded=[{'lo':np.asarray(b['lo'])-.3,'hi':np.asarray(b['hi'])+.3} for b in boxes]
                extracted=extractor.extract(points,expanded);p=extracted.get('points')
                if p is None:self.last_state='ground_support_unavailable';self.result=None;continue
                result=None
                if self.tracking and self.tracking['segment']==segment and self.last_global is not None and ctx['at_ns']-self.last_global<10_000_000_000:
                    result=self.matcher.match(p,ctx.get('location'),revision=revision,cancel=self.stop_event.is_set,
                                              map_from_query_seed=self.tracking['map_from_odom']@odom,seed_component=self.tracking['component'])
                if result is None or result['status']!='candidate':
                    result=self.matcher.match(p,ctx.get('location'),revision=revision,cancel=self.stop_event.is_set);self.last_global=ctx['at_ns']
                if result['status']=='candidate':
                    match=result['matches'][0];self.tracking={'map_from_odom':np.asarray(match['map_from_query'])@inverse(odom),'component':match['component'],'segment':segment}
                    if self.alignment_history and self.alignment_history[-1][0:2]!=(segment,match['component']):self.alignment_history.clear()
                    self.alignment_history.append((segment,match['component'],ctx['at_ns'],self.tracking['map_from_odom'].copy()))
                else:self.tracking=None
                confirmed=self.recognizer.observe(ctx['at_ns'],str(segment),odom,result);self.last_state=confirmed['status']
                if confirmed['status']!='confirmed':
                    with self.lock:self.result=None
                    if confirmed['status'] in ('conflict','ambiguous','unknown'):self.alignment_history.clear()
                    continue
                if self.verified_product:
                    path,size,mtime=self.verified_product
                    if (path.stat().st_size,path.stat().st_mtime_ns)!=(size,mtime):
                        self.store.product(revision,'spatial_index');self.verified_product=(path,path.stat().st_size,path.stat().st_mtime_ns)
                T=np.asarray(confirmed['map_from_odom'])@odom
                local=self.maps.local_snapshot(revision,confirmed['component'],T[:3,3],verify=False)
                # All I/O and tree construction remain in this worker. Bound
                # memory/query cost rather than silently sampling dense maps.
                if len(local['points'])>MAX_LOCAL_POINTS:
                    with self.lock:self.result=None;self.last_state='local_map_budget_exceeded'
                    continue
                align=np.asarray(confirmed['map_from_odom'])
                offsets=[np.linalg.norm(item[3][:3,3]-align[:3,3]) for item in self.alignment_history]
                angles=[math.acos(float(np.clip((np.trace(item[3][:3,:3].T@align[:3,:3])-1)/2,-1,1))) for item in self.alignment_history]
                position_bound=.15+3*confirmed['match']['rmse_m']+max(offsets,default=math.inf)
                yaw_bound=.005+max(angles,default=math.inf)
                tree=cKDTree(local['points'],copy_data=True) if len(local['points']) else None
                if tree is not None:tree.data.setflags(write=False)
                coverage=frozenset(local['coverage'])
                with self.lock:
                    if segment!=self.segment:continue
                    self.result={**confirmed,'at_ns':ctx['at_ns'],'segment':segment,'revision':revision,'coverage':coverage,
                                 'tree':tree,'speed_mps':ctx['speed_mps'],'yaw_rate_rps':ctx['yaw_rate_rps'],
                                 'produced_monotonic':time.monotonic(),'local_origin':T[:3,3].copy(),
                                 'mount_epoch':_epoch(ctx.get('mount_epoch')),'calibration_id':_epoch(ctx.get('calibration_id')),
                                 'calibration_verified':ctx.get('calibration_verified') is True,
                                 'position_bound_m':position_bound,'yaw_bound_rad':yaw_bound}
            except Exception as e:
                with self.lock:self.result=None;self.last_state='unavailable: '+type(e).__name__
            finally:self.last_processed_ns=ctx['at_ns']

    def snapshot(self,context,mount_epoch=None,calibration_id=None):
        """Pin the last confirmed map before FCW; no worker wait or storage I/O.

        Caller supplies verified active calibration/mount identity. Old API
        callers can still obtain diagnostic annotations but cannot enable priors.
        """
        with self.lock:
            prior=self.result;odom=self.odom.copy();segment=self.segment
            revision=self.revision;state=self.last_state;last_frame=self.last_frame
        def unknown(reason):return MapPriorSnapshot(reason=reason,revision=revision)
        if self.stop_event.is_set():return unknown('map_disconnected')
        if not prior:return unknown(state)
        if prior.get('status')!='confirmed' or state!='confirmed':return unknown('localization_not_confirmed')
        if prior['revision']!=revision:return unknown('revision_changed')
        if prior['segment']!=segment:return unknown('motion_segment_changed')
        epoch=_epoch(mount_epoch if mount_epoch is not None else context.get('mount_epoch'))
        calibration=_epoch(calibration_id if calibration_id is not None else context.get('calibration_id'))
        if not epoch or not calibration:return unknown('mount_or_calibration_unknown')
        if (epoch,calibration)!=(prior.get('mount_epoch'),prior.get('calibration_id')):return unknown('mount_or_calibration_changed')
        if context.get('calibration_verified') is not True or prior.get('calibration_verified') is not True:return unknown('calibration_unverified')
        if not context.get('motion_ready'):return unknown('motion_unavailable')
        try:
            age=(context['at_ns']-prior['at_ns'])/1e9;now=time.monotonic()
            wall_age=now-prior['produced_monotonic'];dt=(context['at_ns']-last_frame)/1e9
            speed=float(context['speed_mps']);yaw=float(context['yaw_rate_rps'])
            if not all(math.isfinite(x) for x in (age,wall_age,dt,speed,yaw)):return unknown('nonfinite_context')
            if not 0<=age<=MAX_POSE_AGE_S or not 0<=wall_age<=MAX_POSE_AGE_S:return unknown('pose_stale_or_clock_discontinuous')
            if not 0<=dt<=.25:return unknown('motion_gap_or_clock_discontinuous')
            if abs(speed-prior['speed_mps'])>=2 or abs(yaw-prior['yaw_rate_rps'])>=.1:return unknown('motion_inconsistent')
            movement,angle=ego_delta(speed,yaw,dt)
            odom[:3,3]+=odom[:3,:3]@np.r_[movement,0.];odom[:3,:3]=odom[:3,:3]@rotation_z(angle)
            transform=np.asarray(prior['map_from_odom'])@odom;transform.setflags(write=False)
            origin=np.asarray(prior['local_origin']).copy();origin.setflags(write=False)
            position=float(prior['position_bound_m'])+.2*age
            yaw_bound=float(prior['yaw_bound_rad'])+.01*age
            if not math.isfinite(position) or not math.isfinite(yaw_bound) or position<0 or not 0<=yaw_bound<=.1:return unknown('pose_uncertainty_unbounded')
            return MapPriorSnapshot(status='available',reason='confirmed_own_map_geometry',revision=revision,
                component=prior['component'],at_ns=context['at_ns'],pose_age_s=age,mount_epoch=epoch,
                calibration_id=calibration,position_bound_m=position,yaw_bound_rad=yaw_bound,
                confirmations=prior['confirmations'],_transform=transform,_tree=prior['tree'],
                _coverage=frozenset(prior['coverage']),_origin=origin,
                _expires_monotonic=now+min(MAX_POSE_AGE_S-age,MAX_POSE_AGE_S-wall_age))
        except (ValueError,TypeError,KeyError,OverflowError):return unknown('invalid_snapshot_context')
    def annotate(self,result,context,mount):
        with self.lock:prior=self.result;odom=self.odom.copy();segment=self.segment
        age=(context['at_ns']-prior['at_ns'])/1e9 if prior else math.inf
        valid=bool(prior and context.get('motion_ready') and prior['segment']==segment and 0<=age<=.75 and
                   abs(context['speed_mps']-prior['speed_mps'])<2 and abs(context['yaw_rate_rps']-prior['yaw_rate_rps'])<.1)
        result['map_status']='localized_advisory' if valid else 'unknown';result['map_localization']={'state':self.last_state,'revision':self.revision,'warning_override':False}
        if not valid:return
        T=np.asarray(prior['map_from_odom'])@odom
        result['map_localization'].update(component=prior['component'],confirmations=prior['confirmations'],pose_age_s=age,
                                          rmse_m=prior['match']['rmse_m'],uncertainty=prior['uncertainty'])
        for track in result.get('tracks',[]):
            bounds=track.get('bounds_m',{});z=float(np.mean([bounds.get('lo',[0,0,1])[2],bounds.get('hi',[0,0,1])[2]]))
            world=T[:3,:3]@np.r_[track['position'],z]+T[:3,3];covered=tuple(np.floor(world/COVERAGE).astype(int)) in prior['coverage']
            distance=float(prior['tree'].query(world)[0]) if prior['tree'] is not None else math.inf;inside=in_corridor(track,context,mount)
            kind='persistent_static_structure' if distance<.8 else 'new_relative_to_map' if covered else 'unknown'
            track['map_evidence']={'kind':kind,'in_corridor_or_unknown':inside,'revision':prior['revision'],
                                   'priority':'lower_roadside' if kind=='persistent_static_structure' and not inside else 'raised_evidence' if kind=='new_relative_to_map' else 'unchanged','warning_override':False}
        by_id={t['id']:t.get('map_evidence') for t in result.get('tracks',[])}
        for candidate in result.get('candidates',[])+result.get('alerts',[]):
            if by_id.get(candidate['track_id']):
                candidate['map_evidence']={**by_id[candidate['track_id']],'priority':'unchanged' if by_id[candidate['track_id']]['kind']=='persistent_static_structure' else by_id[candidate['track_id']]['priority']}
    def close(self):self.stop_event.set();self.thread.join(timeout=2)

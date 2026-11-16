"""Persistent place retrieval plus conservative multi-frame confirmation.

The radial/height fingerprint is a shortlist only. Partial-FOV M1 views still
require geometric overlap; no 360-degree or opposite-direction guarantee is made.
Uncertain alternatives remain separate hypotheses, never silently merged.
"""
from dataclasses import asdict
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from route_map import icp,downsample,rotation_z
from .contracts import checked_pose
from .geodesy import horizontal_distance

DESCRIPTOR_VERSION='radial-height-partial-view-v1'

def descriptor(points):
    p=np.asarray(points,float)[:,:3];p=p[np.isfinite(p).all(1)]
    radius=np.linalg.norm(p[:,:2],axis=1);keep=(radius>=2)&(radius<70)&(p[:,2]>.3)&(p[:,2]<6)
    p=p[keep];radius=radius[keep]
    if len(p)<80:return None
    # Independent implementation of a simple radial/height histogram; no neural
    # embedding or third-party descriptor code. Rotation-invariant retrieval
    # intentionally cannot resolve orientation or repeated-looking structures.
    rings=np.minimum(11,(radius/70*12).astype(int));height=np.minimum(3,(p[:,2]/6*4).astype(int))
    counts=np.zeros((12,4));np.add.at(counts,(rings,height),1)
    vector=np.sqrt(counts.reshape(-1)/len(p))
    return vector.astype(np.float32)

def angle_between(A,B):return float(Rotation.from_matrix(np.asarray(A)[:3,:3].T@np.asarray(B)[:3,:3]).magnitude())

def transform_matrix(R,t):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T

class DescriptorIndex:
    version=DESCRIPTOR_VERSION
    def __init__(self,store):self.store=store;self.generation=None;self.ids=[];self.sessions=[];self.tree=None
    def add(self,keyframe,cloud):
        vector=descriptor(cloud)
        if vector is not None:self.store.add_descriptor(keyframe.id,DESCRIPTOR_VERSION,vector)
        return vector is not None
    def refresh(self):
        generation=self.store.descriptor_generation()
        if generation==self.generation:return
        rows=list(self.store.descriptors(DESCRIPTOR_VERSION));self.ids=[r[0] for r in rows];self.sessions=[r[1] for r in rows]
        self.tree=cKDTree(np.asarray([r[2] for r in rows])) if rows else None;self.generation=generation
    def candidates(self,cloud,location=None,exclude_session=None,limit=24,exclude_ids=None,allowed_ids=None):
        self.refresh();vector=descriptor(cloud)
        if vector is None or self.tree is None:return []
        exclude_ids=set(exclude_ids or ())
        allowed_ids=None if allowed_ids is None else set(allowed_ids)
        # Include geographic candidates from all recorded history, not only the
        # newest keyframes. Descriptor candidates provide a GPS-outage fallback.
        geographic=[]
        if location and (location.get('usable_for_fusion') or location.get('usable_for_index')):
            p=location['position'];radius=150 if location.get('usable_for_fusion') else 500
            geographic=self.store.nearby(p['latitude_deg'],p['longitude_deg'],radius,exclude_session,limit=limit)
        count=min(len(self.ids),max(limit*4,64));ranked=[]
        while True:
            distances,indices=self.tree.query(vector,k=count)
            ranked=[{'id':self.ids[int(i)],'descriptor_distance':float(d),'candidate_source':'geometry'} for d,i in zip(np.atleast_1d(distances),np.atleast_1d(indices)) if self.sessions[int(i)]!=exclude_session and self.ids[int(i)] not in exclude_ids and (allowed_ids is None or self.ids[int(i)] in allowed_ids)]
            if len(ranked)>=limit or count==len(self.ids):break
            count=min(len(self.ids),count*2)
        selected={r['id']:{**r,'candidate_source':'geographic'} for r in geographic if r['id'] not in exclude_ids and (allowed_ids is None or r['id'] in allowed_ids)}
        for r in ranked[:limit]:
            if r['id'] in selected:selected[r['id']].update(descriptor_distance=r['descriptor_distance'])
            else:selected[r['id']]=r
        return list(selected.values())

class ICPRegistration:
    version='trimmed-icp-v2'
    max_points=3200
    def align(self,source,target,seed=None):
        # Keep the matcher's sampling budget consistent through registration
        # and reciprocal checking. Re-subsampling a previously thinned source
        # against a denser reference can hide valid alternative alignments.
        source=downsample(source,limit=self.max_points);target=downsample(target,limit=self.max_points)
        if min(len(source),len(target))<80:return None
        for p in (source,target):
            spread=np.linalg.svd(p-p.mean(0),compute_uv=False)
            if spread[-1]/max(spread[0],1e-9)<.025:return None
        R,t=(np.eye(3),np.zeros(3)) if seed is None else (seed[0].copy(),seed[1].copy())
        tree=cKDTree(target)
        for _ in range(16):
            distances,indices=tree.query(source@R.T+t);good=distances<=min(2.,max(.08,float(np.quantile(distances,.7))))
            if good.sum()<60:return None
            a,b=source[good],target[indices[good]];ac=a-a.mean(0);bc=b-b.mean(0)
            U,_,Vt=np.linalg.svd(ac.T@bc);new_R=Vt.T@U.T
            if np.linalg.det(new_R)<0:Vt[-1]*=-1;new_R=Vt.T@U.T
            new_t=b.mean(0)-a.mean(0)@new_R.T;change=np.linalg.norm(new_R-R)+np.linalg.norm(new_t-t);R,t=new_R,new_t
            if change<.0003:break
        distances,_=tree.query(source@R.T+t);inside=distances<.35;overlap=float(inside.mean())
        rms=float(np.sqrt(np.mean(distances[inside]**2))) if inside.any() else math.inf
        if overlap<.65 or rms>.22:return None
        return {'R':R,'t':t,'rmse_m':rms,'overlap':overlap}

class PlaceMatcher:
    def __init__(self,store,index=None,registration=None):
        self.store=store;self.index=index or DescriptorIndex(store);self.registration=registration or ICPRegistration()
        self._pose_revision=None;self._poses={}
    def match(self,cloud,location=None,exclude_session=None,limit=24,cancel=None,exclude_ids=None,revision=None,map_from_query_seed=None,seed_component=None):
        budget=getattr(self.registration,'max_points',1600)
        query=downsample(cloud,limit=budget);revision=revision or self.store.current_revision()
        if revision!=self._pose_revision:self._poses=self.store.revision_poses(revision);self._pose_revision=revision
        poses=self._poses;matches=[];rejected=[]
        if map_from_query_seed is None:
            candidates=self.index.candidates(query,location,exclude_session,limit,exclude_ids,set(poses))
        else:
            predicted=checked_pose(map_from_query_seed);excluded=set(exclude_ids or ())
            nearby=sorted((np.linalg.norm(np.asarray(p['map_from_body'])[:3,3]-predicted[:3,3]),key) for key,p in poses.items() if p['component']==seed_component and key not in excluded)
            candidates=[{'id':key,'candidate_source':'tracked_local_geometry'} for distance,key in nearby[:4] if distance<30 and self.store.keyframe(key).session_id!=exclude_session]
        for candidate in candidates:
            if cancel and cancel():
                from offline_jobs import WorkPaused
                raise WorkPaused()
            key=candidate['id'];reference=self.store.keyframe(key)
            if key not in poses:continue
            geographic_conflict=False
            if location and location.get('usable_for_fusion') and reference.location and reference.location.get('usable_for_fusion'):
                a,b=location['position'],reference.location['position'];distance=horizontal_distance((a['latitude_deg'],a['longitude_deg']),(b['latitude_deg'],b['longitude_deg']))
                margin=80+3*(location['horizontal_accuracy_m']+reference.location['horizontal_accuracy_m'])
                if distance>margin:
                    geographic_conflict=True;rejected.append({'reference':key,'reason':'geographic_conflict_requires_independent_check','distance_m':distance})
            target=downsample(self.store.load_cloud(reference.cloud_hash),limit=budget);fits=[]
            if map_from_query_seed is None:
                seeds=[]
                for angle in np.arange(0,2*math.pi,math.pi/6):
                    R=rotation_z(angle);seeds.append((R,target.mean(0)-query.mean(0)@R.T))
            else:
                predicted_relative=np.linalg.inv(checked_pose(poses[key]['map_from_body']))@predicted
                seeds=[(predicted_relative[:3,:3],predicted_relative[:3,3])]
            for R,t in seeds:
                fit=self.registration.align(query,target,(R,t))
                if not fit:continue
                # Test the reciprocal overlap only in the part of the reference
                # observable in this forward-facing view, rather than forcing
                # agreement with behind-the-van or out-of-FOV map points.
                in_query=(target-fit['t'])@fit['R']
                visible=(in_query[:,0]>2)&(in_query[:,0]<70)&(np.abs(np.arctan2(in_query[:,1],in_query[:,0]))<math.radians(60))
                if visible.sum()<60:continue
                reciprocal=float((cKDTree(query).query(in_query[visible])[0]<.4).mean())
                if reciprocal<.6:continue
                T=checked_pose(transform_matrix(fit['R'],fit['t']));world=checked_pose(poses[key]['map_from_body'])@T
                if map_from_query_seed is not None and (np.linalg.norm(world[:3,3]-predicted[:3,3])>1.5 or angle_between(world,predicted)>.12):continue
                score=fit['rmse_m']+.25*(1-fit['overlap'])+.2*(1-reciprocal)
                result={'reference_id':key,'component':poses[key]['component'],'reference_from_query':T.tolist(),'map_from_query':world.tolist(),
                        'score':score,'rmse_m':fit['rmse_m'],'overlap':fit['overlap'],'reciprocal_overlap':reciprocal,
                        'registration_backend':self.registration.version,'descriptor_version':self.index.version,
                        'geographic_conflict':geographic_conflict,
                        'map_revision':revision,'candidate_source':candidate['candidate_source']}
                fits.append(result)
            if fits:matches.extend(fits)
            else:rejected.append({'reference':key,'reason':'insufficient_geometric_overlap_or_degenerate'})
        # Many neighboring/reference snapshots may converge on one actual place.
        # Collapse equivalent solutions, retaining distinct locations as rivals.
        unique=[]
        for result in sorted(matches,key=lambda r:r['score']):
            T=np.asarray(result['map_from_query'])
            if any(r['component']==result['component'] and np.linalg.norm(np.asarray(r['map_from_query'])[:3,3]-T[:3,3])<1 and angle_between(r['map_from_query'],T)<math.radians(5) for r in unique):continue
            unique.append(result)
        ambiguous=len(unique)>1 and unique[1]['score']<=unique[0]['score']*1.3+.025
        conflict=bool(unique and unique[0].get('geographic_conflict'))
        return {'status':'ambiguous' if ambiguous else 'conflict' if conflict else 'candidate' if unique else 'unknown','matches':unique,'rejected':rejected,
                'search_mode':'global_retrieval' if map_from_query_seed is None else 'motion_seeded_tracking',
                'note':'A single fit is not a confirmed route match.'}

class SequenceRecognizer:
    """Confirm stable map-from-odometry alignment across separated observations."""
    def __init__(self):self.reset()
    def reset(self):self.last_at=None;self.segment=None;self.revision=None;self.hypotheses=[]
    def observe(self,at_ns,segment_id,odom_from_body,result):
        odom=checked_pose(odom_from_body)
        revision=result['matches'][0].get('map_revision') if result.get('matches') else None
        if self.last_at is not None and (at_ns<=self.last_at or at_ns-self.last_at>2_500_000_000 or self.segment!=segment_id or self.revision!=revision):self.reset()
        self.last_at=at_ns;self.segment=segment_id;self.revision=revision
        next_hypotheses=[]
        for match in result.get('matches',[])[:8]:
            alignment=checked_pose(match['map_from_query'])@np.linalg.inv(odom)
            prior=next((h for h in self.hypotheses if h['component']==match['component'] and
                        np.linalg.norm(h['alignment'][:3,3]-alignment[:3,3])<1. and angle_between(h['alignment'],alignment)<math.radians(5)),None)
            h={'component':match['component'],'alignment':alignment,'count':1,'start_at':at_ns,'start_position':odom[:3,3].copy(),'matches':[match]}
            if prior:h.update(count=prior['count']+1,start_at=prior['start_at'],start_position=prior['start_position'],matches=(prior['matches']+[match])[-8:])
            next_hypotheses.append(h)
        self.hypotheses=next_hypotheses
        qualified=[h for h in self.hypotheses if h['count']>=3 and at_ns-h['start_at']>=1_000_000_000 and np.linalg.norm(odom[:3,3]-h['start_position'])>=2.]
        if len(qualified)==1 and result.get('status')=='candidate':
            h=qualified[0]
            return {'status':'confirmed','component':h['component'],'map_from_odom':h['alignment'].tolist(),'confirmations':h['count'],
                    'matched_references':[r['reference_id'] for r in h['matches']],'match':h['matches'][-1],
                    'uncertainty':'geometric consistency checks; not a calibrated probability'}
        return {'status':'conflict' if result.get('status')=='conflict' else 'ambiguous' if len(qualified)>1 or result.get('status')=='ambiguous' else 'pending' if self.hypotheses else 'unknown',
                'confirmations':max((h['count'] for h in self.hypotheses),default=0),'reason':'Need a unique consistent match across at least three observations and 2 m of motion.'}

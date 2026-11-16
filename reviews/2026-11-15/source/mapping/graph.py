"""Replaceable SE(3) pose-graph backend over immutable observations.

Measured edges, GNSS hints and corrected poses stay separate. Disconnected
components remain separate. GNSS provides coarse horizontal anchoring only;
receiver accuracy is not a guarantee and the antenna lever arm is unsurveyed.
"""
from collections import defaultdict
import math,heapq,itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from offline_jobs import check_work
from .contracts import checked_pose,checked_uncertainty,identity
from .geodesy import geodetic_to_ecef,ecef_to_enu

VERSION='robust-se3-graph-v1'

def pack(T):return np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_rotvec()]
def unpack(v):
    T=np.eye(4);T[:3,3]=v[:3];T[:3,:3]=Rotation.from_rotvec(v[3:]).as_matrix();return T
def inverse(T):
    out=np.eye(4);out[:3,:3]=T[:3,:3].T;out[:3,3]=-T[:3,:3].T@T[:3,3];return out
def edge_residual(A,B,M):
    d=inverse(M)@inverse(A)@B
    return np.r_[d[:3,3],Rotation.from_matrix(d[:3,:3]).as_rotvec()]

def checked_edges(frames,edges,cancel=None):
    accepted=[];rejected=[];groups=defaultdict(list)
    for edge in edges:
        check_work(cancel)
        if edge.get('state','accepted')!='accepted':continue
        try:
            a,b=frames[edge['source_id']],frames[edge['target_id']]
            checked_pose(edge['source_from_target']);checked_uncertainty(edge['std'])
            if a.id==b.id or edge['kind'] not in ('odometry','scan_registration','revisit'):raise ValueError('invalid edge identity/type')
            if edge['kind']!='revisit' and (a.session_id!=b.session_id or a.segment_id!=b.segment_id):raise ValueError('ordinary motion edge crosses a segment/session gap')
            if edge['kind']=='revisit':
                p=edge.get('provenance',{})
                if not p.get('confirmation_group') or p.get('confirmations',0)<3 or p.get('baseline_m',0)<2 or p.get('overlap',0)<.65 or p.get('reciprocal_overlap',0)<.6 or p.get('rmse_m',99)>.22:
                    raise ValueError('revisit lacks separated multi-view geometric evidence')
                groups[p['confirmation_group']].append(edge)
            else:accepted.append(edge)
        except (KeyError,TypeError,ValueError) as e:rejected.append({'id':edge.get('id'),'reason':str(e)})
    for group,rows in groups.items():
        check_work(cancel)
        if len(rows)>32:
            rejected.extend({'id':r['id'],'reason':'revisit group exceeds bounded confirmation window'} for r in rows);continue
        targets={r['target_id'] for r in rows}
        points=np.array([checked_pose(frames[k].odom_from_body)[:3,3] for k in targets])
        separated=len(targets)>=3 and np.max(np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2))>=2
        stamps=[frames[k].at_ns for k in targets]
        one_segment=len({(frames[k].session_id,frames[k].segment_id) for k in targets})==1
        if not separated or max(stamps)-min(stamps)<1_000_000_000 or not one_segment:
            rejected.extend({'id':r['id'],'reason':'revisit group does not contain three separated keyframes in one continuous segment'} for r in rows)
        else:accepted.extend(rows)
    # A loop inside an already connected motion chain has an independent check.
    # Its allowance grows with the chain's accumulated engineering uncertainty;
    # do not compare solely to the loop's own optimistic fit residual.
    base_graph=defaultdict(list)
    for e in accepted:
        if e['kind']=='revisit':continue
        M=checked_pose(e['source_from_target']);variance=max(e['std'][:3])**2;rotation_variance=max(e['std'][3:])**2
        base_graph[e['source_id']].append((e['target_id'],M,variance,rotation_variance))
        base_graph[e['target_id']].append((e['source_id'],inverse(M),variance,rotation_variance))
    bad=set()
    base_components={}
    for start in base_graph:
        if start in base_components:continue
        pending=[start]
        while pending:
            node=pending.pop()
            if node in base_components:continue
            base_components[node]=start;pending.extend(v[0] for v in base_graph[node] if v[0] not in base_components)
    for e in accepted:
        if e['kind']!='revisit':continue
        if base_components.get(e['source_id'])!=base_components.get(e['target_id']):continue
        counter=itertools.count();queue=[(0.,next(counter),e['source_id'],np.eye(4),0.,0.)];seen=set()
        while queue:
            variance,_,node,T,rv,length=heapq.heappop(queue)
            if node in seen:continue
            seen.add(node)
            if len(seen)%256==0:check_work(cancel)
            if node==e['target_id']:
                r=edge_residual(np.eye(4),T,checked_pose(e['source_from_target']))
                if np.linalg.norm(r[:3])>max(3.,6*math.sqrt(variance)+.02*length) or np.linalg.norm(r[3:])>max(.15,6*math.sqrt(rv)+.001*length):bad.add(e['provenance']['confirmation_group'])
                break
            for nxt,M,v,rvar in base_graph[node]:
                if nxt not in seen:heapq.heappush(queue,(variance+v,next(counter),nxt,T@M,rv+rvar,length+np.linalg.norm(M[:3,3])))
    rejected.extend({'id':e['id'],'reason':'revisit disagrees with independent temporal chain'} for e in accepted if e.get('provenance',{}).get('confirmation_group') in bad)
    accepted=[e for e in accepted if e.get('provenance',{}).get('confirmation_group') not in bad]
    return accepted,rejected

def components(frames,edges):
    graph={key:[] for key in frames}
    for edge in edges:
        a,b=edge['source_id'],edge['target_id'];M=checked_pose(edge['source_from_target'])
        graph[a].append((b,M,edge['kind']));graph[b].append((a,inverse(M),edge['kind']))
    seen=set();result=[]
    for root in sorted(frames,key=lambda k:(frames[k].quality.get('capture_wall_ns',0),frames[k].session_id,frames[k].at_ns,k)):
        if root in seen:continue
        poses={};counter=itertools.count();queue=[(0,next(counter),root,np.eye(4))]
        while queue:
            _,_,node,T=heapq.heappop(queue)
            if node in seen:continue
            seen.add(node);poses[node]=T
            # Exhaust ordinary temporal connectivity before using a loop to
            # initialize another component. A bad shortcut must not seed a tear
            # in an otherwise consistent trajectory and appear self-consistent.
            for target,M,kind in graph[node]:
                if target in seen:continue
                heapq.heappush(queue,(1 if kind=='revisit' else 0,next(counter),target,T@M))
        result.append((root,poses))
    return result

def horizontal_anchors(frames,poses,location_priors=None):
    rows=[];used=set()
    for key in poses:
        frame=frames[key];fix=(location_priors or {}).get(key,frame.location) or {};p=fix.get('position') or {}
        if not fix.get('usable_for_fusion') or fix.get('horizontal_accuracy_m',999)>30:continue
        # A one-second-old position is not the location of a moving current
        # scan. Avoid silently treating last-known GNSS as simultaneous.
        if abs(frame.at_ns-fix.get('at_ns',-10**20))>200_000_000:continue
        token=(frame.session_id,fix.get('independence_group','comma_gnss'),fix.get('at_ns'))
        if token in used:continue
        used.add(token)
        if not all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k]) for k in ('latitude_deg','longitude_deg')):continue
        rows.append((key,fix))
    if len(rows)<3:return poses,[],{'horizontal_georeference':False,'reason':'fewer than three simultaneous usable GNSS observations'}
    first=rows[0][1]['position'];origin=[first['latitude_deg'],first['longitude_deg'],first.get('altitude_ellipsoid_m') or 0.]
    xy=np.array([poses[k][:2,3] for k,_ in rows]);geo=[];sigma=[]
    for key,fix in rows:
        p=fix['position'];alt=p.get('altitude_ellipsoid_m')
        ecef=geodetic_to_ecef(p['latitude_deg'],p['longitude_deg'],origin[2] if alt is None else alt)
        geo.append(ecef_to_enu(ecef,origin)[:2]);sigma.append(max(2.,fix['horizontal_accuracy_m'])+2.)
    geo=np.asarray(geo);sigma=np.asarray(sigma);best=None
    # Bounded deterministic pair hypotheses make an isolated confident GNSS
    # jump less able to set the whole map's orientation.
    ids=np.unique(np.linspace(0,len(rows)-1,min(12,len(rows))).astype(int))
    for a in ids:
        for b in ids:
            local=xy[b]-xy[a];global_delta=geo[b]-geo[a]
            if np.linalg.norm(local)<20 or np.linalg.norm(global_delta)<20:continue
            yaw=math.atan2(global_delta[1],global_delta[0])-math.atan2(local[1],local[0])
            c,s=math.cos(yaw),math.sin(yaw);R=np.array([[c,-s],[s,c]])
            t=np.median(geo-xy@R.T,axis=0);dist=np.linalg.norm(xy@R.T+t-geo,axis=1);good=dist<=np.maximum(10.,3*sigma)
            score=(int(good.sum()),-float(np.median(dist)))
            if best is None or score>best[0]:best=(score,R,t,good)
    if best is None or best[3].sum()<3 or best[3].mean()<.6:return poses,[],{'horizontal_georeference':False,'reason':'GNSS/trajectory baseline or consistency insufficient'}
    _,R,t,good=best;T=np.eye(4);T[:2,:2]=R;T[:2,3]=t
    aligned={k:T@p for k,p in poses.items()}
    priors=[(rows[i][0],geo[i],sigma[i]) for i in range(len(rows)) if good[i]]
    return aligned,priors,{'horizontal_georeference':True,'vertical_georeference':False,'origin_wgs84':origin,
                           'gnss_observations':len(priors),'gnss_rejected':int((~good).sum()),
                           'gnss_used_keyframes':[rows[i][0] for i in range(len(rows)) if good[i]],
                           'gnss_rejected_keyframes':[rows[i][0] for i in range(len(rows)) if not good[i]],
                           'uncertainty_note':'Receiver accuracy inflated by 2m for unsurveyed lever arm; horizontal priors only, not surveyed ground truth.'}

class PoseGraph:
    version=VERSION
    def __init__(self,max_evaluations=100):self.max_evaluations=max_evaluations
    def _solve_component(self,root,initial,frames,edges,cancel,location_priors=None):
        initial,priors,geo=horizontal_anchors(frames,initial,location_priors)
        keys=sorted(k for k in initial if k!=root);indices={key:i for i,key in enumerate(keys)}
        rows=[e for e in edges if e['source_id'] in initial]
        matrices=[checked_pose(e['source_from_target']) for e in rows];stds=[np.array(checked_uncertainty(e['std'])) for e in rows]
        def poses_from(x):return {root:initial[root],**{k:unpack(x[i*6:i*6+6]) for k,i in indices.items()}}
        def residual(x):
            check_work(cancel);poses=poses_from(x);values=[]
            for edge,M,std in zip(rows,matrices,stds):values.extend(edge_residual(poses[edge['source_id']],poses[edge['target_id']],M)/std)
            for key,xy,sigma in priors:values.extend((poses[key][:2,3]-xy)/sigma)
            return np.array(values)
        x0=np.concatenate([pack(initial[k]) for k in keys]) if keys else np.array([])
        first=residual(x0)
        if keys and len(first) and np.max(np.abs(first))>1e-7:
            sparsity=lil_matrix((len(first),len(x0)),dtype=int);row=0
            for e in rows:
                for key in (e['source_id'],e['target_id']):
                    if key in indices:sparsity[row:row+6,indices[key]*6:indices[key]*6+6]=1
                row+=6
            for key,_,_ in priors:
                if key in indices:sparsity[row:row+2,indices[key]*6:indices[key]*6+6]=1
                row+=2
            robust=np.r_[np.repeat([e['kind']=='revisit' for e in rows],6),np.ones(len(priors)*2,dtype=bool)].astype(bool)
            def loss(z):
                rho=np.vstack((z,np.ones_like(z),np.zeros_like(z)))
                q=np.sqrt(1+z[robust]);rho[0,robust]=2*(q-1);rho[1,robust]=1/q;rho[2,robust]=-.5/q**3
                return rho
            solved=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),loss=loss,f_scale=1.,max_nfev=self.max_evaluations,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            final=poses_from(solved.x);converged=bool(solved.success and np.isfinite(solved.x).all());evaluations=int(solved.nfev)
        else:final=initial;converged=True;evaluations=0
        diagnostics=[]
        for e,M,std in zip(rows,matrices,stds):
            r=edge_residual(final[e['source_id']],final[e['target_id']],M)
            diagnostics.append({'id':e['id'],'kind':e['kind'],'translation_residual_m':float(np.linalg.norm(r[:3])),
                                'rotation_residual_rad':float(np.linalg.norm(r[3:])),'normalized_norm':float(np.linalg.norm(r/std)),
                                'group':e.get('provenance',{}).get('confirmation_group')})
        return final,diagnostics,{'root':root,'nodes':len(final),'converged':converged,'evaluations':evaluations,**geo}

    def solve(self,keyframes,constraints,location_priors=None,cancel=None):
        frames={f.id:f for f in keyframes}
        for frame in frames.values():frame.validate()
        if not frames:return {'validated':False,'poses':{},'reason':'no keyframes','rejected_constraints':[]}
        edges,rejected=checked_edges(frames,constraints,cancel);poses={};reports=[];diagnostics=[]
        for attempt in range(3):
            check_work(cancel);poses={};reports=[];diagnostics=[]
            for root,initial in components(frames,edges):
                solved,edge_report,report=self._solve_component(root,initial,frames,edges,cancel,location_priors)
                anchor=frames[root];source=anchor.source_ref
                component_id='area-'+identity({'session':anchor.session_id,'raw_device_ns':source.get('first_device_ns',anchor.at_ns),'context_line':source.get('context_line')})[:24]
                report['component_id']=component_id
                reports.append(report);diagnostics.extend(edge_report)
                for key,T in solved.items():
                    poses[key]={'component':component_id,'map_from_body':T.tolist(),
                                'uncertainty':{'kind':'engineering_constraint_weights_not_calibrated_covariance','horizontal_georeference':report['horizontal_georeference']}}
            bad_groups={d['group'] for d in diagnostics if d['kind']=='revisit' and d['normalized_norm']>8}
            if not bad_groups:break
            removed=[e for e in edges if e.get('provenance',{}).get('confirmation_group') in bad_groups]
            rejected.extend({'id':e['id'],'reason':'revisit group inconsistent with graph after robust optimization'} for e in removed)
            edges=[e for e in edges if e not in removed]
        unresolved=[d for d in diagnostics if d['normalized_norm']>12]
        valid=all(r['converged'] for r in reports) and not unresolved and not bad_groups
        return {'validated':bool(valid),'poses':poses,'components':reports,'edge_diagnostics':diagnostics,'rejected_constraints':rejected,
                'unresolved_constraints':len(unresolved)+(len(bad_groups) if bad_groups else 0),'algorithm':self.version,
                'reason':'checked graph solution' if valid else 'nonconvergence or inconsistent constraints; previous revision must remain active'}

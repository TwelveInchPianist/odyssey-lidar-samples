"""Lidar-only bounded rigid registration with retained SE(3) increments."""
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from selfcal_motion import IndependentMotion as PlanarMotion, _velocity

VERSION='lidar-independent-motion-se3-v1'

def rigid_valid(value):
    try:
        r=np.asarray(value['rotation'],float);t=np.asarray(value['translation_m'],float)
        return bool(r.shape==(3,3) and t.shape==(3,) and np.isfinite(r).all() and np.isfinite(t).all() and
                    np.allclose(r.T@r,np.eye(3),atol=1e-6) and abs(np.linalg.det(r)-1)<1e-6)
    except (KeyError,TypeError,ValueError):return False

def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def twist(rotation,translation):
    w=Rotation.from_matrix(rotation).as_rotvec();theta=float(np.linalg.norm(w));W=skew(w)
    J=np.eye(3)+(.5 if theta<1e-7 else (1-math.cos(theta))/theta**2)*W+(1/6 if theta<1e-7 else (theta-math.sin(theta))/theta**3)*(W@W)
    return np.linalg.solve(J,translation),w

def fraction(rotation,translation,scale):
    v,w=twist(rotation,translation);w=w*scale;theta=float(np.linalg.norm(w));W=skew(w)
    J=np.eye(3)+(.5 if theta<1e-7 else (1-math.cos(theta))/theta**2)*W+(1/6 if theta<1e-7 else (theta-math.sin(theta))/theta**3)*(W@W)
    return Rotation.from_rotvec(w).as_matrix(),J@(v*scale)

class IndependentMotion3D:
    def __init__(self,max_points=1600):
        self.max_points=min(3000,max(200,max_points));self.max_target_points=min(24000,self.max_points*12)
        self.reset()
    def reset(self):
        self.previous=None;self.previous_ns=None;self.last_transform=None
    def _features(self,points):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]<3:return np.empty((0,3))
        p=p[np.isfinite(p[:,:3]).all(1)&(p[:,0]>2)&(p[:,0]<95)&(abs(p[:,1])<38)&(abs(p[:,2])<12),:3]
        if len(p)<60:return np.empty((0,3))
        _,inverse=np.unique(np.floor(p[:,:2]).astype(np.int32),axis=0,return_inverse=True)
        low=np.full(inverse.max()+1,np.inf);high=-low.copy()
        np.minimum.at(low,inverse,p[:,2]);np.maximum.at(high,inverse,p[:,2]);p=p[(high-low)[inverse]>=.25]
        _,ids=np.unique(np.floor(p/.25).astype(np.int32),axis=0,return_index=True);p=p[ids]
        if len(p)>self.max_target_points:p=p[np.linspace(0,len(p)-1,self.max_target_points).astype(int)]
        return p
    @staticmethod
    def _fit(query,target,tree,rotation,translation):
        r,t=rotation.copy(),translation.copy()
        for _ in range(32):
            moved=query@r.T+t;dist,ids=tree.query(moved);good=dist<=min(1.5,max(.08,float(np.quantile(dist,.7))))
            if good.sum()<60:return None
            a,b=query[good],target[ids[good]];u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
            nr=vt.T@u.T
            if np.linalg.det(nr)<0:vt[-1]*=-1;nr=vt.T@u.T
            nt=b.mean(0)-nr@a.mean(0);change=np.linalg.norm(nr-r)+np.linalg.norm(nt-t);r,t=nr,nt
            if change<.0001:break
        moved=query@r.T+t;dist,ids=tree.query(moved);good=dist<.35
        if good.sum()<60:return None
        rms=float(np.sqrt(np.mean(dist[good]**2)));rate=float(good.mean())
        return {'r':r,'t':t,'good':good,'residual':moved-target[ids],'rms':rms,'fraction':rate,'score':rms+.2*(1-rate)}
    def update(self,points,at_ns,*,provenance=None):
        base={'version':VERSION,'valid':False,'reference_kind':'lidar_scan_registration_empirical','surveyed_ground_truth':False,
              'independent_of_comma':True,'input_contract':'un-deskewed lidar with one fixed reference rotation','registration_dimensions':3}
        if provenance and (any(provenance.get(k) for k in ('ego_deskewed','comma_seeded','motion_compensated','deskew_source')) or provenance.get('seed_source') not in (None,'lidar_only')):
            self.reset();return dict(base,independent_of_comma=False,reason='circular reference: externally seeded or ego-deskewed points')
        if type(at_ns) is not int or at_ns<=0:self.reset();return dict(base,reason='invalid lidar clock')
        try:current=self._features(points)
        except (ValueError,TypeError,OverflowError):self.reset();return dict(base,reason='invalid lidar geometry')
        old,previous=self.previous,self.previous_ns;self.previous,self.previous_ns=current,at_ns
        base.update(start_ns=previous,end_ns=at_ns,at_ns=at_ns if previous is None else (previous+at_ns)//2)
        if previous is None:return dict(base,reason='waiting for a second lidar scan')
        dt=(at_ns-previous)/1e9;base['dt_s']=dt
        if not .035<=dt<=.25:self.last_transform=None;return dict(base,reason='lidar clock gap or reversal; reference segment restarted')
        if min(len(old),len(current))<60:return dict(base,reason='insufficient static structure')
        def geometry(p):
            _,ids=np.unique(np.floor(p[:,:2]/.2).astype(int),axis=0,return_index=True)
            q=p[ids,:2]
            if len(q)>6000:q=q[np.linspace(0,len(q)-1,6000).astype(int)]
            return PlanarMotion._geometry(q)
        quality=geometry(current)
        if quality is None or geometry(old) is None:return dict(base,reason='insufficient static structure or poorly conditioned scene')
        query=old[np.linspace(0,len(old)-1,min(self.max_points,len(old))).astype(int)];tree=cKDTree(current)
        starts=[(np.eye(3),np.zeros(3))]
        if self.last_transform is not None:starts.append(self.last_transform)
        for angle in (-.08,-.04,0.,.04,.08) if self.last_transform is None else (0.,):
            r=Rotation.from_rotvec([0.,0.,angle]).as_matrix();rotated=query@r.T
            seed=PlanarMotion._seeds(rotated[:,:2],current[:,:2])[1]
            starts.append((r,np.r_[seed,0.]))
        fits=[self._fit(query,current,tree,r,t) for r,t in starts];fits=sorted((f for f in fits if f),key=lambda f:f['score'])
        if not fits:self.last_transform=None;return dict(base,reason='3D scan registration found no supported transform')
        best=fits[0];r,t=best['r'],best['t'];ego_r=r.T;ego_t=-ego_r@t;v,w=twist(ego_r,ego_t)
        for alt in fits[1:]:
            if (np.linalg.norm(alt['t']-t)>.35 or np.linalg.norm(Rotation.from_matrix(alt['r']@r.T).as_rotvec())>.008) and alt['score']-best['score']<.025:
                self.last_transform=None;return dict(base,reason='3D scan registration ambiguous across lidar-only starts')
        good=best['good'];_,inverse=np.unique(np.floor(query[good,:2]/[8.,5.]).astype(int),axis=0,return_inverse=True)
        medians=[np.median(best['residual'][good][inverse==i],axis=0) for i in range(inverse.max()+1) if np.count_nonzero(inverse==i)>=5]
        spread=max((float(np.linalg.norm(x)) for x in medians),default=float('inf'))
        if best['rms']>.18 or best['fraction']<.45 or np.linalg.norm(w/dt)>2 or abs(w[2]/dt)>1.2 or np.linalg.norm(v/dt)>55 or len(medians)<6 or spread>.28:
            self.last_transform=None;return dict(base,reason='3D static-scene consensus failed',quality=dict(quality,rmse_m=best['rms']))
        self.last_transform=(r.copy(),t.copy())
        error=max(.008,best['rms']/math.sqrt(min(12,len(medians))),spread*.35)
        lever=max(5.,float(np.sqrt(np.mean(np.sum((query[good]-query[good].mean(0))**2,axis=1)))))
        quality.update(rmse_m=best['rms'],inlier_fraction=best['fraction'],sector_residual_m=spread,matched_points=int(good.sum()),
            query_points=len(query),target_points=len(current),translation_error_m=error,yaw_error_rad=max(.0002,error/lever),
            rotation_error_rad=max(.0002,error/lever),uncertainty_kind='empirical 3D region-consistency estimate, not a guaranteed bound',static_scene_assumption=True)
        return dict(base,valid=True,reason='lidar-only full 3D relative motion available',quality=quality,
            ego_increment_se3={'rotation':ego_r.tolist(),'translation_m':ego_t.tolist()},
            ego_increment_se2={'translation_m':ego_t[:2].tolist(),'yaw_rad':float(w[2])},
            velocity_lidar_mps=(v[:2]/dt).tolist(),velocity_lidar_xyz_mps=(v/dt).tolist(),angular_velocity_lidar_rps=(w/dt).tolist(),speed_mps=float(v[0]/dt),yaw_rate_rps=float(w[2]/dt))

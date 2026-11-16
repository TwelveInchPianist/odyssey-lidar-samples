"""Independent scan-registration yaw estimates for timing diagnosis.

Never seeded by comma yaw. Rejection is preferable to claiming timing from a
featureless road or one moving object. Outputs remain calibration proposals.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

class ScanMotion:
    def __init__(self):self.previous=None;self.previous_ns=None;self.last_transform=None

    def update(self,points,stamp_ns):
        p=points[(points[:,0]>4)&(points[:,0]<70)&(np.abs(points[:,1])<25)&(points[:,2]>.3)&(points[:,2]<4),:2]
        if len(p)>0:
            _,idx=np.unique(np.floor(p/.35).astype(int),axis=0,return_index=True);p=p[idx]
        if len(p)>1800:p=p[::math.ceil(len(p)/1800)]
        old=self.previous;last=self.previous_ns;self.previous=p;self.previous_ns=stamp_ns
        if old is None or last is None or min(len(old),len(p))<80:
            return {'valid':False,'reason':'insufficient independent static-scene features'}
        dt=(stamp_ns-last)/1e9
        if not .04<dt<.2:return {'valid':False,'reason':'registration scan timing discontinuity'}
        if np.ptp(p[:,0])<10 or np.ptp(p[:,1])<4:return {'valid':False,'reason':'scene geometry too narrow'}
        tree=cKDTree(p);R=np.eye(2);t=np.zeros(2)
        if self.last_transform is not None:R,t=self.last_transform
        rmse=99;inliers=0
        for _ in range(18):
            moved=old@R.T+t;dist,index=tree.query(moved,k=1)
            cut=min(2.5,float(np.quantile(dist,.65)))
            good=dist<=max(.08,cut)
            if good.sum()<60:return {'valid':False,'reason':'registration too few matches'}
            a,b=old[good],p[index[good]];ac=a-a.mean(0);bc=b-b.mean(0)
            _,sing,_=np.linalg.svd(ac.T@ac)
            if sing[-1]/max(sing[0],1e-9)<.015:return {'valid':False,'reason':'registration geometry is degenerate'}
            U,_,Vt=np.linalg.svd(ac.T@bc);next_R=Vt.T@U.T
            if np.linalg.det(next_R)<0:Vt[-1]*=-1;next_R=Vt.T@U.T
            next_t=b.mean(0)-a.mean(0)@next_R.T
            change=np.linalg.norm(next_R-R)+np.linalg.norm(next_t-t)
            R,t=next_R,next_t
            rmse=float(np.sqrt(np.mean(np.sum((a@R.T+t-b)**2,axis=1))))
            inliers=int(good.sum())
            if change<.0005:break
        angle=math.atan2(R[1,0],R[0,0]);rate=-angle/dt
        valid=bool(rmse<.22 and abs(rate)<.8 and np.linalg.norm(t)/dt<55 and inliers/len(old)>.4)
        if valid:self.last_transform=(R,t)
        else:self.last_transform=None
        return {'valid':valid,'reason':'proposal only; independent validation required' if valid else 'registration residual/motion rejected',
                'at_ns':int((stamp_ns+last)//2),'yaw_rate_rps':rate,'rmse_m':rmse,'matches':inliers}

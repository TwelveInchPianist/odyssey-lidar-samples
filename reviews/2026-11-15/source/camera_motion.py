"""Measured image motion for track association, never range or hazard evidence.

Sparse optical flow estimates the dominant inter-frame image transform. A weak
or implausible fit is returned as unavailable rather than inventing stabilization.
Transforms do not modify images, detector boxes, class scores or observations.
"""
import math


def transform_box(box,matrix):
    corners=[(box[0],box[1]),(box[2],box[1]),(box[0],box[3]),(box[2],box[3])]
    moved=[(matrix[0][0]*x+matrix[0][1]*y+matrix[0][2],
            matrix[1][0]*x+matrix[1][1]*y+matrix[1][2]) for x,y in corners]
    return [min(p[0] for p in moved),min(p[1] for p in moved),
            max(p[0] for p in moved),max(p[1] for p in moved)]


class CameraMotion:
    def __init__(self):self.previous=None;self.last_ns=None;self.shape=None

    def update(self,image,source_ns):
        import cv2,numpy as np
        h,w=image.shape[:2];width=640;height=round(h*width/w)
        gray=cv2.cvtColor(cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        old=self.previous;last=self.last_ns;shape=self.shape
        self.previous=gray;self.last_ns=source_ns;self.shape=(h,w)
        result={'valid':False,'source_ns':source_ns,'previous_source_ns':last,
                'purpose':'image_track_association_only','matrix':None}
        if old is None or shape!=(h,w) or not 0<source_ns-last<=600_000_000:
            return {**result,'reason':'first_frame_gap_or_shape_change'}
        points=cv2.goodFeaturesToTrack(old,maxCorners=400,qualityLevel=.01,minDistance=8,blockSize=7)
        if points is None or len(points)<30:return {**result,'reason':'too_few_image_features'}
        moved,status,_=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(21,21),maxLevel=3)
        if moved is None:return {**result,'reason':'flow_unavailable'}
        reverse,back_status,_=cv2.calcOpticalFlowPyrLK(gray,old,moved,None,winSize=(21,21),maxLevel=3)
        if reverse is None:return {**result,'reason':'reverse_flow_unavailable'}
        good=(status.ravel()!=0)&(back_status.ravel()!=0)&(np.linalg.norm(reverse-points,axis=2).ravel()<1.)
        a=points.reshape(-1,2)[good];b=moved.reshape(-1,2)[good]
        if len(a)<30:return {**result,'reason':'too_few_consistent_flows','features':len(a)}
        matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,confidence=.99,maxIters=1000)
        if matrix is None:return {**result,'reason':'dominant_motion_unavailable'}
        count=int(inliers.sum());fraction=count/len(a)
        scale=float(math.hypot(matrix[0,0],matrix[1,0]));angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
        result.update(features=len(a),inliers=count,inlier_fraction=fraction,scale=scale,rotation_deg=angle)
        if count<30 or fraction<.55 or not .90<=scale<=1.10 or abs(angle)>10 or abs(matrix[0,2])>.15*width or abs(matrix[1,2])>.15*height:
            return {**result,'reason':'dominant_motion_not_consistent'}
        residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)[inliers.ravel()!=0]
        # Scaling back to native pixels preserves the actual detector coordinates.
        sx=w/width;sy=h/height
        native=[[float(matrix[0,0]),float(matrix[0,1]*sx/sy),float(matrix[0,2]*sx)],
                [float(matrix[1,0]*sy/sx),float(matrix[1,1]),float(matrix[1,2]*sy)]]
        return {**result,'valid':True,'reason':'observed_dominant_image_motion',
                'matrix':native,'residual_p95_scaled_px':float(np.percentile(residual,95))}

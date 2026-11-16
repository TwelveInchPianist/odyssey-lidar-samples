"""One fixed lidar-only road reference, chosen without held-out blocks."""
import math
import numpy as np

def choose_reference(rows):
    groups={}
    for row in rows:
        try:
            normal=np.asarray(row['sensor_normal'],float);at=row['at_ns']
            if (type(at) is not int or normal.shape!=(3,) or not np.isfinite(normal).all() or normal[2]<=0 or
                row['support_points']<200 or not 0<=row['fit_rmse_m']<=.15 or row['span_x_m']<10 or row['span_y_m']<4):continue
            normal/=np.linalg.norm(normal)
            value=[row['plane_offset_m'],math.atan2(normal[1],normal[2]),math.atan2(-normal[0],math.hypot(normal[1],normal[2]))]
            if not .5<value[0]<4 or max(abs(v) for v in value[1:])>math.radians(20):continue
            groups.setdefault(at//2_000_000_000,[]).append(value)
        except (KeyError,TypeError,ValueError,ZeroDivisionError):continue
    blocks={k:np.median(v,axis=0) for k,v in groups.items() if len(v)>=3}
    ordered=sorted(blocks);cut=max(1,min(len(ordered)-1,int(math.floor(len(ordered)*.7))))
    result={'eligible':False,'method':'fixed_lidar_reference_from_fit_only_ground_blocks_v1',
        'qualified_blocks':len(ordered),'fit_blocks':ordered[:cut],'holdout_blocks_excluded':ordered[cut:],
        'minimum_support_points':200,'heldout_used_for_selection':False,'mount_ground_truth':False}
    if len(ordered)<6:result['reason']='Need six well-supported ground blocks before choosing a fixed reference.';return None,result
    values=np.array([blocks[k] for k in ordered[:cut]]);center=np.median(values,axis=0)
    mad=np.median(abs(values-center),axis=0)*1.4826
    good=np.all(abs(values-center)<=np.maximum(3*mad,[.08,math.radians(.8),math.radians(.8)]),axis=1)
    if good.sum()<4 or good.mean()<.7:result['reason']='Fitting road blocks do not support a common reference.';return None,result
    height,roll,pitch=np.median(values[good],axis=0)
    normal=np.array([-math.sin(pitch),math.cos(pitch)*math.sin(roll),math.cos(pitch)*math.cos(roll)])
    cross=np.cross(normal,[0.,0.,1.]);x,y,z=cross;S=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    rotation=np.eye(3)+S+S@S/(1+normal[2])
    result.update(eligible=True,reason=None,inlier_fit_blocks=int(good.sum()),rotation=rotation.tolist(),
        road_reference_height_m=float(height),road_reference_roll_pitch_deg=np.degrees([roll,pitch]).tolist(),
        limitation='A road/body reference, not an independent mounting survey. All later motion/object rows use this exact fixed rotation.')
    return rotation,result

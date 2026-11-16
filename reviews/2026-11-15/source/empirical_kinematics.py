"""Shared camera-origin turn kinematics; no calibration authority here."""
import math
import numpy as np
LEGACY='zero_nominal_with_observed_error_bound'
MODEL='forward_lever_arm_with_observed_error_bound'
def lever(motion):
    name=motion.get('lateral_velocity_model')
    if name==LEGACY:return 0.
    if name!=MODEL:raise ValueError('Unknown lateral motion model')
    value=motion.get('kinematic_forward_lever_m');bound=motion.get('kinematic_forward_lever_error_m')
    if (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=4 or
        type(bound) not in (int,float) or not math.isfinite(bound) or not 0<bound<=6):raise ValueError('Invalid kinematic lever estimate')
    return float(value)
def path(speed,yaw,times,forward_lever=0.):
    times=np.asarray(times,float);angle=yaw*times
    if abs(yaw)<1e-7:base=np.column_stack((speed*times,np.zeros_like(times)))
    else:base=np.column_stack((speed/yaw*np.sin(angle),speed/yaw*(1-np.cos(angle))))
    return base+forward_lever*np.column_stack((np.cos(angle)-1,np.sin(angle)))

def origin_velocity_error(angular_velocity, position_error):
    """Axis-wise bound for omega cross uncertain lever arm."""
    w=np.abs(np.asarray(angular_velocity,float));b=np.asarray(position_error,float)
    if w.shape!=(3,) or b.shape!=(3,) or not np.isfinite(w).all() or not np.isfinite(b).all() or np.any(b<0):
        raise ValueError('Invalid angular velocity or position uncertainty')
    return np.array([w[1]*b[2]+w[2]*b[1],w[2]*b[0]+w[0]*b[2],w[0]*b[1]+w[1]*b[0]])

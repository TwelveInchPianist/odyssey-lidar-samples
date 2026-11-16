"""Independent lidar geometry candidate; rough priors never become survey truth."""
import math,hashlib,json
import numpy as np
from selfcal_alignment import _ground,_vec,_number,_epoch
from selfcal_motion import _windows

def fit_lidar_geometry(ground_rows,motion_rows,metadata):
    ground=_ground(list(ground_rows),metadata);prior=metadata.get('lidar_geometry_prior')
    result={'eligible':False,'method':'lidar_motion_and_bounded_installation_v1','evidence_type':'sensor_derived_empirical',
        'ground_truth':False,'not_surveyed':True,'radar_required':False,'mount_epoch':_epoch(metadata.get('mount_epoch')),
        'ground':ground,'reasons':[],'geometry':{},'mount':{},'geometry_basis':'independent_lidar_bounded_prior',
        'requires_observed_ground':True,'radar_alignment':{'eligible':False,'reasons':['Radar is separately qualified.']}}
    if (not isinstance(prior,dict) or prior.get('mount_epoch')!=result['mount_epoch'] or
        prior.get('frame')!='camera_ground_flu' or prior.get('evidence_kind')!='rough_bounded_installation' or
        prior.get('surveyed') is not False or not isinstance(prior.get('assumptions'),list) or not prior['assumptions']):
        result['reasons']=['Independent lidar geometry needs an explicit current-mount position prior with bounds; radar failure is not the reason.'];return result
    position=_vec(prior.get('position_m'),3);bounds=_vec(prior.get('position_error_bound_m'),3)
    if position is None or bounds is None or np.any(bounds<=0) or np.any(bounds>[3,1,1]) or np.any(abs(position)>[5,2,4]):
        result['reasons']=['Independent installation position or its declared uncertainty is invalid.'];return result
    fit=[];hold=[];quality=[]
    for window in _windows(motion_rows):
        segments=[];R=np.eye(3);t=np.zeros(3);duration=0.;translation_error=0.;angle_error=0.;last=None
        from selfcal_motion3d import twist
        from scipy.spatial.transform import Rotation
        for row in window['rows']:
            if abs(row['yaw_rate_rps'])>.015 or row['speed_mps']<5 or (last is not None and row['start_ns']!=last):
                R=np.eye(3);t=np.zeros(3);duration=0.;translation_error=0.;angle_error=0.;last=None
                if abs(row['yaw_rate_rps'])>.015 or row['speed_mps']<5:continue
            raw=row.get('ego_increment_se3')
            if raw:r=np.asarray(raw['rotation']);part=np.asarray(raw['translation_m'])
            else:
                angle=row['ego_increment_se2']['yaw_rad'];r=Rotation.from_rotvec([0,0,angle]).as_matrix();part=np.r_[row['ego_increment_se2']['translation_m'],0.]
            q=row['quality'];ae=q.get('rotation_error_rad',q['yaw_error_rad'])
            translation_error+=q['translation_error_m']+np.linalg.norm(part)*angle_error
            angle_error+=ae;t+=R@part;R=R@r;duration+=row['dt_s'];last=row['end_ns']
            if duration>=2.-1e-8:
                linear,angular=twist(R,t);distance=float(np.linalg.norm(linear[:2]))
                # Restrict the small-angle Jacobian bound; no extrapolation
                # through missing registration or unsupported curvature.
                if distance>=10 and linear[0]>0 and np.linalg.norm(angular)<=.12:
                    bound=1.1*(translation_error/distance+angle_error)
                    segments.append({'angle':-math.atan2(linear[1],linear[0]),'error':bound,'distance_m':distance})
                R=np.eye(3);t=np.zeros(3);duration=0.;translation_error=0.;angle_error=0.;last=None
        if not segments:continue
        center=float(np.median([v['angle'] for v in segments]))
        item={'window':window['id'],'angle':center,'error':float(max(v['error'] for v in segments)),
              'within_window_spread':float(max(abs(v['angle']-center) for v in segments)),
              'segments':segments,'reference':'contiguous two-second SE3 screw displacement; bounded accumulated errors'}
        (hold if window['role']=='holdout' else fit).append(item)
    result['yaw_evidence']={'fit_windows':fit,'holdout_windows':hold,'selection':'lidar-derived near-straight forward motion',
        'assumptions':'Near-straight non-slipping motion; four-metre effective lever bound; one-degree additional axis assumption floor.',
        'fit_does_not_use_holdout':True}
    if len(fit)<3 or len(hold)<2:
        result['reasons'].append('Independent yaw needs three fitting and two held-out straight-motion windows.');return result
    yaw=float(np.median([r['angle'] for r in fit]))
    fit_spread=max(abs(r['angle']-yaw) for r in fit)
    yaw_bound=max(abs(r['angle']-yaw)+r['within_window_spread']+r['error'] for r in hold)+fit_spread+math.asin(4*.015/5)+math.radians(1.)
    result['yaw_evidence'].update(yaw_rad=yaw,error_bound_rad=yaw_bound)
    if abs(yaw)>.175 or yaw_bound>math.radians(5):result['reasons'].append('Independent lidar yaw uncertainty exceeds five degrees.')
    if ground.get('eligible') is not True:result['reasons']+=ground.get('reasons') or ['Independent road/body reference is unsupported.']
    # Do not replace uncertain tape height with a more precise-looking plane.
    h=ground.get('height_m');hb=ground.get('height_error_bound_m')
    if _number(h) is None or _number(hb) is None:result['reasons'].append('Road-height evidence unavailable.');return result
    position[2]=h;bounds[2]=max(bounds[2]+abs(h-prior['position_m'][2]),hb)
    rotation=np.asarray(ground['level_rotation_sensor_to_road']);z=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
    r=z@rotation;angles=np.degrees([math.atan2(r[2,1],r[2,2]),math.asin(np.clip(-r[2,0],-1,1)),math.atan2(r[1,0],r[0,0])]).tolist()
    result['geometry']={'yaw_rad':yaw,'sensor_origin_camera_flu_m':position.tolist(),'translation_error_bound_m':bounds.tolist(),
        'yaw_error_bound_rad':yaw_bound,'geometry_basis':result['geometry_basis'],'requires_observed_ground':True,
        'frame':'camera_ground_flu','position_evidence':prior,'radar_used':False}
    result['mount']={'xyz':position.tolist(),'rpy_deg':angles};result['eligible']=not result['reasons']
    result['proposal_id']=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result

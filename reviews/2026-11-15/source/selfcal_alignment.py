"""Empirical alignment proposals from own sensor data; never survey ground truth.

The input XY convention is ground-leveled lidar FLU and approximately camera-
origin radar FLU. Factory origin/axis assumptions remain common systematic
uncertainties, distinct from fitted relative geometry. This module writes no
configuration and does not grant permission to enable warning output.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares

SCHEMA_VERSION=1
BLOCK_NS=2_000_000_000
LIMITS={'ground_blocks':6,'radar_blocks':12,'fit_tracks':4,'holdout_new_tracks':2,
        'range_span_m':20.,'within_track_span_m':8.,'speed_span_for_delay_mps':5.,
        'association_margin_m':.5,'ground_rmse_m':.15,'relative_y_bound_m':.75,
        'relative_x_bound_m':1.25,'relative_yaw_bound_deg':1.,'minimum_supported_range_m':15.,
        'tight_ground_height_bound_m':.35,'tight_ground_angle_bound_deg':2.,
        'road_relative_height_bound_m':.6,'road_relative_angle_bound_deg':5.}


def _number(v):
    if isinstance(v,np.generic):v=v.item()
    if type(v) not in (int,float):return None
    try:return float(v) if math.isfinite(float(v)) else None
    except (ValueError,OverflowError):return None


def _vec(v,n):
    try:
        if not isinstance(v,(list,tuple,np.ndarray)) or len(v)!=n:return None
        values=[_number(x) for x in v]
        return np.array(values,float) if all(x is not None for x in values) else None
    except (TypeError,ValueError):return None


def _epoch(v):
    if isinstance(v,dict):v=v.get('id')
    return v if isinstance(v,str) and v.strip() else None


def _scope(row,metadata):
    if not isinstance(row,dict):return None
    stamp=row.get('at_ns');session=row.get('session_id',metadata.get('session_id'))
    current=_epoch(metadata.get('mount_epoch'))
    if type(stamp) is not int or stamp<=0 or not isinstance(session,str) or not session:return None
    if current is None or _epoch(row.get('mount_epoch'))!=current:return None
    # Caller block labels cannot turn duplicated frames into independent blocks.
    return session,stamp//BLOCK_NS


def _split(blocks):
    ordered=sorted(set(blocks));cut=max(1,min(len(ordered)-1,int(math.floor(len(ordered)*.7))))
    return set(ordered[:cut]),set(ordered[cut:])


def _rotation(theta):
    c,s=math.cos(theta),math.sin(theta)
    return np.array([[c,-s],[s,c]])


def _level_rotation(roll,pitch):
    normal=np.array([-math.sin(pitch),math.cos(pitch)*math.sin(roll),math.cos(pitch)*math.cos(roll)])
    target=np.array([0.,0.,1.]);axis=np.cross(normal,target);c=float(normal@target)
    skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    return np.eye(3)+skew+skew@skew/(1+c)


def _reference_rotation(rows,metadata):
    matrices=[]
    if metadata.get('fixed_level_rotation') is not None:matrices.append(metadata['fixed_level_rotation'])
    matrices.extend(row['reference_level_rotation'] for row in rows if _scope(row,metadata) is not None and row.get('reference_level_rotation') is not None)
    if not matrices:return None,None
    reference=None
    for raw in matrices:
        try:rotation=np.asarray(raw,float)
        except (ValueError,TypeError,OverflowError):return None,'Invalid fixed leveling reference.'
        if (rotation.shape!=(3,3) or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
                not math.isclose(float(np.linalg.det(rotation)),1.,abs_tol=1e-6) or
                min(rotation[0,0],rotation[1,1])<.8 or rotation[2,2]<.9):
            return None,'Invalid fixed leveling reference.'
        if reference is None:reference=rotation
        elif not np.allclose(reference,rotation,atol=1e-7):return None,'Mixed leveling reference frames cannot share one alignment fit.'
    return reference,None


def _ground(rows,metadata):
    groups=defaultdict(list);rejected=Counter();seen=set()
    for row in rows:
        key=_scope(row,metadata)
        if key is None:rejected['invalid_timestamp_session_or_mount_epoch']+=1;continue
        duplicate=(key[0],row['at_ns'])
        if duplicate in seen:rejected['duplicate_frame']+=1;continue
        seen.add(duplicate)
        rmse=_number(row.get('fit_rmse_m'));support=_number(row.get('support_points'))
        sx=_number(row.get('span_x_m'));sy=_number(row.get('span_y_m'))
        if (rmse is None or not 0<=rmse<=LIMITS['ground_rmse_m'] or support is None or support<200 or
                sx is None or sx<10 or sy is None or sy<4):
            rejected['weak_ground_geometry']+=1;continue
        abc=_vec(row.get('sensor_abc'),3)
        if abc is not None:
            normal=np.array([-abc[0],-abc[1],1.]);offset=-abc[2]
        else:
            normal=_vec(row.get('sensor_normal'),3);offset=_number(row.get('plane_offset_m'))
            if normal is None or offset is None:rejected['missing_sensor_plane']+=1;continue
        norm=float(np.linalg.norm(normal))
        if not .01<norm<100:rejected['invalid_sensor_plane']+=1;continue
        normal=normal/norm;offset=offset/norm
        if normal[2]<0:normal=-normal;offset=-offset
        height=offset
        roll=math.atan2(float(normal[1]),float(normal[2]))
        pitch=math.atan2(float(-normal[0]),math.hypot(float(normal[1]),float(normal[2])))
        if not .5<height<4 or max(abs(roll),abs(pitch))>math.radians(20):
            rejected['implausible_roof_plane']+=1;continue
        groups[key].append([height,roll,pitch])
    summaries={k:np.median(v,axis=0) for k,v in groups.items() if len(v)>=3}
    result={'status':'insufficient','eligible':False,'height_m':None,'roll_deg':None,'pitch_deg':None,
            'height_error_bound_m':None,'roll_pitch_error_bound_deg':None,'blocks':len(summaries),
            'rejected_rows':dict(rejected),'not_surveyed':True,
            'assumption':'Locally observed road is approximately level relative to the vehicle body; road grade/bank and body tilt are confounded.'}
    if len(summaries)<LIMITS['ground_blocks']:
        result['reasons']=['Need at least six two-second ground blocks with broad, well-fitted pavement.'];return result
    fit,hold=_split(summaries);a=np.array([summaries[k] for k in sorted(fit)])
    b=np.array([summaries[k] for k in sorted(hold)]);center=np.median(a,axis=0)
    mad=np.median(np.abs(a-center),axis=0)*1.4826
    good=np.all(np.abs(a-center)<=np.maximum(3*mad,[.08,math.radians(.8),math.radians(.8)]),axis=1)
    if good.sum()<4 or good.mean()<.7:
        result['reasons']=['Ground observations vary too much across independent blocks.'];return result
    center=np.median(a[good],axis=0)
    rng=np.random.default_rng(1091)
    boot=np.array([np.median(a[good][rng.integers(0,int(good.sum()),int(good.sum()))],axis=0) for _ in range(128)])
    spread=np.max(np.abs(boot-center),axis=0)
    median_center=center.copy()
    reference,reference_error=_reference_rotation(rows,metadata)
    if reference_error:
        result['reasons']=[reference_error];return result
    if reference is not None:
        center[1:]=[math.atan2(reference[2,1],reference[2,2]),math.asin(float(np.clip(-reference[2,0],-1,1)))]
    hold_error=np.max(np.abs(b-center),axis=0)
    if reference is not None:
        hold_error[1:]=np.maximum(hold_error[1:],np.max(np.abs(a[good,1:]-center[1:]),axis=0))
    # Floors are explicit road/body assumptions, not measured statistical error.
    bounds=hold_error+spread+np.array([.05,math.radians(.5),math.radians(.5)])
    accepted=bool(bounds[0]<=LIMITS['tight_ground_height_bound_m'] and
                  np.all(bounds[1:]<=math.radians(LIMITS['tight_ground_angle_bound_deg'])))
    # A road plane cannot distinguish mounting error from road bank/body pitch.
    # Broader *observed* variation can support a separate road-relative mode,
    # but only after held-out planar radar/lidar agreement and only while a
    # current/recent observed road is available. Never trust its global floor.
    provisional=bool(not accepted and reference is not None and
        bounds[0]<=LIMITS['road_relative_height_bound_m'] and
        np.all(bounds[1:]<=math.radians(LIMITS['road_relative_angle_bound_deg'])))
    result.update(status='conditional_road_level_estimate' if accepted else
                         'road_relative_candidate_requires_planar_validation' if provisional else 'unstable_holdout',eligible=accepted,
        provisional_road_relative_candidate=provisional,requires_observed_ground=provisional,
        geometry_basis='empirical_tight_road_level' if accepted else 'provisional_road_relative' if provisional else 'unsupported_road_relative_geometry',
        variation_scope='observed road/body normal and height variation; not separately identified fixed sensor-pose error',
        height_m=float(center[0]),roll_deg=math.degrees(float(center[1])),pitch_deg=math.degrees(float(center[2])),
        height_error_bound_m=float(bounds[0]),roll_pitch_error_bound_deg=np.degrees(bounds[1:]).tolist(),
        level_rotation_sensor_to_road=(reference if reference is not None else _level_rotation(float(center[1]),float(center[2]))).tolist(),
        reference_level_rotation_preserved=reference is not None,
        median_road_roll_pitch_deg=np.degrees(median_center[1:]).tolist(),
        fit_blocks=len(fit),holdout_blocks=len(hold),fit_inlier_blocks=int(good.sum()),
        holdout_max_error={'height_m':float(hold_error[0]),'roll_pitch_deg':np.degrees(hold_error[1:]).tolist()},
        assumption_floor={'height_m':.05,'roll_pitch_deg':[.5,.5]},
        reasons=[] if accepted else ['Broader road-relative geometry requires successful independent held-out planar alignment and observed live ground.'] if provisional else
                ['Held-out ground variation exceeds supported road-relative bounds or lacks a fixed reference frame.'])
    reference_quality=metadata.get('reference_quality')
    if isinstance(reference_quality,dict) and reference_quality.get('eligible') is not True:
        result.update(eligible=False,provisional_road_relative_candidate=False,requires_observed_ground=False,
            status='unqualified_fixed_reference',reasons=[reference_quality.get('reason') or 'Fixed road reference is not qualified.'])
    return result


def _pairs(rows,metadata):
    accepted=[];rejected=Counter();seen=set();byframe=defaultdict(list)
    for row in rows:
        key=_scope(row,metadata)
        if key is None:rejected['invalid_timestamp_session_or_mount_epoch']+=1;continue
        lid=row.get('lidar_track_id');rid=row.get('radar_track_id')
        if type(lid) not in (str,int) or type(rid) not in (str,int) or isinstance(lid,bool) or isinstance(rid,bool):
            rejected['missing_track_identity']+=1;continue
        signature=(key[0],row['at_ns'],str(lid),str(rid))
        if signature in seen:rejected['duplicate_pair']+=1;continue
        seen.add(signature)
        l=_vec(row.get('lidar_xy'),2);r=_vec(row.get('radar_xy'),2)
        rawlo=row.get('lidar_lo');rawhi=row.get('lidar_hi')
        lo=_vec(rawlo[:2],2) if isinstance(rawlo,(list,tuple,np.ndarray)) and len(rawlo)>=2 else None
        hi=_vec(rawhi[:2],2) if isinstance(rawhi,(list,tuple,np.ndarray)) and len(rawhi)>=2 else None
        dt=_number(row.get('time_delta_s'));timing=_number(row.get('timing_error_bound_s'))
        speed=_number(row.get('ego_speed'));vrel=_number(row.get('radar_vrel'))
        margin=_number(row.get('association_margin_m'))
        ground=row.get('ground_quality') or {};quality=row.get('static_quality') or {}
        if not isinstance(ground,dict) or not isinstance(quality,dict):rejected['quality_missing']+=1;continue
        rmse=_number(ground.get('fit_rmse_m',ground.get('rmse_m')))
        age=_number(quality.get('age_s',row.get('lidar_track_age_s')))
        residual=_number(quality.get('residual_m',row.get('lidar_track_residual_m')))
        yaw=_number(row.get('ego_yaw_rate_rps',0.))
        if (l is None or r is None or lo is None or hi is None or np.any(hi<lo) or
                np.any(np.abs(l)>200) or np.any(np.abs(r)>200) or not 5<=l[0]<=120 or
                not np.all(l>=lo-.1) or not np.all(l<=hi+.1)):
            rejected['invalid_or_nonlocal_geometry']+=1;continue
        extent=(hi-lo)/2
        if extent[0]>2 or extent[1]>.8:
            rejected['broad_or_merged_target_reference']+=1;continue
        if row.get('unique_match') is not True or margin is None or margin<LIMITS['association_margin_m']:
            rejected['association_not_unique']+=1;continue
        if (rmse is None or not 0<=rmse<=.15 or ground.get('source','fit')!='fit' or
                age is None or age<.3 or residual is None or not 0<=residual<=.5 or
                quality.get('radar_measured',quality.get('measured',row.get('radar_measured'))) is not True):
            rejected['weak_ground_or_track_stability']+=1;continue
        convention=row.get('time_delta_convention','radar_publication_minus_lidar')
        if convention=='lidar_minus_radar_publication' and dt is not None:dt=-dt
        elif convention!='radar_publication_minus_lidar':rejected['unknown_time_delta_convention']+=1;continue
        point_span=_number(row.get('lidar_point_time_half_span_s',0.))
        if point_span is None or not 0<=point_span<=.06:rejected['point_time_span_unbounded']+=1;continue
        if (dt is None or abs(dt)>.12 or timing is None or not 0<=timing<=.04 or speed is None or
                not 0<=speed<=45 or vrel is None or abs(vrel+speed)>1.5 or yaw is None or abs(yaw)>.05):
            rejected['static_or_temporal_screen_failed']+=1;continue
        prior=_vec(row.get('radar_origin_prior_m',metadata.get('radar_origin_prior_m',[1.52,0.])),2)
        prior_bound=_vec(row.get('radar_origin_prior_error_bound_m',row.get('radar_origin_prior_bound_m',
                             metadata.get('radar_origin_prior_error_bound_m',[1.,.35]))),2)
        if (prior is None or prior_bound is None or np.any(prior_bound<0) or
                np.any(np.abs(prior)>[5,2]) or np.any(prior_bound>[3,1])):
            rejected['origin_prior_missing_or_invalid']+=1;continue
        # Identity is scoped by source and mount/session, never a naked track ID.
        object_key=(key[0],str(row.get('source_epoch','unknown')),str(lid),str(rid))
        item={'block':key,'object':object_key,'stamp':row['at_ns'],'lidar':l,'radar':r,
              'extent':extent,'speed':speed,'vrel':vrel,'dt':dt,'timing':timing+point_span,'prior':prior,'prior_bound':prior_bound,
              'object_velocity':_vec(row.get('lidar_ground_velocity_mps'),2),
              'ego_velocity':_vec(row.get('lidar_independent_ego_velocity_mps'),2),
              'independent_yaw':_number(row.get('lidar_independent_yaw_rps')),
              'velocity_bound':_vec(row.get('lidar_velocity_error_bound_mps'),2),
              'yaw_bound':_number(row.get('lidar_yaw_error_bound_rps')),
              'z_bounds':([float(rawlo[2]),float(rawhi[2])] if len(rawlo)>=3 and len(rawhi)>=3 and
                          _number(rawlo[2]) is not None and _number(rawhi[2]) is not None and rawhi[2]>=rawlo[2] else None),
              'includes_lidar_alignment':row.get('includes_lidar_alignment') is True,
              'includes_radar_latency':row.get('includes_radar_latency') is True}
        byframe[(key[0],row['at_ns'])].append(item)
    for frame in byframe.values():
        left=Counter(x['object'][2] for x in frame);right=Counter(x['object'][3] for x in frame)
        for item in frame:
            if left[item['object'][2]]!=1 or right[item['object'][3]]!=1:
                rejected['competing_pair_same_frame']+=1
            else:accepted.append(item)
    groups=defaultdict(list)
    for row in accepted:groups[(row['block'],row['object'])].append(row)
    summaries=[]
    common_prior=_vec(metadata.get('radar_origin_prior_m',[1.52,0.]),2)
    if common_prior is None:common_prior=np.array([1.52,0.])
    for (block,obj),values in sorted(groups.items()):
        if len(values)<3 or max(x['stamp'] for x in values)-min(x['stamp'] for x in values)<150_000_000:
            rejected['short_object_block']+=len(values);continue
        velocity_rows=[x for x in values if x['object_velocity'] is not None and x['ego_velocity'] is not None and
                       x['independent_yaw'] is not None and x['velocity_bound'] is not None and np.all(x['velocity_bound']>0) and
                       np.all(x['velocity_bound']<=10) and x['yaw_bound'] is not None and 0<x['yaw_bound']<=.2]
        summaries.append({'block':block,'object':obj,'samples':len(values),
            'lidar':np.median([x['lidar'] for x in values],axis=0),
            'radar':np.median([x['radar']+common_prior-x['prior'] for x in values],axis=0),
            'extent':np.max([x['extent'] for x in values],axis=0),
            'speed':float(np.median([x['speed'] for x in values])),
            'vrel':float(np.median([x['vrel'] for x in values])),
            'dt':float(np.median([x['dt'] for x in values])),
            'timing':max(x['timing'] for x in values),
            'prior_bound':np.max([x['prior_bound'] for x in values],axis=0),
            'z_bounds':([min(x['z_bounds'][0] for x in values),max(x['z_bounds'][1] for x in values)]
                        if all(x['z_bounds'] is not None for x in values) else None),
            'independent_velocity':({'object':np.median([x['object_velocity'] for x in velocity_rows],axis=0),
                                     'ego':np.median([x['ego_velocity'] for x in velocity_rows],axis=0),
                                     'yaw':float(np.median([x['independent_yaw'] for x in velocity_rows])),
                                     'bound':np.max([x['velocity_bound'] for x in velocity_rows],axis=0),
                                     'yaw_bound':max(x['yaw_bound'] for x in velocity_rows)} if len(velocity_rows)>=3 else None),
            'complete_timing':all(x['includes_lidar_alignment'] and x['includes_radar_latency'] for x in values)})
    return summaries,dict(rejected),common_prior


def _velocity_support(fit,hold,parameters,relative_bound,yaw_bound,prior):
    """Independent lidar relative velocity is the comparison, never comma vEgo."""
    result={'available':False,'method':'independent_lidar_relative_velocity_vs_native_oem_vrel',
            'reference_is_ground_truth':False,'bias_fitted_on_holdout':False}
    fit=[r for r in fit if r['independent_velocity'] is not None]
    hold=[r for r in hold if r['independent_velocity'] is not None]
    result.update(fit_object_blocks=len(fit),holdout_object_blocks=len(hold),
                  fit_temporal_blocks=len({r['block'] for r in fit}),holdout_temporal_blocks=len({r['block'] for r in hold}))
    if (len(fit)<8 or len(hold)<4 or len({r['block'] for r in fit})<4 or len({r['block'] for r in hold})<2 or
            len({r['object'] for r in hold}-{r['object'] for r in fit})<2):
        result['reason']='Independent lidar velocity/error observations are missing or lack held-out objects.';return result
    rotation=_rotation(parameters[0]);lever=prior-parameters[1:3]
    def difference(row):
        measured=row['independent_velocity'];relative=rotation@(measured['object']-measured['ego'])
        predicted=relative[0]+measured['yaw']*lever[1]
        return row['vrel']-float(predicted)
    bias=float(np.median([difference(r) for r in fit]))
    held=np.array([abs(difference(r)-bias) for r in hold])
    reference=[];lidar_bounds=[]
    for row in hold:
        v=row['independent_velocity'];bound=np.abs(rotation)@v['bound'];lidar_bounds.append(bound)
        reference.append(float(bound[0]+np.linalg.norm(lever)*v['yaw_bound']+
            abs(v['yaw'])*np.linalg.norm(relative_bound)+np.linalg.norm(v['object']-v['ego'])*yaw_bound))
    bound=float(np.max(held)+max(reference))
    result.update(velocity_bias_mps=bias,velocity_error_bound_mps=bound,
        heldout_max_residual_mps=float(np.max(held)),reference_error_bound_mps=max(reference),
        lidar_velocity_error_bound_mps=np.max(lidar_bounds,axis=0).tolist(),
        observed_domain={'range_m':[float(min(r['radar'][0]-prior[0] for r in hold)),float(max(r['radar'][0]-prior[0] for r in hold))],
                         'abs_bearing_rad':max(abs(math.atan2(r['radar'][1]-prior[1],r['radar'][0]-prior[0])) for r in hold),
                         'vrel_mps':[min(r['vrel'] for r in hold),max(r['vrel'] for r in hold)],
                         'ego_speed_mps':[min(r['speed'] for r in hold),max(r['speed'] for r in hold)],
                         'abs_yaw_rate_rps':max(abs(r['independent_velocity']['yaw']) for r in hold)},
        uncertainty_scope='Held-out maximum discrepancy plus independent registration/track/lever-arm error; no individual sensor accuracy claim.',
        static_scene_calibration=True)
    result['available']=bool(abs(bias)<=3 and 0<bound<=3 and all(x>0 for x in result['lidar_velocity_error_bound_mps']))
    if not result['available']:result['reason']='Held-out velocity bias or uncertainty is too broad for useful optional fusion.'
    return result


def _predict(parameters,lidar,speed,dt):
    return lidar@_rotation(parameters[0]).T+parameters[1:3]-np.column_stack((speed*(dt+parameters[3]),np.zeros(len(lidar))))


def _solve(rows,initial=None,delay_fixed=None,weights=None):
    l=np.array([r['lidar'] for r in rows]);q=np.array([r['radar'] for r in rows])
    v=np.array([r['speed'] for r in rows]);dt=np.array([r['dt'] for r in rows])
    counts=Counter(r['object'] for r in rows)
    base=np.array([1/counts[r['object']] for r in rows]);base/=max(float(np.mean(base)),1e-9)
    if weights is not None:base*=weights
    if initial is None:
        compensated=q+np.column_stack((v*dt,np.zeros(len(v))))
        theta=[];rng=np.random.default_rng(91)
        for _ in range(128):
            i,j=rng.integers(0,len(rows),2)
            delta=l[i]-l[j];other=compensated[i]-compensated[j]
            if abs(delta[0])<12:continue
            angle=math.atan2(float(delta[0]*other[1]-delta[1]*other[0]),float(delta@other))
            if abs(angle)<.175:theta.append(angle)
        angle=float(np.median(theta)) if theta else 0.
        translation=np.median(compensated-l@_rotation(angle).T,axis=0)
        initial=np.r_[angle,np.clip(translation,[-4,-1.8],[4,1.8]),0.]
    initial=np.asarray(initial,float).copy()
    if delay_fixed is not None:initial[3]=delay_fixed
    index=np.arange(4) if delay_fixed is None else np.arange(3)
    def unpack(p):
        result=initial.copy();result[index]=p;return result
    def residual(p):
        return ((_predict(unpack(p),l,v,dt)-q)/np.array([.6,.3])*np.sqrt(base)[:,None]).ravel()
    low=np.array([-.175,-5.,-2.,-.15]);high=np.array([.175,5.,2.,.15])
    solution=least_squares(residual,np.clip(initial[index],low[index]+1e-8,high[index]-1e-8),
                           bounds=(low[index],high[index]),loss='soft_l1',f_scale=1.,max_nfev=150)
    return unpack(solution.x),_predict(unpack(solution.x),l,v,dt)-q


def _radar(rows,metadata):
    summaries,rejected,prior=_pairs(rows,metadata)
    result={'status':'insufficient','eligible':False,'reasons':[],'blocks':len({r['block'] for r in summaries}),
            'object_blocks':len(summaries),'distinct_track_associations':len({r['object'] for r in summaries}),
            'rejected_rows':rejected,'origin_prior_m':prior.tolist(),'origin_prior_not_measured':True,
            'identity_note':'Distinct track associations are not guaranteed distinct physical objects.'}
    if len({r['block'] for r in summaries})<LIMITS['radar_blocks']:
        result['reasons'].append('Need at least twelve two-second blocks of stable, exclusive radar/lidar correspondences.');return result
    if len(summaries)>1200:
        result['reasons'].append('Calibration batch exceeds 1200 independent object/block summaries; use a smaller bounded capture window.');return result
    fit_blocks,hold_blocks=_split(r['block'] for r in summaries)
    fit=[r for r in summaries if r['block'] in fit_blocks];hold=[r for r in summaries if r['block'] in hold_blocks]
    fit_objects={r['object'] for r in fit};hold_objects={r['object'] for r in hold}
    if len(fit_objects)<4 or len(hold_objects-fit_objects)<2:
        result['reasons'].append('Need four fitted track associations and two newly encountered held-out associations.');return result
    ranges=np.array([r['lidar'][0] for r in fit]);span=float(np.ptp(np.percentile(ranges,[5,95])))
    byobject=defaultdict(list)
    for r in fit:byobject[r['object']].append(r['lidar'][0])
    within=[float(np.ptp(v)) for v in byobject.values()]
    result.update(fit_blocks=len(fit_blocks),holdout_blocks=len(hold_blocks),fit_tracks=len(fit_objects),
                  holdout_new_tracks=len(hold_objects-fit_objects),fit_range_span_m=span,
                  tracks_with_range_leverage=sum(v>=LIMITS['within_track_span_m'] for v in within))
    if span<LIMITS['range_span_m'] or sum(v>=LIMITS['within_track_span_m'] for v in within)<3:
        result['reasons'].append('Insufficient range leverage to separate yaw from lateral offset.');return result
    speeds=np.array([r['speed'] for r in fit]);speed_span=float(np.ptp(np.percentile(speeds,[5,95])))
    timing=metadata.get('relative_timing')
    fixed_delay=None;external_delay_bound=None
    if (isinstance(timing,dict) and timing.get('reference')=='radar_publication_relative_to_lidar' and
            timing.get('evidence_type')=='sensor_derived_empirical'):
        value=_number(timing.get('offset_s'));bound=_number(timing.get('error_bound_s'))
        if value is not None and bound is not None and abs(value)<=.15 and 0<=bound<=.04:
            fixed_delay=value;external_delay_bound=bound
    complete=all(r['complete_timing'] for r in fit+hold)
    joint_delay=fixed_delay is None and not complete and speed_span>=LIMITS['speed_span_for_delay_mps']
    unknown_delay=fixed_delay is None and not complete and not joint_delay
    if fixed_delay is None and not joint_delay:fixed_delay=0.
    parameters,residual=_solve(fit,delay_fixed=fixed_delay)
    # Fitted outliers are robustly excluded; held-out rows are NEVER filtered by
    # whether they agree with the proposal.
    inlier=np.all(np.abs(residual)<=np.array([1.0,.5]),axis=1)
    fraction=float(np.mean(inlier));result['fit_inlier_fraction']=fraction
    if fraction<.7 or len({r['object'] for r,k in zip(fit,inlier) if k})<4:
        result['reasons'].append('No strong common rigid alignment across independent target associations.');return result
    clean=[r for r,k in zip(fit,inlier) if k]
    parameters,residual=_solve(clean,parameters,delay_fixed=fixed_delay)
    jac=[]
    for row in clean:
        xy=_rotation(parameters[0])@row['lidar']
        jac.extend([[-xy[1],1.,0.,-row['speed']],[xy[0],0.,1.,0.]])
    jac=np.asarray(jac)[:,:4 if joint_delay else 3]
    column_norm=np.linalg.norm(jac,axis=0)
    singular=np.linalg.svd(jac/np.maximum(column_norm,1e-12),compute_uv=False)
    condition=float(singular[0]/max(singular[-1],1e-12))
    l=np.array([r['lidar'] for r in hold]);q=np.array([r['radar'] for r in hold])
    v=np.array([r['speed'] for r in hold]);dt=np.array([r['dt'] for r in hold])
    validation=_predict(parameters,l,v,dt)-q
    # Count a block once even if hundreds of repeated scans see the same object.
    hold_error=[]
    for block in sorted(hold_blocks):
        values=np.array([error for r,error in zip(hold,validation) if r['block']==block])
        if len(values):hold_error.append(np.max(np.abs(values),axis=0))
    hold_max=np.max(hold_error,axis=0)
    objects=sorted({r['object'] for r in clean});blocks=sorted({r['block'] for r in clean})
    rng=np.random.default_rng(2091);boot=[]
    for _ in range(96):
        object_counts=Counter(objects[i] for i in rng.integers(0,len(objects),len(objects)))
        block_counts=Counter(blocks[i] for i in rng.integers(0,len(blocks),len(blocks)))
        weights=np.array([object_counts[r['object']]*block_counts[r['block']] for r in clean],float)
        if np.count_nonzero(weights)<6:continue
        try:
            estimate,_=_solve(clean,parameters,delay_fixed=fixed_delay,weights=weights)
            boot.append(estimate)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):continue
    if len(boot)<48:
        result['reasons'].append('Insufficient independent bootstrap support.');return result
    bootstrap=np.max(np.abs(np.array(boot)-parameters),axis=0)
    extent=np.max([r['extent'] for r in clean+hold],axis=0)
    prior_bound=np.max([r['prior_bound'] for r in summaries],axis=0)
    timing_bound=max(r['timing'] for r in clean+hold)
    # Shared factory origin, target point offsets and unknown sensor delays do
    # not become smaller merely because more rows were collected.
    delay_bound=external_delay_bound if external_delay_bound is not None else .15 if unknown_delay else float(bootstrap[3]+.005)
    relative_bound=hold_max+bootstrap[1:3]+extent+np.array([.05,.05])
    relative_bound[0]+=max(r['speed'] for r in clean+hold)*(delay_bound+timing_bound)
    relative_yaw_bound=float(bootstrap[0]+math.radians(.05))
    axis_prior=_number(metadata.get('radar_axis_prior_error_bound_deg',.5))
    if axis_prior is None or not 0<=axis_prior<=5:axis_prior=.5
    common_yaw=math.radians(axis_prior)
    total_bound=relative_bound+prior_bound;total_yaw=relative_yaw_bound+common_yaw
    observed_range=float(min(max(r['lidar'][0] for r in clean),max(r['lidar'][0] for r in hold)))
    # This is an explicitly provisional geometry budget, not detection range.
    supported=max(0.,min(observed_range,(1.25-total_bound[1])/max(total_yaw,1e-6)))
    reasons=[]
    if condition>100:reasons.append('Yaw, translation or relative delay remain poorly separated by the available motion geometry.')
    if np.any(hold_max>[.75,.5]):reasons.append('Held-out target alignment drifts or mismatches exceed the bounded coarse-geometry model.')
    if relative_bound[0]>LIMITS['relative_x_bound_m']:reasons.append('Longitudinal alignment remains too uncertain, including target-point and timing ambiguity.')
    if relative_bound[1]>LIMITS['relative_y_bound_m']:reasons.append('Relative lateral alignment remains too uncertain.')
    if relative_yaw_bound>math.radians(LIMITS['relative_yaw_bound_deg']):reasons.append('Yaw bootstrap spread is too large.')
    if abs(parameters[0])>.16 or abs(parameters[1])>4.5 or abs(parameters[2])>1.5:reasons.append('Fitted transform is inconsistent with an approximately forward-facing roof mount.')
    if joint_delay and (abs(parameters[3])>.14 or delay_bound>.04):reasons.append('Relative delay is weakly identified or at the search boundary.')
    if unknown_delay:reasons.append('Constant-speed evidence cannot separate longitudinal origin from unknown relative sensor delay.')
    if supported<LIMITS['minimum_supported_range_m']:reasons.append('Combined geometry uncertainty supports less than 15 m under the provisional corridor budget.')
    usable=['relative_yaw','relative_lateral_translation'] if (hold_max[1]<=.5 and relative_bound[1]<=.75 and relative_yaw_bound<=math.radians(1)) else []
    if not unknown_delay and relative_bound[0]<=1.25 and hold_max[0]<=.75:usable.append('relative_longitudinal_translation')
    result.update(status='empirical_limited_alignment' if not reasons else 'partial_or_unstable',eligible=not reasons,reasons=reasons,
        yaw_rad=float(parameters[0]),translation_m=parameters[1:3].tolist(),relative_delay_s=float(parameters[3]),
        relative_time_offset_s=float(parameters[3]),
        relative_time_offset_error_bound_s=float(delay_bound),
        time_offset_convention='effective radar time = radar publication + offset, relative to mapped lidar midpoint',
        relative_delay_error_bound_s=float(delay_bound),delay_method='externally supplied empirical timing' if external_delay_bound is not None else
            'joint static-target/speed nuisance fit' if joint_delay else 'complete input timing' if complete else 'unobservable origin-delay combination',
        timing_is_absolute_ground_truth=False,fit_speed_span_mps=speed_span,holdout_max_residual_m=hold_max.tolist(),
        scaled_design_condition=condition,
        bootstrap_max_parameter_spread=bootstrap.tolist(),bootstrap_replicates=len(boot),
        target_return_systematic_bound_m=extent.tolist(),relative_fit_error_bound_m=relative_bound.tolist(),
        relative_yaw_error_bound_rad=relative_yaw_bound,origin_prior_error_bound_m=prior_bound.tolist(),
        axis_prior_error_bound_rad=common_yaw,total_translation_error_bound_m=total_bound.tolist(),
        total_yaw_error_bound_rad=total_yaw,usable_components=usable,
        geometry_only_supported_range_m=supported,geometry_only_two_second_speed_mps=supported/2.,
        warning_horizon_s=2.,detection_range_validated=False,
        timing_note='Joint delay is an empirical combined alignment parameter, not independent proof of either sensor clock or latency.')
    result['velocity_calibration']=_velocity_support(clean,hold,parameters,relative_bound,relative_yaw_bound,prior)
    valid_z=[r['z_bounds'] for r in hold if r.get('z_bounds') is not None]
    result['planar_validation']={'heldout_object_blocks':len(hold),'heldout_temporal_blocks':len(hold_blocks),
        'heldout_new_track_associations':len(hold_objects-fit_objects),
        'lidar_z_coverage_available':len(valid_z)==len(hold),
        'lidar_reference_z_m':[min(z[0] for z in valid_z),max(z[1] for z in valid_z)] if valid_z else None,
        'projection_error_basis':'actual held-out XY residuals in the fixed leveling frame, not inferred road-normal angle times range',
        'projection_reference':'ground-leveled lidar to nominal OEM radar frame',
        'coverage_scope':'observed target surfaces and ranges only; other heights/ranges are not certified'}
    return result


def fit_alignment(ground_observations,radar_pairs,metadata):
    """Return a reproducible, bounded empirical proposal without mutating input.

    metadata must identify the current mount epoch. Historical rows from another
    mount are rejected even if their numerical fit is attractive.
    """
    metadata=metadata if isinstance(metadata,dict) else {}
    ground_rows=list(ground_observations or []);pairs=list(radar_pairs or [])
    if len(ground_rows)>100_000 or len(pairs)>200_000:
        return {'schema_version':SCHEMA_VERSION,'eligible':False,'evidence_type':'sensor_derived_empirical',
                'not_surveyed':True,'reasons':['Input exceeds bounded calibration batch size.'],'mount_epoch':_epoch(metadata.get('mount_epoch'))}
    reference,reference_error=_reference_rotation(ground_rows+pairs,metadata)
    if reference is not None:metadata={**metadata,'fixed_level_rotation':reference.tolist()}
    ground=_ground(ground_rows,metadata);radar=_radar(pairs,metadata)
    if reference_error:
        ground['eligible']=False;ground.setdefault('reasons',[]).append(reference_error)
        radar['eligible']=False;radar.setdefault('reasons',[]).append(reference_error)
    if ground.get('provisional_road_relative_candidate'):
        planar=radar.get('planar_validation') or {}
        if (reference_error is None and radar.get('eligible') is True and planar.get('lidar_z_coverage_available') is True and
                planar.get('heldout_new_track_associations',0)>=2 and planar.get('heldout_temporal_blocks',0)>=2):
            ground.update(eligible=True,status='provisional_road_relative',reasons=[],
                planar_projection_support='independent held-out radar/lidar XY agreement in the same fixed leveling frame',
                required_runtime_ground_sources=['fit','held'],mount_prior_fallback_allowed=False)
    relative_yaw=radar.get('yaw_rad');translation=radar.get('translation_m',[None,None])
    h=ground.get('height_m');roll=ground.get('roll_deg');pitch=ground.get('pitch_deg')
    bound=radar.get('total_translation_error_bound_m',[None,None]);hbound=ground.get('height_error_bound_m')
    rpbound=ground.get('roll_pitch_error_bound_deg') or [None,None]
    ybound=radar.get('total_yaw_error_bound_rad')
    usable=list(radar.get('usable_components',[]))
    if ground.get('eligible'):usable.extend(['height_above_observed_road','conditional_road_roll_pitch'])
    geometry={'yaw_rad':relative_yaw,'sensor_origin_camera_flu_m':[*translation,h],
              'translation_error_bound_m':[*bound,hbound],'yaw_error_bound_rad':ybound,
              'roll_pitch_rad':[math.radians(v) if v is not None else None for v in (roll,pitch)],
              'roll_pitch_error_bound_rad':[math.radians(v) if v is not None else None for v in rpbound],
              'usable_components':usable,'frame':'camera_ground_flu_with_explicit_nominal_radar_prior',
              'geometry_basis':ground.get('geometry_basis','unsupported_road_relative_geometry'),
              'requires_observed_ground':ground.get('requires_observed_ground',False),
              'roll_pitch_uncertainty_scope':'observed road/body plane variation, not isolated fixed mount rotation error',
              'planar_projection_error_basis':(radar.get('planar_validation') or {}).get('projection_error_basis'),
              'planar_projection_validation':radar.get('planar_validation')}
    # The extractor uses the shortest normal-leveling rotation. Compose that
    # exact convention with fitted yaw, rather than silently replacing it with
    # an Euler leveling sequence that introduces a small extra heading rotation.
    mount_angles=[roll,pitch,math.degrees(relative_yaw) if relative_yaw is not None else None]
    if relative_yaw is not None and ground.get('level_rotation_sensor_to_road') is not None:
        z=np.eye(3);z[:2,:2]=_rotation(relative_yaw)
        rotation=z@np.asarray(ground['level_rotation_sensor_to_road'])
        mount_angles=np.degrees([math.atan2(rotation[2,1],rotation[2,2]),
                                math.asin(float(np.clip(-rotation[2,0],-1,1))),
                                math.atan2(rotation[1,0],rotation[0,0])]).tolist()
    result={'schema_version':SCHEMA_VERSION,'method':'independent_block_robust_static_correspondence_v1',
            'evidence_type':'sensor_derived_empirical','not_surveyed':True,'ground_truth':False,
            'mount_epoch':_epoch(metadata.get('mount_epoch')),'eligible':bool(ground.get('eligible') and radar.get('eligible')),
            'geometry_basis':geometry['geometry_basis'],'requires_observed_ground':geometry['requires_observed_ground'],
            'ground':ground,'radar_alignment':radar,'geometry':geometry,
            'mount':{'xyz':geometry['sensor_origin_camera_flu_m'],
                     'rpy_deg':mount_angles},
            'uncertainty':{'position_m':geometry['translation_error_bound_m'],
                           'rpy_deg':[*rpbound,math.degrees(ybound) if ybound is not None else None],
                           'roll_pitch_scope':geometry['roll_pitch_uncertainty_scope']},
            'radar':{'origin_m':[*radar['origin_prior_m'],0.],'yaw_rad':0.,
                     'origin_reference':'planar ground projection, not measured physical radar height',
                     'origin_prior_not_measured':True,'position_error_bound_m':radar.get('origin_prior_error_bound_m',[1.,.35]),
                     'relative_fit_error_bound_m':radar.get('relative_fit_error_bound_m'),
                     'relative_time_offset_s':radar.get('relative_time_offset_s'),
                     'time_offset_convention':radar.get('time_offset_convention'),
                     'relative_delay_s':radar.get('relative_delay_s'),'relative_delay_error_bound_s':radar.get('relative_delay_error_bound_s')},
            'reasons':ground.get('reasons',[])+radar.get('reasons',[]),
            'metrics':{'ground_input_rows':len(ground_rows),'radar_input_rows':len(pairs),'block_duration_s':2.,
                       'limits':LIMITS},
            'source_ids':sorted({r['source_id'] for r in ground_rows+pairs if isinstance(r,dict) and isinstance(r.get('source_id'),str)}),
            'session_ids':sorted({r['session_id'] for r in ground_rows+pairs if isinstance(r,dict) and isinstance(r.get('session_id'),str)}),
            'limits':['No survey or target measurements; alignment is conditional on road/body and OEM-radar frame assumptions.',
                      'Geometric support is not a demonstrated obstacle detection or warning distance.',
                      'Only this mount epoch may use the proposal; explicit application and monitoring are required.']}
    payload=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    result['proposal_id']=hashlib.sha256(payload.encode()).hexdigest()
    return result

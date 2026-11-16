"""Forced-mode crossing evidence; in-path boxes have no added age requirement."""
import numpy as np

STABLE_EDGE_ERROR_M=.40
STABLE_VELOCITY_ABS_MPS=2.5
STABLE_VELOCITY_REL=.30


def inherited_motion_contradiction(track):
    """Withdraw inherited velocity's veto power after a fresh rigid contradiction.

    This is not a stationary-object classification or replacement measurement.
    Missing/ambiguous/new observations and changing visible footprints do not
    acquire this exception. The stored filtered velocity is never modified.
    """
    result={'contradicted':False}
    if (getattr(track,'misses',1) or getattr(track,'association_ambiguous',True) or
        getattr(track,'detection_index',None) is None or getattr(track,'motion_samples',0)<1 or
        getattr(track,'hits',0)<3 or getattr(track,'age_s',0.)<.2):return result
    try:
        measured=np.asarray(track.measured_velocity,float);filtered=np.asarray(track.velocity,float)
        span=np.asarray(track.hi,float)[:2]-np.asarray(track.lo,float)[:2]
        edge=float(track.motion_edge_error_m)
        if (measured.shape!=(2,) or filtered.shape!=(2,) or span.shape!=(2,) or
            not np.isfinite(np.r_[measured,filtered,span,edge]).all() or
            np.any(span<0) or np.max(span)>8. or not 0<=edge<=STABLE_EDGE_ERROR_M):return result
        from far_objects import MOVING_TRACK_MPS
        measured_speed=float(np.linalg.norm(measured));filtered_speed=float(np.linalg.norm(filtered))
        difference=float(np.linalg.norm(filtered-measured))
        tolerance=max(STABLE_VELOCITY_ABS_MPS,STABLE_VELOCITY_REL*measured_speed)
        if not (measured_speed<MOVING_TRACK_MPS<=filtered_speed and difference>tolerance):return result
        return {'contradicted':True,'basis':'fresh_measurement_contradicts_inherited_moving_velocity',
                'filtered_velocity_mps':filtered.tolist(),'fresh_measured_velocity_mps':measured.tolist(),
                'velocity_disagreement_mps':difference,'existing_consistency_tolerance_mps':tolerance,
                'footprint_edge_error_m':edge,'fresh_motion_samples':int(track.motion_samples),
                'stationary_classification':False,'filtered_velocity_modified':False}
    except (AttributeError,TypeError,ValueError,OverflowError):return result

def observe(track,detection,movement,angle,dt):
    from fcw_engine import rotate2
    previous=track.observed_position
    if previous is not None and .025<=dt<=.25:
        stationary=rotate2(np.asarray(previous)-movement,-angle)
        measured=(detection['center']-stationary)/dt
        corners=np.array([[x,y] for x in (track.lo[0],track.hi[0]) for y in (track.lo[1],track.hi[1])])
        moved=rotate2(corners-movement,-angle);delta=detection['center']-stationary
        edge_error=max(float(np.linalg.norm(detection['lo'][:2]-moved.min(0)-delta)),float(np.linalg.norm(detection['hi'][:2]-moved.max(0)-delta)))
        prior=track.measured_velocity
        difference=float(np.linalg.norm(measured-rotate2(np.asarray(prior),-angle))) if prior is not None else float('inf')
        speed=float(np.linalg.norm(measured));valid=bool(np.isfinite(measured).all() and speed<60 and not track.association_ambiguous)
        strong=valid and edge_error<=.20 and difference<=max(1.5,.20*speed)
        stable=valid and edge_error<=STABLE_EDGE_ERROR_M and difference<=max(STABLE_VELOCITY_ABS_MPS,STABLE_VELOCITY_REL*speed)
        track.strong_motion_s=track.strong_motion_s+dt if strong else 0.
        track.stable_motion_s=track.stable_motion_s+dt if stable else 0.
        track.motion_samples=track.motion_samples+1 if valid else 0;track.motion_edge_error_m=edge_error
        track.measured_velocity=measured.copy() if valid else None
    else:
        track.strong_motion_s=track.stable_motion_s=0.;track.motion_samples=0;track.measured_velocity=None
    track.observed_position=detection['center'].copy()

def reset(track):
    track.observed_position=None;track.measured_velocity=None;track.motion_samples=0;track.strong_motion_s=track.stable_motion_s=0.

def evidence(track):
    fast=track.motion_samples>=2 and track.strong_motion_s>=.09
    mature=track.hits>=10 and track.age_s>=.9 and track.stable_motion_s>=.5
    result={'trusted':bool((fast or mature) and not track.misses and not track.association_ambiguous),
        'basis':'rigid_observation_agreement' if fast else 'sustained_observation_agreement' if mature else 'unknown',
        'samples':track.motion_samples,'strong_consistency_s':track.strong_motion_s,'sustained_consistency_s':track.stable_motion_s,
        'edge_disagreement_m':track.motion_edge_error_m,'unknown_is_not_stationary':True}
    return result


def resolve_motion_evidence(track,context,raw_contradiction=None):
    """Resolve the existing sensor agreement before withdrawing motion authority.

    Association/vision-selection/promotion predicates below are the unchanged
    forced-mode rules. Their resolved evidence is shared by history retirement,
    initial threat assessment and the later relative-motion policy.
    """
    from radar_fusion import associate
    from fcw_engine import THREE_SENSOR_MIN_PROB,THREE_SENSOR_MIN_MPS,THREE_SENSOR_TOL_MPS
    motion=evidence(track)
    assoc=associate(track,context.get('radar_points') or [],float(context.get('speed_mps') or 0.),bool(motion['trusted']))
    vl=None
    for ld in (context.get('vision_leads') or []):
        if ld.get('prob',0.)>=.5 and abs(ld['x']-float(track.lo[0]))<5. and abs(ld['y']-float(track.position[1]))<2.:
            if vl is None or abs(ld['x']-float(track.lo[0]))<abs(vl['x']-float(track.lo[0])):vl=ld
    vision=({**vl,'dx_m':round(vl['x']-float(track.lo[0]),2)} if vl else None)
    if not motion['trusted'] and assoc and not assoc.get('stationary') and vl and vl.get('prob',0.)>=THREE_SENSOR_MIN_PROB and abs(vision['dx_m'])<=2.:
        v=np.asarray(track.velocity,float);rg=float(assoc.get('ground_speed_mps') or 0.);vv=float(vl.get('v') or 0.)
        if min(rg,vv,float(v[0]))>=THREE_SENSOR_MIN_MPS and max(rg,vv,float(v[0]))-min(rg,vv,float(v[0]))<=THREE_SENSOR_TOL_MPS and abs(float(v[1]))<.8:
            motion={**motion,'trusted':True,'basis':'radar_vision_lidar_agreement'}
    raw=inherited_motion_contradiction(track) if raw_contradiction is None else raw_contradiction
    authority=raw
    if raw['contradicted']:
        # The old in-path policy already uses this associated radar's ground
        # speed when motion is trusted. A changing visible lidar patch cannot
        # silently remove that existing independent moving evidence.
        corroborated=bool(motion['trusted'] and assoc and not assoc.get('stationary'))
        if corroborated:
            authority={**raw,'contradicted':False,'lidar_contradiction':True,
                       'resolution':'existing_trusted_motion_with_associated_moving_radar',
                       'existing_motion_basis':motion['basis'],
                       'radar_id':assoc['point']['id'],'radar_ground_speed_mps':assoc['ground_speed_mps'],
                       'vision_lead':vision,'new_trust_rule':False}
        else:
            motion={**motion,'trusted':False,'basis':raw['basis'],
                    'observation_consistency_trusted':bool(motion['trusted']),
                    'inherited_motion_contradiction':raw}
    return {'authority':authority,'evidence':motion,'radar':assoc,'vision_lead':vision}

def occupies_path(track,context,mount,max_range=40.):
    """Intersect observed AABB with the path corridor, including object extent.

    Extend the last tangent only for occupancy classification to the range cap;
    the actual warning still uses the original threat horizon. This preserves
    the immediate route for distant in-path oncoming objects.
    """
    path=np.asarray(context.get('path',[]),float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():return False
    d=path[-1]-path[-2];norm=np.linalg.norm(d)
    if norm>1e-6:
        extra=max(0.,max_range-float(np.linalg.norm(path[-1])))
        if extra:path=np.vstack((path,path[-1]+d/norm*extra))
    lo=np.asarray(track.lo[:2]);hi=np.asarray(track.hi[:2]);center=(lo+hi)/2;half=np.maximum(.15,(hi-lo)/2)
    d=np.diff(path,axis=0);length=np.linalg.norm(d,axis=1);good=length>1e-6
    if not good.any():return False
    tangent=d[good]/length[good,None];normal=np.c_[-tangent[:,1],tangent[:,0]]
    relative=center-path[:-1][good];along=np.sum(relative*tangent,axis=1);cross=np.abs(np.sum(relative*normal,axis=1))
    xhalf=np.abs(tangent)@half;yhalf=np.abs(normal)@half
    return bool(np.any((cross<=mount.width_m/2+.25+yhalf)&(along+xhalf>=-1.)&(along-xhalf<=length[good]+mount.bumper_x_m)))

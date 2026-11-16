"""Derive optional radar support from an applied ordinary-drive report.

No survey booleans are fabricated. Missing velocity/timing identifiability only
disables optional radar, never the separately supported lidar FCW corridor.
The runtime owns report/profile integrity checks and explicit Apply.
"""
import copy
import math
from object_fusion import default_fusion_calibration


def _finite(v):return type(v) in (int,float) and math.isfinite(v)
def _vector(v,n):return isinstance(v,(list,tuple)) and len(v)==n and all(_finite(x) for x in v)


def build_fusion_profile(report,empirical_profile):
    p=empirical_profile if isinstance(empirical_profile,dict) else {}
    r=report if isinstance(report,dict) else {}
    out=default_fusion_calibration(p.get('mount_epoch'))
    out.update(eligible=False,reasons=[],evidence_basis='sensor_derived_empirical',surveyed_ground_truth=False)
    def fail(reason):
        out['reasons'].append(reason)
        return out
    if (r.get('eligible') is not True or p.get('method')!='sensor_derived_empirical' or
            p.get('surveyed_ground_truth') is not False or not p.get('id') or p.get('id')!=r.get('id') or
            not p.get('mount_epoch') or p['mount_epoch']!=r.get('mount_epoch')):
        return fail('Current mounting-period empirical report/profile identity is unavailable.')
    if r.get('qualification_path')=='independent_lidar':
        return fail('Independent lidar profile remains usable without radar; cross-frame radar fusion has not been qualified for this profile.')
    alignment=r.get('alignment') or {};a=alignment.get('radar_alignment') or {}
    motion=r.get('motion_analysis') or {};velocity=a.get('velocity_calibration') or {}
    if alignment.get('eligible') is not True or a.get('eligible') is not True:
        return fail('Relative radar/lidar geometry is not supported by this drive.')
    if velocity.get('available') is not True:
        return fail('Independent held-out radar/lidar velocity agreement is unavailable; lidar FCW remains independent.')
    offset=a.get('relative_time_offset_s');offset_bound=a.get('relative_time_offset_error_bound_s',a.get('relative_delay_error_bound_s'))
    if (a.get('delay_method')=='unobservable origin-delay combination' or not _finite(offset) or
            not _finite(offset_bound) or not 0<offset_bound<=.04):
        return fail('Radar-relative timing and longitudinal origin are not separately identified.')
    old_adjustment=r.get('time_reference_adjustment_s');new_adjustment=p.get('lidar_time_adjustment_s')
    timing=p.get('timing') or {};motion_offset=timing.get('offset_s');motion_bound=timing.get('error_bound_s')
    if (not all(_finite(x) for x in (old_adjustment,new_adjustment,motion_offset,motion_bound)) or
            timing.get('available') is not True or not 0<motion_bound<=.1):
        return fail('Corrected lidar/motion timeline lacks an empirical error bound.')
    correction=new_adjustment-old_adjustment
    if abs(correction-motion_offset)>1e-9:
        return fail('Applied lidar time correction differs from the report motion-reference offset.')
    # In the fitting timeline delta = radar publication - old lidar midpoint.
    # If lidar time moves by C, radar effective time must also move by C to
    # preserve the fitted relative interval. This says nothing about individual
    # radar hardware latency, whose physical clock is not separately observed.
    effective_offset=offset+correction;bound=offset_bound+motion_bound
    if abs(effective_offset)>1 or bound>.15:
        return fail('Combined empirical relative timing is too broad for bounded optional radar association.')
    origin=a.get('origin_prior_m');origin_bound=a.get('origin_prior_error_bound_m')
    position=a.get('relative_fit_error_bound_m');angle=a.get('relative_yaw_error_bound_rad')
    axis=a.get('axis_prior_error_bound_rad');lidar_velocity=velocity.get('lidar_velocity_error_bound_mps')
    vbound=velocity.get('velocity_error_bound_mps');vbias=velocity.get('velocity_bias_mps');domain=velocity.get('observed_domain')
    if (not _vector(origin,2) or not _vector(origin_bound,2) or any(x<0 for x in origin_bound) or
            not _vector(position,2) or any(x<=0 for x in position) or not _finite(angle) or not 0<=angle<=.05 or
            not _finite(axis) or not 0<=axis<=.1 or not _vector(lidar_velocity,2) or any(not 0<x<=20 for x in lidar_velocity) or
            not _finite(vbound) or not 0<vbound<=3 or not _finite(vbias) or abs(vbias)>3 or not isinstance(domain,dict)):
        return fail('Empirical geometry or independently observed velocity bounds are incomplete.')
    for key in ('range_m','vrel_mps','ego_speed_mps'):
        if not _vector(domain.get(key),2) or domain[key][0]>domain[key][1]:
            return fail('Observed radar velocity domain is malformed.')
    if (not _finite(domain.get('abs_bearing_rad')) or not 0<=domain['abs_bearing_rad']<=1.6 or
            not _finite(domain.get('abs_yaw_rate_rps')) or not 0<=domain['abs_yaw_rate_rps']<=.2):
        return fail('Observed radar bearing/turning coverage is unavailable.')
    evidence=r.get('digest')
    if not isinstance(evidence,str) or len(evidence)!=64:
        return fail('Bound calibration report identity is unavailable.')
    out['lidar']={'uncertainty_verified':False,'empirical_uncertainty_supported':True,
        'evidence_basis':'sensor_derived_empirical','evidence_id':evidence,
        'position_error_bound_m':list(position),'velocity_error_bound_mps':list(lidar_velocity),
        'position_error_scope':'Combined relative association error is allocated here once, not separately to both sensor positions.'}
    out['radar']={'calibration_epoch':p['mount_epoch'],'evidence_id':evidence,
        'evidence_basis':'sensor_derived_empirical','surveyed_ground_truth':False,
        'empirical_support':{'relative_geometry':True,'relative_timing':True,'velocity':True},
        'extrinsics_verified':False,'timing_verified':False,'uncertainty_verified':False,'semantics_verified':False,
        'origin_m':[*origin,0.],'origin_error_bound_m':0.,'position_error_bound_m':[0.,0.],
        'position_error_scope':'Relative residual/translation error is carried once in lidar; shared factory frame uncertainty cancels in paired association.',
        'yaw_rad':0.,'yaw_error_bound_rad':angle,
        'time_offset_s':effective_offset,'time_error_bound_s':bound,
        'time_basis':'relative_to_corrected_lidar_time','individual_sensor_latency_verified':False,
        'fitted_relative_offset_s':offset,'lidar_reference_shift_s':correction,
        'applied_profile_id':p['id'],'lidar_time_adjustment_s':new_adjustment,
        'velocity_semantics':'longitudinal_relative','velocity_bias_mps':vbias,
        'velocity_error_bound_mps':vbound,'observed_domain':copy.deepcopy(domain),
        'velocity_reference':'Independent lidar object-minus-ego twist and held-out native OEM vRel; comma speed was not the teacher.'}
    out['common_reference']={'group':evidence,'origin_prior_m':list(origin),
        'translation_error_bound_m':list(origin_bound),'yaw_error_bound_rad':axis,
        'basis':'nominal OEM/camera reference, not surveyed',
        'association_handling':'Common origin/axis uncertainty is not added independently to both sensor positions.',
        'corridor_handling':'Must remain in vehicle-reference/footprint/corridor uncertainty; not erased by relative association.'}
    out['eligible']=True
    out['limits']=['Radar is optional and cannot veto lidar alerts.',
                   'Camera remains diagnostic until separately supported.',
                   'Velocity support is restricted to the observed range, bearing, closing-speed and motion domain.',
                   'Empirical consistency is not ground truth or physical warning-range qualification.']
    return out

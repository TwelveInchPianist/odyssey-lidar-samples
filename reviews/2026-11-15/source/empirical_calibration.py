"""Bounded runtime use of ordinary-drive calibration, explicitly empirical.

Surveyed commissioning remains a separate contract. This module never invents
reference measurements or silently applies newly estimated detector parameters.
"""
import copy,hashlib,json,math,re
from pathlib import Path
import numpy as np
from corridor import MODEL_PATH_REASONS
from fcw_engine import Mount

VERSION='ordinary-drive-calibration-profile-v1'
MAX_HORIZON_S=2.
MAX_TIMING_S=.10
MAX_LATERAL_M=1.
MAX_LONGITUDINAL_M=4.5

def finite(x):return type(x) in (int,float) and math.isfinite(x)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def vector(v,n):return isinstance(v,(list,tuple)) and len(v)==n and all(finite(x) for x in v)

def from_report(report,config,folder):
    """Return an unapplied, provenance-bound profile with no surveyed claims."""
    from selfcal import VERSION as REPORT_VERSION
    if not isinstance(report,dict) or not re.fullmatch(r'[0-9a-f]{32}',str(report.get('id',''))):
        raise ValueError('Invalid calibration proposal identity')
    if report.get('algorithm')!=REPORT_VERSION or report.get('eligible') is not True:
        raise ValueError('This drive has not produced an eligible empirical calibration')
    if not config.get('mount_epoch') or report.get('mount_epoch')!=config['mount_epoch']:
        raise ValueError('This report belongs to a different mounting period')
    if report.get('digest')!=digest({k:v for k,v in report.items() if k!='digest'}):
        raise ValueError('Calibration report checksum changed')
    a=report['alignment'];m=report['motion_analysis']
    mount=copy.deepcopy(config['mount']);mount.update(copy.deepcopy(a['mount']))
    # Keep the legacy stored flags honest. Runtime promotes an effective copy
    # only after this entire empirical profile and its source report are checked.
    mount.update(measured=False,alignment_verified=False);Mount(**mount).validate()
    p={'schema_version':1,'version':VERSION,'id':report['id'],'mount_epoch':config['mount_epoch'],
       'method':'sensor_derived_empirical','surveyed_ground_truth':False,
       'mount':mount,'geometry':copy.deepcopy(a['geometry']),
       'geometry_basis':a.get('geometry_basis','conditional_road_level_estimate'),
       'requires_observed_ground':bool(a.get('requires_observed_ground',False)),
       'timing':copy.deepcopy(m['timing']),'motion':copy.deepcopy(m['motion']),
       'corridor':copy.deepcopy(m['corridor']),
       'lidar_time_adjustment_s':float(report.get('time_reference_adjustment_s',0))+m['timing']['offset_s'],
       'source':{'folder':Path(folder).name,'session_id':report['session_id'],
                 'applied_report':'calibration/empirical/'+report['id']+'.json',
                 'report_digest':report['digest'],'files':copy.deepcopy(report['source_files'])},
       'detector_thresholds_changed':False,'parameter_activation':'explicit_apply',
       'limits':{'max_horizon_s':MAX_HORIZON_S,'max_timing_s':MAX_TIMING_S,
                 'max_lateral_m':MAX_LATERAL_M,'max_longitudinal_m':MAX_LONGITUDINAL_M}}
    from empirical_fusion import build_fusion_profile
    p['fusion_profile']=build_fusion_profile(report,p)
    p['digest']=digest(p)
    check=validate(p)
    if not check['eligible']:raise ValueError('; '.join(check['reasons']))
    p['runtime_preview']=preview(p)
    if not p['runtime_preview']['has_supported_conditions']:
        raise ValueError('Estimated geometry, timing and motion errors do not yet support a usable short warning corridor; more consistent sensor observations are needed')
    p['digest']=digest({k:v for k,v in p.items() if k!='digest'})
    return p

def preview(profile):
    """Calculate feasible conditions; these are not extra field observations."""
    from empirical_corridor import prepare_context
    m=profile['motion'];rows=[];at=100_000_000_000
    lo=max(1.,m.get('min_observed_speed_mps',1.));hi=m['max_verified_speed_mps']
    if hi<lo:return {'has_supported_conditions':False,'conditions':[],'physical_warning_validation':False}
    for speed in np.linspace(lo,hi,5):
        for yaw in (0.,min(.05,m['max_verified_yaw_rate_rps']),-min(.05,m['max_verified_yaw_rate_rps'])):
            ctx={'at_ns':at,'speed_mps':float(speed)+m['speed_bias_mps'],'yaw_rate_rps':yaw+m['yaw_bias_rps'],
                 'acceleration_mps2':0.,'ready':False,'reasons':['low model confidence'],'motion_ready':True,
                 'speed_source_ns':at,'motion_source_ns':at,'steering_source_ns':at,'steering_valid':True,
                 'steering_angle_deg':0.,'steering_rate_deg_s':0.,'clock_bound_s':.005,'path':np.empty((0,2))}
            q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':profile['timing']['error_bound_s']}
            out=prepare_context(ctx,profile,q)
            if out.get('independent_corridor',{}).get('active'):
                rows.append({'speed_mps':float(speed),'yaw_rate_rps':yaw,'horizon_s':out['warning_horizon_s'],
                             'geometric_forward_m':out['path_horizon_m']})
    return {'has_supported_conditions':bool(rows),'conditions':rows,'physical_warning_validation':False,
            'scope':'Calculated steady-motion feasibility with fresh sources and5ms clock bound; not a detection-range measurement'}

def validate(profile):
    reasons=[];p=profile if isinstance(profile,dict) else {}
    if p.get('version')!=VERSION or p.get('schema_version')!=1:reasons.append('unknown empirical calibration version')
    if p.get('method')!='sensor_derived_empirical' or p.get('surveyed_ground_truth') is not False:reasons.append('empirical evidence basis missing')
    if (p.get('geometry_basis') in ('provisional_road_relative','independent_lidar_bounded_prior') and p.get('requires_observed_ground') is not True):
        reasons.append('provisional road-relative geometry requires observed ground')
    if not re.fullmatch(r'[0-9a-f]{32}',str(p.get('id',''))):reasons.append('invalid empirical profile identity')
    try:
        if p.get('digest')!=digest({k:v for k,v in p.items() if k!='digest'}):reasons.append('empirical profile checksum changed')
        Mount(**p['mount']).validate()
    except (KeyError,ValueError,TypeError):reasons.append('empirical geometry invalid')
    g=p.get('geometry',{});m=p.get('motion',{});t=p.get('timing',{});c=p.get('corridor',{})
    g=g if isinstance(g,dict) else {};m=m if isinstance(m,dict) else {}
    t=t if isinstance(t,dict) else {};c=c if isinstance(c,dict) else {}
    if (not vector(g.get('sensor_origin_camera_flu_m'),3) or not vector(g.get('translation_error_bound_m'),3) or
        any(x<0 or x>6 for x in g.get('translation_error_bound_m',[]))):reasons.append('alignment estimate or uncertainty missing')
    if not finite(g.get('yaw_error_bound_rad')) or not 0<=g['yaw_error_bound_rad']<=math.radians(5):reasons.append('yaw alignment uncertainty too large')
    if t.get('available') is not True or not finite(t.get('error_bound_s')) or not 0<t['error_bound_s']<=MAX_TIMING_S:reasons.append('relative timing needs a repeatable bound within100ms for this limited mode')
    if not finite(p.get('lidar_time_adjustment_s')) or abs(p['lidar_time_adjustment_s'])>1:reasons.append('lidar time adjustment invalid')
    from empirical_kinematics import lever
    try:lever(m)
    except ValueError:reasons.append('invalid empirical lateral motion model or lever')
    if m.get('available') is not True:reasons.append('independent motion comparison unavailable')
    for key,limit in (('speed_bias_mps',5),('yaw_bias_rps',.2),('speed_error_bound_mps',5),('yaw_error_bound_rps',.2)):
        if not finite(m.get(key)) or abs(m[key])>limit:reasons.append('invalid empirical '+key)
        elif 'error_bound' in key and m[key]<=0:reasons.append('empirical '+key+' must be positive')
    if not finite(m.get('lateral_velocity_error_bound_mps')) or not 0<m['lateral_velocity_error_bound_mps']<=10:reasons.append('lateral velocity error bound is unavailable')
    if not finite(m.get('max_verified_speed_mps')) or not 1<=m['max_verified_speed_mps']<=55:reasons.append('driven speed coverage unavailable')
    if not finite(m.get('max_verified_yaw_rate_rps')) or not 0<=m['max_verified_yaw_rate_rps']<=1.5:reasons.append('driven turning coverage unavailable')
    if c.get('available') is not True or not isinstance(c.get('horizons'),list):reasons.append('held-out relative trajectory coverage unavailable')
    elif any(not isinstance(row,dict) for row in c['horizons']):reasons.append('invalid held-out horizon record')
    elif not any(row.get('supported') is True and finite(row.get('horizon_s')) and 0<row['horizon_s']<=MAX_HORIZON_S for row in c['horizons']):reasons.append('no supported short warning horizon')
    return {'eligible':not reasons,'reasons':list(dict.fromkeys(reasons)),'method':'sensor_derived_empirical','surveyed_ground_truth':False}

def check_active(config,record_root):
    """Verify the small applied artifact and unchanged source metadata at start."""
    p=config.get('empirical_calibration');check=validate(p)
    if not check['eligible']:return {'trusted':False,**check}
    reasons=[]
    if p.get('mount_epoch')!=config.get('mount_epoch'):reasons.append('mounting period changed')
    if p.get('mount')!=config.get('mount'):reasons.append('saved geometry changed after calibration Apply')
    try:
        source=p['source'];folder=Path(record_root)/source['folder']
        if folder.resolve().parent!=Path(record_root).resolve() or folder.is_symlink() or getattr(folder,'is_junction',lambda:False)():raise ValueError('source folder outside recording root')
        path=(Path(record_root).resolve().parent/source['applied_report']).resolve()
        expected=Path(record_root).resolve().parent/'calibration'/'empirical'/(p['id']+'.json')
        if path!=expected or path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('invalid applied calibration artifact path')
        if path.stat().st_size>2*1024*1024:raise ValueError('calibration report exceeds startup budget')
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('digest')!=source['report_digest'] or r['digest']!=digest({k:v for k,v in r.items() if k!='digest'}):raise ValueError('calibration evidence changed')
        files=source.get('files')
        if (not isinstance(files,dict) or not {'session.json','final-state.json','frame-contexts.jsonl'}<=set(files) or
            not any(re.fullmatch(r'lidar-[0-9]+\.pcap',name) for name in files)):
            raise ValueError('Applied calibration has no complete source recording manifest')
        for name,identity in files.items():
            file=folder/name
            if Path(name).name!=name or file.is_symlink() or getattr(file,'is_junction',lambda:False)():raise ValueError('invalid source file path')
            st=file.stat()
            if (st.st_size,st.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('source recording changed')
        from selfcal_capture import _fingerprints
        if _fingerprints(folder,None)!=files:raise ValueError('source metadata or bounded raw-file digest changed')
        original=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
        if (original.get('capture_mode')!='live' or original.get('synthetic_fixture') or
            original.get('session_id')!=source.get('session_id') or original.get('mount_epoch')!=p['mount_epoch']):
            raise ValueError('Applied profile is not bound to a physical source session in this mounting period')
        expected_profile=from_report(r,config,folder)
        if expected_profile['digest']!=p['digest']:
            raise ValueError('Applied calibration parameters differ from their immutable evidence report')
    except (OSError,KeyError,TypeError,ValueError) as e:reasons.append(str(e))
    return {'trusted':not reasons,'eligible':not reasons,'reasons':reasons,
            'calibration_id':p['digest'] if not reasons else None,'method':'sensor_derived_empirical',
            'source_integrity_scope':'applied report hash and original-source size/mtime; no full-PCAP rehash each startup'}

def effective_config(config):
    """Caller must have passed check_active. Original stored flags are unchanged."""
    p=config['empirical_calibration'];out=copy.deepcopy(config)
    out['mount']=dict(p['mount'],measured=True,alignment_verified=True)
    out.update(timing_verified=True,lidar_timing_bound_s=p['timing']['error_bound_s'],
               lidar_time_adjustment_s=p['lidar_time_adjustment_s'],calibration_basis='sensor_derived_empirical')
    return out

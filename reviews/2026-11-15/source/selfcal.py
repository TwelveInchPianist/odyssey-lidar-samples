"""Ordinary-drive empirical calibration: produce evidence, never self-apply.

This is a separate evidence contract from surveyed commissioning. Unknown
components stay unknown. Reports identify the actual source mounting period,
bounded analysis coverage and the assumptions behind each proposed parameter.
"""
import copy, hashlib, json, math, uuid
import runtime_clock as time
from pathlib import Path
from offline_jobs import check_work
from hindsight import write_json

VERSION='ordinary-drive-selfcal-v1'
REPORT='self-calibration.json'
CAPTURE='self-calibration-observations.json'
MAX_REPORT_BYTES=16*1024*1024

def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _hash(value):return hashlib.sha256(_json(value)).hexdigest()
def _finite(value):return type(value) in (int,float) and math.isfinite(value)
def _epoch(value):return value.get('id') if isinstance(value,dict) else value if isinstance(value,str) and value else None

def run_session(folder,data_root=None,cancel=None,progress=None,budget=None):
    """Cancellable post-drive analysis; original raw files are read-only."""
    from selfcal_capture import analyze_session,_fingerprints,VERSION as CAPTURE_VERSION
    from selfcal_alignment import fit_alignment
    from selfcal_motion import fit_motion
    from runtime import APP_VERSION
    folder=Path(folder).resolve();check_work(cancel)
    cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    if cfg.get('capture_mode')!='live' or cfg.get('synthetic_fixture'):
        return {'status':'unavailable','reason':'A completed physical lidar recording is needed','eligible':False}
    if not (folder/'final-state.json').is_file():raise ValueError('Finish recording before calibration analysis')
    sources=_fingerprints(folder,cancel)
    metadata={'mount_epoch':_epoch(cfg.get('mount_epoch')),'session_id':cfg.get('session_id'),
              'configuration':cfg,'evidence_kind':'sensor_derived_empirical','surveyed_ground_truth':False}
    capture=None;cache=folder/CAPTURE
    if cache.is_file() and cache.stat().st_size<=MAX_REPORT_BYTES:
        try:
            old=json.loads(cache.read_text(encoding='utf-8'))
            capture_digest=old.get('capture_digest') if isinstance(old,dict) else None
            intact=(isinstance(capture_digest,str) and capture_digest==_hash({k:v for k,v in old.items() if k!='capture_digest'}))
            if (intact and old.get('source_files')==sources and old.get('selfcal_version')==VERSION and
                old.get('algorithm')==CAPTURE_VERSION and old.get('producer_version')==APP_VERSION and
                old.get('status')=='complete' and old.get('requested_budget')==budget):capture=old
        except (OSError,ValueError):pass
    if capture is None:
        if progress:progress({'stage':'calibration','message':'Reading raw lidar and comma observations'})
        capture=analyze_session(folder,cancel,progress,budget=budget)
        capture['selfcal_version']=VERSION
        capture['producer_version']=APP_VERSION
        capture['requested_budget']=budget
        capture['capture_digest']=_hash({k:v for k,v in capture.items() if k!='capture_digest'})
        check_work(cancel)
        if len(_json(capture))>MAX_REPORT_BYTES:raise ValueError('Calibration observations exceed the bounded report budget')
        write_json(cache,capture)
    check_work(cancel)
    metadata.update(fixed_level_rotation=capture.get('fixed_level_rotation'),reference_quality=capture.get('reference_quality'))
    radar_alignment=fit_alignment(capture.get('ground_observations',[]),capture.get('radar_pairs',[]),metadata)
    from selfcal_lidar_geometry import fit_lidar_geometry
    metadata['lidar_geometry_prior']=cfg.get('lidar_geometry_prior')
    independent_alignment=fit_lidar_geometry(capture.get('ground_observations',[]),capture.get('motion_rows',[]),metadata)
    # New reports qualify lidar independently. Missing priors remain explicit;
    # a radar fit never silently supplies authority for this path.
    # Previously saved legacy profiles retain their original evidence contract.
    alignment=radar_alignment if cfg.get('calibration_geometry_mode')=='legacy_radar_derived' else independent_alignment
    motion=fit_motion(capture.get('motion_rows',[]),capture.get('comma_rows',[]),alignment=alignment)
    reasons=[]
    if capture.get('status')!='complete':
        reasons.append('Calibration extraction is incomplete; no profile can be approved until every selected fitting/holdout window has been processed.')
    if not metadata['mount_epoch']:reasons.append('This recording has no mounting-period identity; start a new mounting period before collecting current calibration')
    if not alignment.get('eligible'):reasons+=alignment.get('reasons') or ['Not enough consistent geometry/alignment observations']
    if not motion.get('eligible'):reasons+=motion.get('reasons') or ['More independently observed motion/timing variation is needed']
    report={'schema_version':1,'algorithm':VERSION,'id':uuid.uuid4().hex,
        'status':'candidate_ready' if not reasons else 'collecting','eligible':not reasons,
        'producer_version':APP_VERSION,
        'source_id':capture.get('source_id'),'session_id':capture.get('session_id'),
        'mount_epoch':metadata['mount_epoch'],'source_files':sources,
        'time_reference_adjustment_s':cfg.get('lidar_time_adjustment_s',0.),
        'alignment':alignment,'independent_lidar_alignment':independent_alignment,'radar_alignment_analysis':radar_alignment,
        'qualification_path':'legacy_radar_derived_geometry' if cfg.get('calibration_geometry_mode')=='legacy_radar_derived' else 'independent_lidar',
        'motion_analysis':motion,'coverage':{**capture.get('coverage',{}),
            'extraction_status':capture.get('status'),'budget_reason':capture.get('budget_reason')},
        'reference_quality':capture.get('reference_quality'),
        'reasons':list(dict.fromkeys(reasons)),'evidence_kind':'sensor_derived_empirical',
        'surveyed_ground_truth':False,'settings_changed':False,'requires_explicit_apply':True,
        'created_wall_ns':time.time_ns(),
        'limitations':capture.get('limitations',[])+[
            'Calibration consistency is not a measurement of obstacle recall or false-warning rate.',
            'The first empirical corridor is limited to the observed speed/turn conditions and at most two seconds.',
            'Candidate changes never apply while recording and never tune detector thresholds automatically.']}
    if capture.get('status')=='budget_limited':
        report['limitations'].append('Analysis reached its bounded processing budget; unexamined data is not evidence of absence.')
        if not report['eligible']:
            report['reasons'].append('Analysis budget reached before all selected data was examined; the remaining raw data has been preserved for a larger offline analysis.')
    check_work(cancel)
    if sources!=_fingerprints(folder,cancel):raise ValueError('Source recording changed during calibration fitting')
    report['digest']=_hash(report)
    if report['eligible']:
        try:
            from empirical_calibration import from_report
            candidate=from_report(report,cfg,folder)
            report['runtime_preview']=candidate['runtime_preview']
        except (ValueError,KeyError,TypeError,OverflowError) as error:
            report.update(eligible=False,status='collecting')
            report['reasons'].append(str(error))
        report['digest']=_hash({k:v for k,v in report.items() if k!='digest'})
    write_json(folder/REPORT,report)
    return {k:report[k] for k in ('status','eligible','session_id','mount_epoch','reasons','coverage','digest','settings_changed')}

def latest_report(record_root,mount_epoch):
    """Small bounded UI read. Old mount reports remain viewable in their sessions."""
    epoch=_epoch(mount_epoch)
    candidates=[]
    for folder in sorted(Path(record_root).glob('*'),reverse=True)[:100]:
        if folder.is_symlink() or getattr(folder,'is_junction',lambda:False)():continue
        path=folder/REPORT
        if not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_REPORT_BYTES:continue
        try:
            report=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(report,dict):continue
            if report.get('algorithm')!=VERSION or report.get('mount_epoch')!=epoch or not epoch:continue
            digest=report.get('digest');payload={k:v for k,v in report.items() if k!='digest'}
            if not isinstance(digest,str) or digest!=_hash(payload):continue
            candidates.append((folder,report))
        except (ValueError,OSError,TypeError):continue
    if not candidates:return None
    # New contradictory data must remain visible instead of being hidden behind
    # an older ready proposal. An already applied profile is tracked separately.
    folder,report=candidates[0]
    return {'folder':folder.name,'path':str(folder/REPORT),'report':report}

def status(record_root,config):
    found=latest_report(record_root,config.get('mount_epoch'))
    active=config.get('empirical_calibration') or {}
    if not isinstance(active,dict):active={}
    proof={'trusted':False,'reasons':[]}
    if active:
        from empirical_calibration import check_active
        proof=check_active(config,record_root)
    result={'method':'ordinary_drive_sensor_calibration','active_profile_id':active.get('id') if proof.get('trusted') else None,
        'active_profile_integrity':proof,
        'mount_epoch':_epoch(config.get('mount_epoch')),'automatic_analysis':config.get('self_calibration_enabled',True),
        'state':'ready_to_record','requires_manual_measurements':False,'proposal':None,
        'message':'Record normal drives with lidar and comma connected. Analysis estimates geometry, timing and supported motion coverage.'}
    if found:
        r=found['report'];result.update(state=r['status'],proposal={
            'id':r['id'],'digest':r['digest'],'folder':found['folder'],'eligible':r['eligible'],
            'alignment':r['alignment'],'motion_analysis':r['motion_analysis'],'reasons':r['reasons'],
            'runtime_preview':r.get('runtime_preview'),
            'coverage':r['coverage'],'surveyed_ground_truth':False},
            message='Calibration candidate ready for review and Apply.' if r['eligible'] else '; '.join(r['reasons']))
    return result

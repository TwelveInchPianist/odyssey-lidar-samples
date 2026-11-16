"""Deterministic paired FCW evidence report; never enables/approves live warnings.

Input is explicitly annotated replay evidence, not inferred truth from silence,
driver reactions, map persistence, or the detector's own outputs. Each variant
must have consumed the same ordered frame identities. This small comparison
contract can be fed by existing replay runners without creating a second exam.
"""
import math

VARIANTS=('lidar_only','lidar_radar','lidar_camera','map_augmented')
TARGET_CLASSES=('vehicle','barrier','substantial_debris')
LIGHTING=('day','night')

def _number(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum:
        raise ValueError(name+' must be a finite number >= '+str(minimum))
    return float(value)

def _optional(value,name):return None if value is None else _number(value,name)

def _measurement(raw,case_id):
    if not isinstance(raw,dict):raise ValueError('Missing case measurement: '+case_id)
    detected=raw.get('detected');warned=raw.get('warned')
    if type(detected) is not bool or type(warned) is not bool:raise ValueError('Detection/warning status must be explicit booleans')
    if warned and not detected:raise ValueError('Warned case must be detected')
    distance=_optional(raw.get('warning_distance_m'),'warning_distance_m')
    ttc=_optional(raw.get('warning_ttc_s'),'warning_ttc_s')
    if not warned and (distance is not None or ttc is not None):raise ValueError('Silent case cannot claim warning distance/timing')
    return {'detected':detected,'warned':warned,'warning_distance_m':distance,'warning_ttc_s':ttc}

def compare_variants(cases,variants,exposure):
    """Compare four named variants using identical frame/case identities.

    cases: [{id, obstacle_class, source:'physical'|'synthetic', lighting,
      label:'hazard'|'nonhazard'|'unknown', annotation_verified, held_out,
      speed_mps, stationary, first_visible_distance_m, first_visible_ttc_s,
      height_m, width_m}]. Debris needs measured dimensions for milestone scope.
    variants: {name: {frame_ids:[stable ordered identities], cases:{id:{detected,
      warned, warning_distance_m, warning_ttc_s}}, urgent_events:[{id,lighting,
      label:'false'|'true'|'unknown', annotation_verified}]}}.
    exposure: {day|night:{hours, held_out, quiet_intervals_reviewed,
      all_urgent_events_reviewed}}. Hours are physical held-out driving only.

    Evaluation returns evidence, deficiencies, and regression failures. It does
    not alter settings, bless calibration, or certify real-world recall.
    """
    if set(variants)!=set(VARIANTS):raise ValueError('All four named variants are required')
    by_id={}
    for case in cases:
        cid=case.get('id')
        if not isinstance(cid,str) or not cid or cid in by_id:raise ValueError('Unique case identities required')
        if case.get('source') not in ('physical','synthetic'):raise ValueError('Case source must distinguish physical/synthetic')
        if case.get('label') not in ('hazard','nonhazard','unknown'):raise ValueError('Explicit case label required')
        if case.get('lighting') not in LIGHTING:raise ValueError('Day/night case cohort required')
        _number(case.get('speed_mps'),'speed_mps')
        for field in ('height_m','width_m','first_visible_distance_m','first_visible_ttc_s'):_optional(case.get(field),field)
        by_id[cid]=case
    frame_ids=None;measurements={}
    for name in VARIANTS:
        raw=variants[name];frames=raw.get('frame_ids')
        if not isinstance(frames,list) or any(not isinstance(x,str) or not x for x in frames) or len(set(frames))!=len(frames):
            raise ValueError('Unique ordered frame identities are required')
        if frame_ids is None:frame_ids=list(frames)
        elif frames!=frame_ids:raise ValueError('Variant frames/order differ: '+name)
        if set(raw.get('cases',{}))!=set(by_id):raise ValueError('Variant cases differ: '+name)
        if by_id and not frames:raise ValueError('Case results require replay frame identities')
        measurements[name]={cid:_measurement(raw['cases'][cid],cid) for cid in by_id}

    baseline=measurements['lidar_only'];reports={}
    for name in VARIANTS:
        rows=[];regressions=[];physical=[];synthetic=[]
        for cid,case in by_id.items():
            found=measurements[name][cid];old=baseline[cid]
            verified=case.get('annotation_verified') is True and case['label']=='hazard'
            in_scope=case.get('obstacle_class') in TARGET_CLASSES
            if case.get('obstacle_class')=='substantial_debris':
                in_scope=case.get('height_m') is not None and case['height_m']>=.5 and case.get('width_m') is not None and case['width_m']>=1.
            required_distance=125. if case['speed_mps']>=31.0 else 4*case['speed_mps']
            visible_distance=case.get('first_visible_distance_m');visible_ttc=case.get('first_visible_ttc_s')
            early_visible=bool(visible_distance is not None and visible_ttc is not None and visible_distance>=required_distance and visible_ttc>=4)
            opportunity='early_visible' if early_visible else 'visibility_unknown' if visible_distance is None or visible_ttc is None else 'late_visible_or_occluded'
            eligible=verified and in_scope and case.get('stationary') is True and early_visible
            timely=None
            if eligible:
                timely=bool(found['warned'] and found['warning_ttc_s'] is not None and found['warning_ttc_s']>=4 and
                            found['warning_distance_m'] is not None and found['warning_distance_m']>=required_distance)
            row={'id':cid,'obstacle_class':case.get('obstacle_class'),'lighting':case['lighting'],
                 'source':case['source'],'held_out':case.get('held_out') is True,
                 'verified_hazard':verified,'milestone_class':bool(in_scope),'visibility_opportunity':opportunity,
                 'required_warning_distance_m':required_distance,'required_warning_ttc_s':4.,
                 'detected':found['detected'],'warned':found['warned'],'timely_warning':timely,
                 'warning_distance_m':found['warning_distance_m'],'warning_ttc_s':found['warning_ttc_s'],
                 'miss':verified and not found['detected'],'warning_miss':verified and not found['warned'],
                 'late_warning':timely is False and found['warned']}
            rows.append(row)
            if verified:(physical if case['source']=='physical' else synthetic).append(row)
            # Even out-of-milestone child/bicycle/animal cases remain regression
            # obligations if their labels are established. False alerts are not
            # protected as if they were genuine target warnings.
            if verified:
                if old['detected'] and not found['detected']:regressions.append({'id':cid,'reason':'new_detection_miss'})
                if old['warned'] and not found['warned']:regressions.append({'id':cid,'reason':'lost_warning'})
                if old['warned'] and found['warned']:
                    for field in ('warning_distance_m','warning_ttc_s'):
                        if old[field] is not None and (found[field] is None or found[field]+1e-9<old[field]):
                            regressions.append({'id':cid,'reason':'shortened_or_unmeasured_'+field,'baseline':old[field],'candidate':found[field]})
        # Count independently labeled urgent events by ID, never per frame.
        events=variants[name].get('urgent_events',[])
        if not isinstance(events,list):raise ValueError('urgent_events must be a list')
        event_ids=set();cohorts={};deficiencies=[]
        for event in events:
            if not isinstance(event.get('id'),str) or not event['id'] or event['id'] in event_ids:raise ValueError('Unique urgent-event identities required')
            event_ids.add(event['id'])
            if event.get('lighting') not in LIGHTING or event.get('label') not in ('false','true','unknown'):raise ValueError('Urgent event needs day/night and explicit label')
        for light in LIGHTING:
            exp=exposure.get(light,{})
            hours=_number(exp.get('hours',0),'exposure hours')
            cohort=[e for e in events if e['lighting']==light]
            false=sum(e['label']=='false' and e.get('annotation_verified') is True for e in cohort)
            unknown=sum(e['label']=='unknown' or e.get('annotation_verified') is not True for e in cohort)
            qualified=bool(hours>=5 and exp.get('held_out') is True and exp.get('quiet_intervals_reviewed') is True and exp.get('all_urgent_events_reviewed') is True and unknown==0)
            rate=false/hours if hours>0 else None
            cohorts[light]={'held_out_hours':hours if exp.get('held_out') is True else 0.,'false_urgent_events':false,
                'unreviewed_urgent_events':unknown,'false_urgent_per_hour':rate,'exposure_qualified':qualified,
                'nuisance_target_met':qualified and rate<=1.,
                'verified_detection_misses':sum(r['miss'] and r['lighting']==light for r in physical),
                'verified_warning_misses':sum(r['warning_miss'] and r['lighting']==light for r in physical),
                'verified_late_warnings':sum(r['late_warning'] and r['lighting']==light for r in physical)}
            if not qualified:deficiencies.append('insufficient_held_out_reviewed_'+light+'_exposure')
            elif rate>1.:deficiencies.append(light+'_false_urgent_rate_above_target')
        # At minimum require a real, held-out early-visible stationary target of
        # each scoped class in each lighting cohort at the stated 70 mph target.
        # This is a coverage minimum, not a claim of statistically proven recall.
        missing=[]
        for kind in TARGET_CLASSES:
            for light in LIGHTING:
                subset=[r for r in physical if r['held_out'] and r['obstacle_class']==kind and r['lighting']==light and
                        by_id[r['id']]['speed_mps']>=31.0 and r['timely_warning'] is not None]
                if not subset:missing.append(kind+':'+light)
        if missing:deficiencies.append('missing_physical_70mph_class_and_lighting_evidence')
        target_failures=[r['id'] for r in physical if r['held_out'] and r['timely_warning'] is False]
        if target_failures:deficiencies.append('physical_warning_distance_or_time_target_failed')
        if not frame_ids:deficiencies.append('no_paired_replay_frames')
        regression_status='fail' if regressions else 'pass' if physical or synthetic else 'insufficient_evidence'
        if regression_status=='insufficient_evidence':deficiencies.append('no_verified_regression_hazards')
        reports[name]={'regression_status':regression_status,'regressions':regressions,'cases':rows,
            'physical_verified_hazard_count':len(physical),'synthetic_verified_hazard_count':len(synthetic),
            'cohorts':cohorts,'missing_physical_coverage':missing,'target_failures':target_failures,
            'milestone_status':'fail' if regressions or target_failures or any(c['exposure_qualified'] and not c['nuisance_target_met'] for c in cohorts.values()) else 'insufficient_evidence' if deficiencies else 'tested_targets_met',
            'deficiencies':deficiencies,
            'label_note':'Candidate reactions, detector silence and synthetic success do not establish physical recall.'}
    return {'schema_version':1,'frame_count':len(frame_ids or []),'paired_inputs':True,
        'baseline':'lidar_only','targets':{'warning_ttc_s':4.,'stationary_range_at_70mph_m':125.,
            'false_urgent_per_hour_max':1.,'minimum_held_out_hours_per_lighting':5.},
        'variants':reports,'self_apply':False,
        'limitations':'Finite annotated replay evidence is not universal recall, closed-loop driving validation, or stopping capability. No automatic promotion.'}

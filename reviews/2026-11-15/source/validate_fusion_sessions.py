"""Compare independently replayed FCW variants without manufacturing labels.

Usage: validate_fusion_sessions.py --ledger FILE [--labels FILE] --output DIR

Ledger JSONL rows: {frame_id, at_ns, variants:{lidar_only, lidar_radar,
lidar_camera, map_augmented}}. Values are actual independently run engine
results with tracks/detections/alerts. Each row contains all four variants.

Optional human labels JSON: {schema_version:1, cases:[CASE], exposure:{day:...,
night:...}, urgent_events:{VARIANT:{EVENT_ID:{lighting,label,
annotation_verified,annotation_source}}}}. CASE carries fusion_validation case
metadata, annotation_source:'manual'|'controlled'|'synthetic_fixture', and
frames:[{frame_id,distance_m,ttc_s,associations:{VARIANT:{track_ids:[],
detection_indices:[]}}}]. Empty associations explicitly record a reviewed miss;
missing associations are incomplete evidence, never an inferred miss or pass.
Urgent EVENT_ID is '<frame_id>|alert:<zero-based index>'. Physical annotation
must be manual/controlled; synthetic_fixture is accepted only for synthetic
cases. Exposure assertions must describe held-out physical driving.

Original ledgers/labels are read-only. Output must be a new directory so a
rerun cannot overwrite original data or a prior review.
"""
import argparse,copy,html,json,math
from pathlib import Path
from fusion_validation import compare_variants,VARIANTS


def _jsonable_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def read_ledger(path):
    rows=[];ids=set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():continue
            row=json.loads(line)
            identity=row.get('frame_id')
            if not isinstance(identity,str) or not identity or identity in ids:raise ValueError('Missing/duplicate frame identity at line '+str(number))
            if not _jsonable_number(row.get('at_ns')) or row['at_ns']<0:raise ValueError('Invalid at_ns at line '+str(number))
            if set(row.get('variants',{}))!=set(VARIANTS):raise ValueError('Four independently replayed variants required at line '+str(number))
            for name in VARIANTS:
                result=row['variants'][name]
                if not isinstance(result,dict):raise ValueError('Variant output must be an engine result')
                for key in ('tracks','alerts'):
                    if key in result and not isinstance(result[key],list):raise ValueError('Variant '+key+' must be a list')
                detections=result.get('detections',0)
                if not isinstance(detections,list) and not (type(detections) is int and detections>=0):raise ValueError('Variant detections must be a list or nonnegative count')
            ids.add(identity);rows.append(row)
    return rows


def _allowed_annotation(item,source='physical'):
    allowed=('manual','controlled') if source=='physical' else ('synthetic_fixture',)
    return item.get('annotation_verified') is True and item.get('annotation_source') in allowed


def build_report(rows,labels=None):
    labels=labels or {'schema_version':1}
    if labels.get('schema_version')!=1:raise ValueError('Unsupported labels schema')
    frame_ids=[row['frame_id'] for row in rows];by_frame={row['frame_id']:row for row in rows}
    order={identity:i for i,identity in enumerate(frame_ids)}
    if len(by_frame)!=len(rows):raise ValueError('Duplicate frame identities')
    variants={name:{'frame_ids':frame_ids,'cases':{},'urgent_events':[]} for name in VARIANTS}
    accepted=[];incomplete=[];counts={}
    for case in labels.get('cases',[]):
        cid=case.get('id');frames=case.get('frames')
        if not isinstance(cid,str) or not cid:raise ValueError('Labeled case identity required')
        if not isinstance(frames,list) or not frames:
            incomplete.append({'id':cid,'reason':'no_frame_associations'});continue
        if not _allowed_annotation(case,case.get('source')):
            incomplete.append({'id':cid,'reason':'not_independent_verified_annotation'});continue
        complete=True;seen=set()
        for frame in frames:
            identity=frame.get('frame_id')
            if identity not in by_frame:raise ValueError('Case refers to absent frame: '+str(identity))
            if identity in seen:raise ValueError('Case repeats frame: '+identity)
            seen.add(identity)
            assoc=frame.get('associations')
            if not isinstance(assoc,dict) or set(assoc)!=set(VARIANTS):complete=False;break
            if any(not isinstance(value,dict) or 'track_ids' not in value or not isinstance(value['track_ids'],list) for value in assoc.values()):complete=False;break
        if not complete:
            incomplete.append({'id':cid,'reason':'missing_variant_target_association'});continue
        metadata={k:v for k,v in case.items() if k!='frames'};accepted.append(metadata)
        sorted_frames=sorted(frames,key=lambda frame:order[frame['frame_id']])
        for name in VARIANTS:
            detected=False;first_warning=None
            for frame in sorted_frames:
                result=by_frame[frame['frame_id']]['variants'][name]
                assoc=frame['associations'][name];track_ids=assoc['track_ids']
                if any(not isinstance(tid,(int,str)) or isinstance(tid,bool) for tid in track_ids):raise ValueError('Invalid associated track ID')
                actual_ids={track['id'] for track in result.get('tracks',[]) if 'id' in track}
                if any(tid not in actual_ids for tid in track_ids):raise ValueError('Annotation refers to absent track in '+name+': '+frame['frame_id'])
                detection_indices=assoc.get('detection_indices',[])
                detection_output=result.get('detections',[])
                if not isinstance(detection_indices,list) or (detection_indices and not isinstance(detection_output,list)) or any(type(index) is not int or index<0 or index>=len(detection_output) for index in detection_indices):
                    raise ValueError('Annotation refers to invalid detection index')
                detected=detected or bool(track_ids or detection_indices)
                if first_warning is None and any(alert.get('track_id') in track_ids for alert in result.get('alerts',[])):
                    first_warning=frame
            variants[name]['cases'][cid]={'detected':detected,'warned':first_warning is not None,
                'warning_distance_m':first_warning.get('distance_m') if first_warning else None,
                'warning_ttc_s':first_warning.get('ttc_s') if first_warning else None}

    unreviewed_any=False
    for name in VARIANTS:
        event_labels=labels.get('urgent_events',{}).get(name,{})
        actual_events=[];urgent_ids=[];states={};missing_lighting=0;unreviewed=0;detections=0;tracks=0;map_effects=0
        nonurgent=0;severity_unknown=0;map_available=0;map_used=0;map_cells_max=0
        for row in rows:
            result=row['variants'][name];state=result.get('status','unknown');states[state]=states.get(state,0)+1
            detection_output=result.get('detections',0)
            detections+=len(detection_output) if isinstance(detection_output,list) else detection_output
            tracks+=len(result.get('tracks',[]))
            map_prior=result.get('map_prior') or {}
            if not isinstance(map_prior,dict):map_prior={}
            map_available+=int(map_prior.get('status')=='available')
            map_used+=int(map_prior.get('ranking_used') is True)
            cells=map_prior.get('coverage_cells')
            if type(cells) is int and cells>=0:map_cells_max=max(map_cells_max,cells)
            if 'ranked_tracks' in map_prior:
                evidence_rows=map_prior['ranked_tracks'] if isinstance(map_prior['ranked_tracks'],list) else []
            else:evidence_rows=[track.get('map_evidence') or {} for track in result.get('tracks',[])]
            for evidence in evidence_rows:
                if isinstance(evidence,dict):map_effects+=int(_jsonable_number(evidence.get('evidence_delta')) and evidence['evidence_delta']!=0)
            for index,alert in enumerate(result.get('alerts',[])):
                event_id=row['frame_id']+'|alert:'+str(index);actual_events.append(event_id)
                if alert.get('severity')=='warning':
                    nonurgent+=1;continue
                if alert.get('severity')!='urgent':
                    severity_unknown+=1;unreviewed+=1
                    if event_labels.get(event_id,{}).get('lighting') not in ('day','night'):missing_lighting+=1
                    continue
                urgent_ids.append(event_id)
                annotation=event_labels.get(event_id,{})
                verified=_allowed_annotation(annotation)
                if not verified or annotation.get('label') not in ('false','true'):unreviewed+=1
                lighting=annotation.get('lighting')
                if lighting not in ('day','night'):
                    missing_lighting+=1;continue
                variants[name]['urgent_events'].append({'id':event_id,'lighting':lighting,
                    'label':annotation.get('label','unknown') if verified else 'unknown','annotation_verified':verified})
        extras=set(event_labels)-set(actual_events)
        if extras:raise ValueError('Urgent annotation refers to absent event in '+name+': '+next(iter(extras)))
        unreviewed_any=unreviewed_any or unreviewed>0 or missing_lighting>0
        counts[name]={'frames':len(rows),'algorithm_alert_events':len(actual_events),
            'algorithm_alert_ids':actual_events,'unreviewed_alert_events':unreviewed,
            'urgent_alert_events':len(urgent_ids),'urgent_alert_ids':urgent_ids,
            'nonurgent_warning_events':nonurgent,'severity_unknown_events':severity_unknown,
            'alerts_without_lighting':missing_lighting,'detections_across_frames':detections,
            'tracks_across_frames':tracks,'frames_by_status':states,'map_track_priority_changes':map_effects,
            'map_available_frames':map_available,'map_available_frame_percent':100*map_available/len(rows) if rows else None,
            'map_priority_used_frames':map_used,'map_max_local_coverage_cells':map_cells_max,
            'map_coverage_note':'Available localization frames and local coverage cells are not guaranteed hazard-detection coverage.'}
    exposure=copy.deepcopy(labels.get('exposure',{}))
    # Review flags cannot hide unknown alerts omitted for lack of day/night ID.
    if unreviewed_any:
        for light in ('day','night'):exposure.setdefault(light,{})['all_urgent_events_reviewed']=False
    comparison=compare_variants(accepted,variants,exposure)
    if incomplete:
        for result in comparison['variants'].values():
            result['deficiencies'].append('incomplete_target_annotations')
            if result['milestone_status']=='tested_targets_met':result['milestone_status']='insufficient_evidence'
    return {'schema_version':1,'comparison':comparison,'ledger_summary':counts,
        'incomplete_target_annotations':incomplete,
        'evidence_status':'human_annotations_supplied' if accepted else 'no_verified_target_annotations',
        'audio_note':'Algorithm alert events in offline replay do not establish that any sound played in the vehicle.',
        'label_note':'Empty alert lists and reaction-derived candidate labels are not ground truth; current uncommissioned sessions may legitimately remain inactive.'}


def render_html(report):
    escape=lambda value:html.escape(str(value))
    comparison=report['comparison'];rows=[]
    for name,result in comparison['variants'].items():
        count=report['ledger_summary'][name]
        rows.append('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in (name,result['milestone_status'],count['algorithm_alert_events'],
            result['physical_verified_hazard_count'],len(result['regressions']),
            result['cohorts']['day']['false_urgent_per_hour'],result['cohorts']['night']['false_urgent_per_hour'],
            result['cohorts']['day']['verified_warning_misses']+result['cohorts']['night']['verified_warning_misses']))+'</tr>')
    detail=[]
    for name,result in comparison['variants'].items():
        detail.append('<h2>'+escape(name)+'</h2><p>'+escape(', '.join(result['deficiencies']) or 'Finite annotated test targets met; no automatic promotion.')+'</p>')
        maps=report['ledger_summary'][name]
        detail.append('<p>Map localization available: '+escape(maps['map_available_frames'])+' frames; priority used: '+escape(maps['map_priority_used_frames'])+
            ' frames; nonzero track priorities: '+escape(maps['map_track_priority_changes'])+'; maximum local coverage cells: '+escape(maps['map_max_local_coverage_cells'])+
            '. '+escape(maps['map_coverage_note'])+'</p>')
        for case in result['cases']:
            detail.append('<p>'+escape(case['id'])+': '+escape(case['visibility_opportunity'])+'; detection '+escape(case['detected'])+
                          '; warning '+escape(case['warned'])+'; distance '+escape(case['warning_distance_m'])+' m; TTC '+escape(case['warning_ttc_s'])+
                          ' s; source '+escape(case['source'])+'</p>')
    return '<!doctype html><html><head><meta charset="utf-8"><title>FCW fusion comparison</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#182330}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #bbc5cf;text-align:left}th{background:#eaf0f6}p{line-height:1.5}</style></head><body>'+\
        '<h1>FCW fusion comparison</h1><p>'+escape(report['evidence_status'])+'. '+escape(comparison['frame_count'])+' paired source frames.</p>'+\
        '<p>Targets: four seconds; approximately 125 m at 70 mph; no more than one false urgent event/hour separately in day and night. Five reviewed held-out hours are required in each lighting cohort.</p>'+\
        '<p><strong>False alarms and miss evidence are shown together. Unknown evidence cannot pass.</strong></p>'+\
        '<table><tr><th>Variant</th><th>Evidence result</th><th>All algorithm alerts</th><th>Physical labeled hazards</th><th>Regressions</th><th>False urgent/hour day</th><th>False urgent/hour night</th><th>Verified warning misses</th></tr>'+''.join(rows)+'</table>'+\
        '<p>'+escape(report['audio_note'])+'</p><p>'+escape(report['label_note'])+'</p>'+''.join(detail)+\
        '<h2>Incomplete annotations</h2><pre>'+escape(json.dumps(report['incomplete_target_annotations'],indent=2))+'</pre>'+\
        '<p>'+escape(comparison['limitations'])+'</p></body></html>'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ledger',required=True);parser.add_argument('--labels');parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    ledger=Path(args.ledger).resolve();labels_path=Path(args.labels).resolve() if args.labels else None
    output=Path(args.output).resolve()
    if output.exists():raise ValueError('Output directory must be new; existing evidence is never overwritten')
    labels=json.loads(labels_path.read_text(encoding='utf-8-sig')) if labels_path else None
    report=build_report(read_ledger(ledger),labels)
    report['inputs']={'ledger':str(ledger),'labels':str(labels_path) if labels_path else None}
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (output/'comparison.html').write_text(render_html(report),encoding='utf-8')
    print(json.dumps({'output':str(output),'frames':report['comparison']['frame_count'],
        'status':{name:result['milestone_status'] for name,result in report['comparison']['variants'].items()}}))
    return 0


if __name__=='__main__':raise SystemExit(main())

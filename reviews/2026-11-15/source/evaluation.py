"""Automatic session audit. Unlabeled data is not called an accuracy test."""
from collections import Counter
import html
import json
import math
from pathlib import Path
from hindsight import analyze_session,input_signature,VERSION as HINDSIGHT_VERSION
from offline_jobs import check_work,WorkPaused

def summarize(session,recompute_hindsight=True,cancel=None):
    session=Path(session);file=session/'decisions.jsonl'
    metadata=json.loads((session/'session.json').read_text(encoding='utf-8')) if (session/'session.json').exists() else {}
    final=json.loads((session/'final-state.json').read_text(encoding='utf-8')) if (session/'final-state.json').exists() else {}
    count=0;first=last=None;reasons=Counter();alerts=[];eligible=0;ground=[];ground_sources=Counter();coverage_frames=Counter();timings={};timing_unknown=0
    restoration=Counter();maximum_extra=0
    if file.exists():
        with file.open(encoding='utf-8') as f:
            for line in f:
                check_work(cancel)
                try:r=json.loads(line)
                except ValueError:continue
                count+=1;ts=r['at_ns']/1e9;first=ts if first is None else min(first,ts);last=ts
                reasons.update(r.get('reasons',[]));eligible+=r.get('status')=='eligible_experimental'
                coverage=r.get('warning_coverage',{})
                coverage_frames[coverage.get('state','not_recorded')]+=1
                timing=r.get('processing_timing')
                if isinstance(timing,dict):
                    key=timing.get('capture_mode','unknown')+'/'+timing.get('decode_scope','not_measured')
                    bucket=timings.setdefault(key,{'frames':0,'engine_ms':[],'decode_ms':[],'deadline_overruns':0,'elapsed_over_150_ms':0})
                    bucket['frames']+=1
                    for name in ('engine_ms','decode_ms'):
                        value=timing.get(name)
                        if type(value) in (int,float) and math.isfinite(value) and value>=0:bucket[name].append(value)
                    bucket['deadline_overruns']+=timing.get('engine_deadline_exceeded') is True
                    bucket['elapsed_over_150_ms']+=timing.get('engine_elapsed_over_150_ms') is True
                else:timing_unknown+=1
                for alert in r.get('alerts',[]):alerts.append({'at_s':ts-first,'audio_delivery':r.get('audio_delivery','unknown'),
                                                              'warning_coverage':coverage or {'state':'not_recorded'},**alert})
                if r.get('ground'):
                    ground_sources[r['ground'].get('source','fit')]+=1
                    if r['ground'].get('source')=='held':
                        restoration['held_frames']+=1
                        provenance=r['ground'].get('held_restoration')
                        if not isinstance(provenance,dict):provenance={}
                        extra=provenance.get('extra_candidate_points')
                        if type(provenance.get('schema_version')) is int and provenance['schema_version']==1 and type(extra) is int and extra>=0:
                            restoration['frames_with_provenance']+=1;restoration['extra_candidate_points']+=extra
                            restoration['frames_with_additions']+=extra>0;maximum_extra=max(maximum_extra,extra)
                        else:restoration['frames_without_provenance']+=1
                    if r['ground'].get('rms_m') is not None and not r['ground'].get('degraded'):ground.append(r['ground']['rms_m'])
    duration=max(0,(last or 0)-(first or 0));labels=[]
    if (session/'labels.jsonl').exists():
        for line in (session/'labels.jsonl').read_text(encoding='utf-8').splitlines():
            try:labels.append(json.loads(line))
            except ValueError:pass
    coverage_events=[]
    coverage_path=session/'coverage-events.jsonl'
    if coverage_path.exists():
        with coverage_path.open(encoding='utf-8') as f:
            for line in f:
                check_work(cancel)
                try:coverage_events.append(json.loads(line))
                except ValueError:continue
    coverage_audio=[e for e in coverage_events if e.get('kind')=='coverage_audio']
    for bucket in timings.values():
        for name in ('engine_ms','decode_ms'):
            values=sorted(bucket[name]);n=len(values)
            bucket[name]={'samples':n,'mean':sum(values)/n if n else None,
                          'p95_nearest_rank':values[max(0,math.ceil(.95*n)-1)] if n else None,'max':values[-1] if n else None}
    processing={'groups':timings,'frames_without_timing':timing_unknown,
                'engine_deadline_overruns':sum(b['deadline_overruns'] for b in timings.values()),
                'scope':'perf_counter_ns durations. Engine.update only; decode is packet-to-points when measured. Neither measures end-to-end latency, queue delay, map work, recording, audio or UI. Synthetic/supplied-point input has unknown decode duration, not zero.'}
    restoration_report={'held_frames':restoration['held_frames'],'frames_with_provenance':restoration['frames_with_provenance'],
        'frames_without_provenance':restoration['frames_without_provenance'],
        'recorded_extra_candidate_points':restoration['extra_candidate_points'] if restoration['frames_with_provenance'] or not restoration['held_frames'] else None,
        'recorded_frames_with_additions':restoration['frames_with_additions'] if restoration['frames_with_provenance'] or not restoration['held_frames'] else None,
        'max_extra_candidate_points_per_recorded_frame':maximum_extra if restoration['frames_with_provenance'] or not restoration['held_frames'] else None,
        'scope':'Before final clustering: repeated per-frame candidate returns added beyond the strict point-local mask. Not accepted detection points, objects, unique points or verified hazards. Geometric groups may include road or nearby structures. Missing legacy provenance is unknown; totals cover only measured frames.'}
    report={'frames':count,'duration_s':duration,'eligible_frames':eligible,
            'processing_timing':processing,
            'held_restoration_provenance':restoration_report,
            'comma_ingestion':{'raw_recording_state':final.get('comma_recording','unknown'),
                **{k:final.get('stats',{}).get(k) for k in ('comma_records_unsaved','comma_non_json','comma_payloads_invalidated',
                     'comma_records_rejected','route_records_rejected','route_records_unsaved','comma_context_failures')}},
            'warning_coverage_frames':dict(coverage_frames),'coverage_events':coverage_events,
            'coverage_audio_submissions':sum(e.get('delivery')=='submitted_to_windows' for e in coverage_audio),
            'coverage_audio_interrupted':sum(e.get('kind')=='audio_interruption' and e.get('phrase') in ('WARNING RANGE LIMITED','WARNING SYSTEM UNAVAILABLE') for e in coverage_events),
            'capture_mode':metadata.get('capture_mode','unknown'),'telemetry_samples':final.get('stats',{}).get('comma_samples',0),
            'blocking_reasons':dict(reasons),'warning_events':alerts,'ground_sources':dict(ground_sources),
            'warnings_per_hour':len(alerts)*3600/duration if duration else None,
            'audio_submissions':sum(a['audio_delivery']=='submitted_to_windows' for a in alerts),
            'muted_decisions':sum(a['audio_delivery']=='muted' for a in alerts),
            'audio_failures_or_expired':sum(a['audio_delivery']=='failed_or_expired' for a in alerts),
            'human_labels':labels,'false_positive_rate':'UNKNOWN: independent labeled ground truth required',
            'miss_rate':'UNKNOWN: no labeled hazard inventory',
            'note':'warning_events counts decisions, including muted ones. Audio submission is not proof the driver heard it. Decision frequency is not false-positive rate. Synthetic tests do not verify reflectivity or road performance.'}
    try:
        cached=session/'hindsight-review.json';hindsight=None
        if not recompute_hindsight and cached.exists():
            old=json.loads(cached.read_text())
            if old.get('status')=='complete' and old.get('schema_version')==HINDSIGHT_VERSION and old.get('input_signature')==input_signature(session):hindsight=old
        if hindsight is None:hindsight=analyze_session(session,cancel)
    except WorkPaused:raise
    except Exception as e:
        hindsight={'status':'failed','error':str(e),'auto_label_counts':{},'labels':[],
                   'false_alarm_rate':'UNKNOWN','miss_rate':'UNKNOWN','settings_changed':False}
        (session/'hindsight-review.json').write_text(json.dumps(hindsight,indent=2),encoding='utf-8')
    report['hindsight']=hindsight
    map_path=session/'map-review.json'
    map_review=json.loads(map_path.read_text()) if map_path.exists() else {'status':'pending','reason':'Map review is pending or unavailable.'}
    report['route_map']=map_review
    route_path=session/'route-review.json'
    route_review=json.loads(route_path.read_text(encoding='utf-8')) if route_path.exists() else {'status':'pending','gps_state':'unavailable'}
    report['route_recording']=route_review
    proposal_path=session/'proposal-review.json'
    proposal=json.loads(proposal_path.read_text()) if proposal_path.exists() else {'status':'pending','paragraph':'Parameter proposal review is pending or unavailable.'}
    report['proposal']=proposal
    (session/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows=''.join('<tr><td>'+html.escape(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in reasons.most_common())
    page='<!doctype html><meta charset="utf-8"><title>Odyssey FCW session review</title><style>body{font:17px Segoe UI;max-width:900px;margin:40px auto;line-height:1.6}td{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    page+=f'<h1>Automatic session review</h1><p>{count} frames · {duration:.1f} seconds · {len(alerts)} warning decisions · {report["audio_submissions"]} audio requests submitted · {report["muted_decisions"]} muted decisions · {eligible} frames eligible for experimental warning logic.</p>'
    page+=f'<p>Ground modes: {html.escape(str(dict(ground_sources)))}. Held or mount-prior ground is degraded evidence with upright-return checks; it is not a fresh road measurement or calibration evidence.</p>'
    page+='<h2>Held-road grouping provenance</h2><p>'+html.escape(restoration_report['scope'])+'</p><pre>'+html.escape(json.dumps({k:v for k,v in restoration_report.items() if k!='scope'},indent=2))+'</pre>'
    page+='<h2>Processing cost and availability</h2><p>'+html.escape(processing['scope'])+'</p>'
    page+=f'<p>{processing["engine_deadline_overruns"]} frames hit the engine’s 150 ms deadline and withheld warnings. {timing_unknown} frames have no recorded timing. Missing measurements are unknown, not zero. The elapsed-over-150 count is reported separately because the engine’s deadline check occurs inside Engine.update.</p>'
    page+='<pre>'+html.escape(json.dumps(processing['groups'],indent=2))+'</pre>'
    page+='<h2>Comma recording quality</h2><p>Malformed records are preserved before normalization. Invalid stream lines are saved byte-for-byte in the rejection journal. Null counters mean this older or incomplete recording did not provide that measurement.</p><pre>'+html.escape(json.dumps(report['comma_ingestion'],indent=2))+'</pre>'
    if (session/'comma-rejected.jsonl').exists():page+='<p><a href="comma-rejected.jsonl">Rejected raw stream lines</a> (original bytes stored as Base64).</p>'
    page+='<h2>Warning coverage and notices</h2>'
    page+=f'<p>{coverage_frames["experimental"]} frames passed nominal experimental checks; {coverage_frames["limited"]} frames had limited coverage; {coverage_frames["unavailable"]} were unavailable; {coverage_frames["not_recorded"]} did not record this field. Warning-eligible frame totals include limited near-field operation.</p>'
    page+='<p>Mount-prior operation processes only upright candidates within 25 m. Held-road operation also has reduced sparse-target coverage. Neither mode establishes a detection range. Limited and unavailable headlines appear immediately; each spoken notice requires 0.5 s of sustained coverage loss. Status speech waits for obstacle speech to finish and does not announce recovery as proof of clear road. Raw coverage transitions remain recorded, including brief failures that do not speak.</p>'
    page+=f'<p>{report["coverage_audio_submissions"]} coverage notices submitted to Windows; {report["coverage_audio_interrupted"]} interrupted by higher-priority audio. A still-current interrupted notice retries after obstacle speech. Submission does not prove playback completed or that anyone heard it. Muted sessions still record coverage transitions, and missing frames generate unavailable transitions through the watchdog.</p>'
    page+='<details><summary>Coverage transitions and notice delivery</summary><pre>'+html.escape(json.dumps(coverage_events[:200],indent=2))+'</pre></details>'
    if coverage_path.exists():page+='<p><a href="coverage-events.jsonl">All coverage transitions and notice delivery attempts</a> (up to 200 shown above).</p>'
    if report['capture_mode'] in ('demo','test') or metadata.get('synthetic_fixture'):page+='<p><strong>Synthetic demonstration: these are not driving observations or sensor-performance results.</strong></p>'
    if report['capture_mode']=='comma':page+=f'<p>Comma-only check: {report["telemetry_samples"]} telemetry messages recorded. Zero lidar frames is expected in this mode.</p>'
    page+='<h2>Automatic hindsight — evidence, not ground truth</h2>'
    if hindsight['status']=='failed':page+='<p>Hindsight failed: '+html.escape(hindsight['error'])+'. No improvement or accuracy conclusion is available.</p>'
    else:
        counts=hindsight['auto_label_counts']
        page+='<table><tr><th>False-alarm evidence</th><th>Miss / possible-hazard evidence</th></tr><tr><td>'+str(counts.get('probable_false_positive',0))+' probable false candidates ('+str(hindsight['probable_false_positive_warnings'])+' warning decisions)</td><td>'+str(hindsight['candidate_misses'])+' candidate misses; '+str(hindsight['possible_hazards'])+' possible hazards</td></tr><tr><td>Confirmed false-alarm rate: UNKNOWN</td><td>Confirmed miss rate: UNKNOWN</td></tr></table>'
        page+=f'<p>{hindsight["reaction_events"]} vehicle-response episodes; {hindsight["delivery_gaps"]} possible audio delivery gaps. A submitted audio request does not prove it was heard.</p>'
        page+=f'<p>{hindsight["candidate_misses_with_no_eligible_frames"]} candidate misses occurred with no warning-eligible frames in the lookback window. Availability failures are recorded in each encounter; they do not establish a detection-threshold problem.</p>'
        page+=f'<p>Odometry coverage: {hindsight["odometry_samples"]} of {hindsight["car_samples"]} car samples in {hindsight["odometry_segments"]} separate local segments. Verified footprint: {hindsight["footprint"]["confirmed"]}. Gaps are not bridged.</p>'
        page+='<p>Clock changes and evidence gaps separate trajectories. Overlapping timestamps are excluded on both sides; later trustworthy intervals can recover without clearing calibration requirements.</p>'
        page+='<p>Driver responses may have unrelated causes; objects may move away. Confidence is qualitative. Fewer warnings alone is not a success measure. No thresholds or settings were changed.</p>'
        page+='<p><a href="auto-labels.jsonl">All automatic labels</a> · <a href="hindsight-review.json">Hindsight diagnostics</a></p><table><tr><th>Evidence</th><th>Confidence</th><th>Reason / encounter</th></tr>'
        for event in hindsight['labels'][:100]:
            page+='<tr><td>'+html.escape(event['label'])+'</td><td>'+html.escape(event['confidence'])+'</td><td>'+html.escape(event['reason'])+' <a href="'+html.escape(event['event_file'],quote=True)+'">Five-second evidence window</a></td></tr>'
        page+='</table><p>Up to 100 events are shown here; all events are retained in the automatic-label file.</p>'
        page+='<details><summary>Why evidence was withheld</summary><pre>'+html.escape(json.dumps(hindsight['diagnostics'],indent=2))+'</pre></details>'
    page+='<h2>Your manual labels — kept separate</h2><pre>'+html.escape(json.dumps(labels,indent=2))+'</pre>'
    from mapping.report import route_section
    page+=route_section(session,route_review,map_review)
    page+='<h2>Learned map coverage</h2><p>'+html.escape(map_review.get('reason',''))+'</p>'
    if map_review.get('status')=='complete':
        areas=(str(len(map_review['graph']['components']))+' local map areas in this revision' if 'components' in map_review.get('graph',{}) else str(map_review.get('new_submaps',0))+' new local submaps')
        page+=f'<p>{map_review["localized_keyframes"]} of {map_review["eligible_keyframes"]} eligible keyframes localized against existing submaps ({map_review["coverage_percent"]:.1f}%). {areas}; {map_review["promoted_static_voxels"]} persistent static voxels; {map_review["forgotten_voxels"]} inactive/forgotten voxels.</p>'
        raw_coverage=map_review.get('extraction',{}).get('raw_frame_coverage_fraction')
        if raw_coverage is not None and raw_coverage<.99:
            page+=f'<p><strong>Partial lidar archive:</strong> {raw_coverage*100:.1f}% of processed frames were recovered from the raw recording. GPS/motion traces may extend beyond the available 3D map evidence.</p>'
    page+='<p>Coverage measures checked recorded views, not the percentage of roads surveyed. Map evidence cannot suppress a corridor warning. Semantic labels such as deer, mailbox or sign are not inferred from persistence alone.</p>'
    if map_review.get('revision'):page+='<p>Map revision: '+html.escape(map_review['revision'][:12])+'. Original observations and previous revisions are retained.</p>'
    page+='<h2>Gated parameter proposal</h2><p>'+html.escape(proposal.get('paragraph',''))+'</p>'
    if (session/'proposal-exam.json').exists():page+='<p><a href="proposal-exam.json">Full fixed-exam results, including baseline misses</a></p>'
    page+='<p>Only the app’s explicit Apply button can adopt an eligible proposal. No automatic adoption or neural-network training occurs.</p>'
    page+='<h2>What prevented warnings</h2><table>'+rows+'</table><h2>Warning events</h2><pre>'+html.escape(json.dumps(alerts,indent=2))+'</pre>'
    page+='<h2>What this does not establish</h2><p>False-positive and missed-hazard rates remain unknown without independently labeled encounters. This report does not certify the system for driving.</p>'
    (session/'review.html').write_text(page,encoding='utf-8')
    return report

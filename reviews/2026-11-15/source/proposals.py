"""Phase 3: propose, examine, require an explicit Apply action, retain rollback."""
import json, uuid
import runtime_clock as time
from pathlib import Path
from policy import canonical_policy,policy_digest
from hindsight import read_jsonl,write_json,input_signature
from exam import run_exam,validate_assets,session_inventory,signature_digest,SUITE_ID

def eligibility(result):
    required={'bikescenes','own_sessions','hk_insertions','own_insertions','bus_occlusion','moving_geometry_and_sparse_targets'}
    categories=result.get('categories',{})
    basic=bool(result.get('status')=='passed' and result.get('suite_id')==SUITE_ID and
                required<=set(categories) and all(categories[k].get('passed') is True for k in required) and
                categories['own_sessions'].get('probable_false_evidence_reduced',0)>0 and
                result.get('baseline_digest')!=result.get('candidate_digest') and result.get('eligible') is True)
    if not basic:return False
    try:
        from adversarial_exam import result_is_eligible
        if not result_is_eligible(categories['moving_geometry_and_sparse_targets']):return False
        bike=categories['bikescenes']
        if bike['frames']!=3021 or bike['baseline_positive_points_caught']<=0 or bike['lost_baseline_positive_points']!=0 or bike['candidate_ground_points_retained']>bike['baseline_ground_points_retained']:return False
        for name,min_backgrounds in (('hk_insertions',4),('own_insertions',1)):
            cases=categories[name]['cases'];backgrounds={c['background'] for c in cases}
            if len(backgrounds)<min_backgrounds:return False
            if not any(c['baseline']['caught'] for c in cases):return False
            for background in backgrounds:
                covered={(c['kind'],c.get('range_m')) for c in cases if c['background']==background}
                if not {(k,r) for k in ('pedestrian','deer') for r in (40,70,100)}<=covered:return False
            for c in cases:
                a,b=c['baseline'],c['candidate']
                if c.get('regression') or (a['caught'] and (not b['caught'] or b['first_warning_distance_m']+.01<a['first_warning_distance_m'])) or b['other_warnings']>a['other_warnings']:return False
                if a.get('track_observed') and (not b.get('track_observed') or b['first_detection_distance_m']+.01<a['first_detection_distance_m']):return False
        own=categories['own_sessions']['sessions']
        if not own or any(r['lost_baseline_catches'] or r['shortened_warnings'] or r['unresolved_useful_labels'] for r in own):return False
        inventory=result['session_inventory']
        if not inventory or not {r['session'] for r in own}<={Path(i['folder']).name for i in inventory}:return False
        required_manual={Path(i['folder']).name for i in result['session_inventory'] if i['has_manual_labels']}
        if not required_manual<={r['session'] for r in own}:return False
        bus=categories['bus_occlusion']['cases']
        if len(bus)<2 or any(r['lost_baseline_bus_points'] or r['candidate_bus_points']<=0 for r in bus):return False
    except (KeyError,TypeError,ValueError):return False
    return True

def propose_after_session(folder,data_root,asset_root,engine_version,progress=None,cancel=None):
    folder=Path(folder);data_root=Path(data_root);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    root=data_root/'proposals';root.mkdir(parents=True,exist_ok=True)
    manual=[r for _,r in read_jsonl(folder/'labels.jsonl')];auto=[r for _,r in read_jsonl(folder/'auto-labels.jsonl')]
    false_count=sum(r.get('label')=='probable_false_positive' for r in auto)+sum(r.get('label')=='false_warning' for r in manual)
    miss_count=sum(r.get('label') in ('candidate_miss','possible_hazard') for r in auto)+sum(r.get('label')=='missed_obstacle' for r in manual)
    summary={'schema_version':1,'status':'no_proposal','false_alarm_evidence':false_count,'miss_or_hazard_evidence':miss_count,
             'eligible':False,'settings_changed':False,'source_session':folder.name}
    def finish():write_json(folder/'proposal-review.json',summary);write_json(root/'latest.json',summary);return summary
    if cfg.get('capture_mode')!='live' or cfg.get('synthetic_fixture') or not cfg.get('automatic_proposals',True):
        summary['paragraph']='No tuning proposal: this is not an enabled real-drive evaluation. False-alarm and miss evidence remain available separately.';return finish()
    hindsight_path=folder/'hindsight-review.json'
    previous=json.loads(hindsight_path.read_text()) if hindsight_path.exists() else {}
    if previous.get('status')!='complete' or previous.get('input_signature')!=input_signature(folder):
        summary['paragraph']='No tuning proposal: hindsight evidence is missing, failed or stale. Settings remain unchanged.';return finish()
    if false_count<3:
        summary['paragraph']=f'No tuning proposal: {false_count} false-alarm evidence events and {miss_count} miss/possible-hazard events were saved. At least three false-alarm evidence events are required before trying a change; fewer warnings alone is not an improvement.';return finish()
    baseline=canonical_policy(cfg.get('perception_policy'));candidate=dict(baseline)
    if baseline['cluster_cell_m']>.35:candidate['cluster_cell_m']=round(max(.3,baseline['cluster_cell_m']-.05),2)
    elif baseline['min_cluster_points']<8:candidate['min_cluster_points']+=1
    else:
        summary['paragraph']='No permitted tuning step remains in the narrow parameter range. Evidence was retained; settings are unchanged.';return finish()
    proposal_id=uuid.uuid4().hex
    result=run_exam(asset_root,data_root/'recordings',baseline,candidate,progress,cancel)
    exam_path=root/(proposal_id+'-exam.json');write_json(exam_path,result)
    write_json(folder/'proposal-exam.json',result)
    allowed=eligibility(result)
    bike=result.get('categories',{}).get('bikescenes',{})
    own=result.get('categories',{}).get('own_sessions',{})
    short=sum(v.get('shortened_warnings',0) for v in own['sessions']) if 'sessions' in own else 'unknown'
    lost=sum(v.get('lost_baseline_catches',0) for v in own['sessions']) if 'sessions' in own else 'unknown'
    change=', '.join(f'{k}: {baseline[k]} → {candidate[k]}' for k in baseline if baseline[k]!=candidate[k])
    target_categories=[result.get('categories',{}).get(k,{}) for k in ('hk_insertions','own_insertions')]
    old_misses=sum(c.get('baseline_target_misses',0) for c in target_categories) if all('baseline_target_misses' in c for c in target_categories) else 'unknown'
    new_misses=sum(c.get('candidate_target_misses',0) for c in target_categories) if all('candidate_target_misses' in c for c in target_categories) else 'unknown'
    old_detection=sum(c.get('baseline_detection_misses',0) for c in target_categories) if all('baseline_detection_misses' in c for c in target_categories) else 'unknown'
    new_detection=sum(c.get('candidate_detection_misses',0) for c in target_categories) if all('candidate_detection_misses' in c for c in target_categories) else 'unknown'
    bus=result.get('categories',{}).get('bus_occlusion',{}).get('cases')
    bus_misses=sum(r['candidate_bus_points']==0 for r in bus) if bus is not None else 'unknown'
    adversarial=result.get('categories',{}).get('moving_geometry_and_sparse_targets',{})
    adverse_summary=(f'Moving-scene off-target warnings: {adversarial.get("baseline_off_target_warnings","unknown")} → {adversarial.get("candidate_off_target_warnings","unknown")}; '
                     f'absent target warnings: {adversarial.get("baseline_sequence_warning_misses","unknown")} → {adversarial.get("candidate_sequence_warning_misses","unknown")}; '
                     f'ray-cast sparse detection misses: {adversarial.get("baseline_sparse_detection_misses","unknown")} → {adversarial.get("candidate_sparse_detection_misses","unknown")}. ')
    paragraph=(f'Proposed {change}. This drive supplied {false_count} false-alarm evidence events and {miss_count} miss/possible-hazard events. '
               f'Replay reduced {own.get("probable_false_evidence_reduced",0)} linked probable-false candidates; BikeScenes lost {bike.get("lost_baseline_positive_points","unknown")} baseline obstacle-labeled points. '
               f'Manual-session replay lost {lost} baseline catches and shortened {short} warnings. Sparse-target detection misses: {old_detection} → {new_detection}; absent target-specific warnings: {old_misses} → {new_misses}; bus cases without detection: {bus_misses}. Exam: {result["status"]}. '
               +adverse_summary+('Eligible within this fixed exam; Apply is required and previous settings will be kept for rollback.' if allowed else 'Not eligible; settings remain unchanged. '+result.get('reason','A regression or missing exam coverage prevented approval.')))
    proposal={'schema_version':1,'id':proposal_id,'engine_version':engine_version,'created_at':time.time(),'state':'eligible' if allowed else 'rejected',
              'baseline_policy':baseline,'candidate_policy':candidate,'baseline_digest':policy_digest(baseline),'candidate_digest':policy_digest(candidate),
              'exam_file':exam_path.name,'exam_digest':result['exam_digest'],'paragraph':paragraph,
              'false_alarm_evidence':false_count,'miss_or_hazard_evidence':miss_count,'source_session':folder.name}
    write_json(root/(proposal_id+'.json'),proposal)
    summary.update(status=proposal['state'],eligible=allowed,id=proposal_id,paragraph=paragraph,exam_file='proposal-exam.json',baseline_policy=baseline,candidate_policy=candidate,
                   engine_version=engine_version,baseline_digest=policy_digest(baseline))
    return finish()

def load_proposal(root,identifier):
    if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid proposal identifier')
    return json.loads((Path(root)/(identifier+'.json')).read_text())

def apply_proposal(config_path,data_root,asset_root,identifier,engine_version):
    """Called only by the explicit authenticated UI action; never by analysis."""
    config_path=Path(config_path);data_root=Path(data_root);root=data_root/'proposals'
    proposal=load_proposal(root,identifier)
    if proposal['state']!='eligible' or proposal['engine_version']!=engine_version:raise ValueError('Proposal is not eligible for this version')
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'));current=canonical_policy(cfg.get('perception_policy'))
    if policy_digest(current)!=proposal['baseline_digest']:raise ValueError('Active settings changed; rerun the exam')
    result=json.loads((root/proposal['exam_file']).read_text())
    recorded=result.get('exam_digest');actual=signature_digest({k:v for k,v in result.items() if k!='exam_digest'})
    if recorded!=actual or recorded!=proposal['exam_digest'] or not eligibility(result):raise ValueError('Exam result is missing, changed or failed')
    candidate=canonical_policy(proposal['candidate_policy'])
    if policy_digest(candidate)!=result['candidate_digest'] or policy_digest(candidate)!=proposal['candidate_digest']:raise ValueError('Proposed parameters changed after the exam')
    _,signature=validate_assets(asset_root,full=False)
    if signature!=result['asset_signature']:raise ValueError('Fixed exam assets changed; rerun the exam')
    if session_inventory(data_root/'recordings')!=result['session_inventory']:raise ValueError('Session or manual-label evidence changed; rerun the exam')
    rollback={'schema_version':1,'before':current,'after_digest':policy_digest(candidate),'proposal_id':identifier,'used':False}
    write_json(root/'rollback.json',rollback)
    cfg['perception_policy']=candidate;write_json(config_path,cfg)
    proposal.update(state='applied',applied_at=time.time());write_json(root/(identifier+'.json'),proposal)
    latest={'schema_version':1,'status':'applied','id':identifier,'eligible':False,'paragraph':proposal['paragraph']+' Applied by explicit user action.','settings_changed':True}
    write_json(root/'latest.json',latest)
    with (root/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'action':'apply','id':identifier,'before':current,'after':candidate,'at':time.time()})+'\n')
    return cfg

def rollback_policy(config_path,data_root):
    config_path=Path(config_path);root=Path(data_root)/'proposals';saved=json.loads((root/'rollback.json').read_text())
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    if saved.get('used') or policy_digest(cfg.get('perception_policy'))!=saved['after_digest']:raise ValueError('Rollback is unavailable for the active settings')
    before=canonical_policy(saved['before']);cfg['perception_policy']=before;write_json(config_path,cfg)
    saved['used']=True;write_json(root/'rollback.json',saved)
    write_json(root/'latest.json',{'schema_version':1,'status':'rolled_back','eligible':False,'paragraph':'Previous perception settings restored by explicit user action.','settings_changed':True})
    with (root/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'action':'rollback','id':saved['proposal_id'],'restored':before,'at':time.time()})+'\n')
    return cfg

def review_state(data_root,engine_version=None,active_policy=None):
    root=Path(data_root)/'proposals';result={'status':'none','eligible':False,'paragraph':'No eligible parameter proposal. Every change requires an exam and your Apply click.'}
    try:
        if (root/'latest.json').exists():result=json.loads((root/'latest.json').read_text())
        result['rollback_available']=(root/'rollback.json').exists() and not json.loads((root/'rollback.json').read_text()).get('used')
        if result.get('eligible') and ((engine_version is not None and result.get('engine_version')!=engine_version) or
                                      (active_policy is not None and result.get('baseline_digest')!=policy_digest(active_policy))):
            result['eligible']=False;result['status']='stale';result['paragraph']+=' The app or active settings changed; a new exam is required.'
    except (ValueError,OSError):result={'status':'error','eligible':False,'rollback_available':False,'paragraph':'Proposal metadata unavailable; settings were not changed.'}
    return result

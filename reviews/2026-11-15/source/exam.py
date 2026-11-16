"""Fixed, append-only regression exam. A missing category never counts as pass.

No training occurs. Synthetic inserted returns test software geometry, not the
physical detectability of deer, clothing, weather or a particular lidar unit.
"""
import hashlib,json,math,zipfile
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import numpy as np
from fcw_engine import Engine,Ground,Mount,clusters,transform
from policy import canonical_policy,policy_digest
from recording import frames
from telemetry import Telemetry
from telemetry import rotation
from hindsight import read_jsonl
from offline_jobs import check_work,WorkPaused

SUITE_ID='odyssey-fcw-fixed-exam-2026-10-v3'
POSITIVE_CLASSES={10,11,13,15,16,18,20,30,31,32,50,51,52,70,71,80,81,99,252,253,254,255,256,257,258,259}
GROUND_CLASSES={40,41,44,48,49,60,72}

def digest_file(path,cancel=None):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(4*1024*1024):check_work(cancel);digest.update(data)
    return digest.hexdigest()

def validate_assets(root,full=True,cancel=None):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('suite_id')!=SUITE_ID or manifest.get('schema_version')!=1:raise ValueError('Unrecognized fixed exam')
    if manifest.get('bikescenes',{}).get('frames')!=3021:raise ValueError('Full labeled BikeScenes set required')
    if manifest.get('insertion_recipe',{}).get('ranges_m')!=[40,70,100]:raise ValueError('All three required ranges must be tested')
    if manifest['insertion_recipe'].get('background_mode')!='road_in_view':raise ValueError('The v2 exam requires the road-in-view recipe')
    if any(c.get('vehicle_height_m')!=3.6 for c in manifest.get('hk_mems',{}).get('cases',[]) if c.get('role')=='bus_occlusion'):
        raise ValueError('The v2 bus cases require their dataset-specific height envelope')
    roles=[c.get('role') for c in manifest.get('hk_mems',{}).get('cases',[])]
    if roles.count('insertion_background')<4 or roles.count('bus_occlusion')<2:raise ValueError('Fixed real-background and bus cases are missing')
    signature={'manifest_sha256':digest_file(root/'manifest.json'),'files':{}}
    for item in manifest['files']:
        p=root/item['path']
        if p.resolve().parent!=root.resolve():raise ValueError('Exam paths must stay within the asset folder')
        st=p.stat()
        if st.st_size!=item['bytes'] or (full and digest_file(p,cancel)!=item['sha256']):raise ValueError('Exam asset integrity failed: '+p.name)
        signature['files'][p.name]=[st.st_size,st.st_mtime_ns]
    if manifest.get('adversarial_corpus')!='regression-corpus.json':raise ValueError('The v3 moving-geometry/sparse-target corpus is required')
    from adversarial_exam import validate_corpus
    corpus=validate_corpus(json.loads((root/'regression-corpus.json').read_text(encoding='utf-8')))
    required={'bikescenes.zip','regression-corpus.json'}|{v['file'] for v in manifest['hk_mems']['cases']}|{v['file'] for v in corpus['sequences']+corpus['sparse']}
    if not required<=set(signature['files']):raise ValueError('Exam assets are not all hashed')
    return manifest,signature

def session_inventory(record_root):
    root=Path(record_root);items=[]
    if not root.exists():return items
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.resolve().parent!=root.resolve() or not (folder/'final-state.json').exists():continue
        try:cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
        except (ValueError,OSError):continue
        if cfg.get('capture_mode')!='live' or cfg.get('synthetic_fixture'):continue
        files={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in folder.iterdir() if p.is_file() and
               (p.name in ('session.json','final-state.json','comma.jsonl','decisions.jsonl','frame-contexts.jsonl','labels.jsonl','auto-labels.jsonl') or p.name.startswith('lidar-'))}
        items.append({'folder':str(folder.resolve()),'files':files,'has_manual_labels':bool((folder/'labels.jsonl').exists() and (folder/'labels.jsonl').stat().st_size)})
    return items

def signature_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def point_mask(points,ground,mount,policy):
    result=np.zeros(len(points),dtype=bool)
    if ground:
        for d in clusters(points,ground,mount,policy,return_membership=True):result[d['indices']]=True
    return result

def bike_exam(root,manifest,baseline,candidate,progress=None,cancel=None):
    m=Mount(xyz=[0,0,float(manifest['bikescenes']['height_m'])]);g=Ground()
    totals={'frames':0,'baseline_positive_points_caught':0,'candidate_positive_points_caught':0,
            'baseline_positive_points_not_caught':0,'candidate_positive_points_not_caught':0,
            'baseline_ground_points_retained':0,'candidate_ground_points_retained':0,'lost_baseline_positive_points':0,
            'frames_with_lost_positive_points':0,'ground_fit_rejected_frames':0}
    with zipfile.ZipFile(Path(root)/'bikescenes.zip') as z:
        for i in range(3021):
            check_work(cancel)
            p=np.frombuffer(z.read(f'robosense_m1p/{i:06d}.bin'),dtype='<f4').reshape(-1,4)
            labels=np.frombuffer(z.read(f'labels/{i:06d}.label'),dtype='<u4')&65535
            if len(p)!=len(labels):raise ValueError('BikeScenes label/point count mismatch')
            body=transform(p,m);ground=g.fit(body)
            roi=np.isfinite(body).all(1)&(body[:,0]>m.bumper_x_m+.3)&(body[:,0]<120)&(abs(body[:,1])<35)
            positive=np.isin(labels,list(POSITIVE_CLASSES))&roi;negative=np.isin(labels,list(GROUND_CLASSES))&roi
            a=point_mask(body,ground,m,baseline)
            # Identical policies use identical deterministic point selection.
            # Real proposals still evaluate each policy separately.
            b=a if baseline==candidate else point_mask(body,ground,m,candidate)
            lost=int((positive&a&~b).sum());totals['lost_baseline_positive_points']+=lost
            totals['frames_with_lost_positive_points']+=bool(lost);totals['frames']+=1
            totals['ground_fit_rejected_frames']+=ground is None
            for name,mask in (('baseline',a),('candidate',b)):
                totals[name+'_positive_points_caught']+=int((positive&mask).sum())
                totals[name+'_positive_points_not_caught']+=int((positive&~mask).sum())
                totals[name+'_ground_points_retained']+=int((negative&mask).sum())
            if progress and i%100==0:progress(f'BikeScenes labeled exam: {i+1}/3021')
    totals['passed']=totals['lost_baseline_positive_points']==0 and totals['candidate_ground_points_retained']<=totals['baseline_ground_points_retained']
    totals['metric_note']='Point-level semantic retention, not object-instance recall or a collision false-alarm rate. The dataset has no instance IDs; overhead/static classes can be intentionally excluded by FCW.'
    return totals

def inserted_scene(background,height,kind,range_m,point_count,policy,recipe,target=True,cancel=None):
    m=Mount(**height) if isinstance(height,dict) else Mount(xyz=[0,0,height],measured=True,alignment_verified=True)
    m.measured=True;m.alignment_verified=True
    body=transform(background,m);ground=Ground().fit(body)
    if ground is None:raise ValueError('Insertion background has no supported ground fit')
    a,b,c=ground['abc'];h=body[:,2]-(a*body[:,0]+b*body[:,1]+c)
    # Explicitly isolate the inserted hazard. Preserve the original background
    # elsewhere, and preserve road returns inside the corridor.
    keep_background=~((abs(body[:,1])<2.)&(body[:,0]>3)&(h>.2))
    intensities=background[:,3] if background.shape[1]>3 else np.zeros(len(body))
    body=body[keep_background];intensities=intensities[keep_background]
    spec=recipe[kind];columns=max(2,point_count//2);ys=np.repeat(np.linspace(-spec['width_m']/2,spec['width_m']/2,columns),2)
    zs=np.tile([.25,spec['height_m']],columns);xs=np.full(len(ys),range_m)
    target_world=np.column_stack([xs,ys,a*xs+b*ys+c+zs])
    engine=Engine(m,policy);speed=recipe['speed_mps'];step=recipe['step_s'];first_warning=None;false_warnings=0;detected=False;first_detection=None;blocked=Counter()
    for i in range(int((range_m-4)/speed/step)+1):
        check_work(cancel)
        travel=speed*i*step;world=body;intensity=intensities
        if target:
            # Refresh the same ego-relative road/background each step. Only
            # the inserted target approaches; translating a finite scan made
            # its road disappear. This remains a controlled logic test, not
            # realistic motion/occlusion for surrounding scenery.
            approaching=target_world-np.array([travel,0,a*travel])
            world=np.concatenate([world,approaching]);intensity=np.r_[intensity,np.full(len(target_world),recipe['intensity'])]
        observed=(world-m.xyz)@rotation(np.radians(m.rpy_deg))
        angles=np.arctan2(observed[:,1],observed[:,0]);pitch=np.arctan2(observed[:,2],np.hypot(observed[:,0],observed[:,1]))
        keep=(observed[:,0]>.2)&(abs(angles)<math.radians(60))&(abs(pitch)<math.radians(12.5))
        p=np.column_stack([observed[keep],intensity[keep]])
        ctx={'at_ns':100_000_000_000+int(i*step*1e9),'speed_mps':speed,'yaw_rate_rps':0.,'ready':True,'motion_ready':True,
             'path':np.column_stack([np.linspace(0,120,61),np.zeros(61)]),'reasons':[]}
        q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':.001}
        result=engine.update(p,ctx,q)
        blocked.update(result['reasons'])
        current=range_m-travel
        for tr in result['tracks']:
            if tr['misses']==0 and abs(tr['position'][0]-current)<1 and abs(tr['position'][1])<1:
                detected=True
                if first_detection is None:first_detection=float(current)
        for warning in result['alerts']:
            if target and abs(warning['position_m'][0]-current)<1 and abs(warning['position_m'][1])<1:
                if first_warning is None:first_warning=float(current)
            else:false_warnings+=1
    return {'caught':first_warning is not None,'first_warning_distance_m':first_warning,'track_observed':detected,
            'first_detection_distance_m':first_detection,'other_warnings':false_warnings,'blocking_reasons':dict(blocked)}

def injection_exam(backgrounds,baseline,candidate,recipe,progress=None,cancel=None):
    cases=[]
    for name,points,height in backgrounds:
        for kind in ('pedestrian','deer'):
            for distance in (40,70,100):
                for count in recipe[kind]['points']:
                    if progress:progress(f'Sparse-target exam: {name}, {kind}, {distance} m, {count} points')
                    a=inserted_scene(points,height,kind,distance,count,baseline,recipe,cancel=cancel)
                    b=inserted_scene(points,height,kind,distance,count,candidate,recipe,cancel=cancel)
                    loss=(a['caught'] and not b['caught']) or (a['track_observed'] and not b['track_observed'])
                    shorter=bool((a['caught'] and b['caught'] and b['first_warning_distance_m']+.01<a['first_warning_distance_m']) or
                                 (a['track_observed'] and b['track_observed'] and b['first_detection_distance_m']+.01<a['first_detection_distance_m']))
                    cases.append({'background':name,'kind':kind,'range_m':distance,'points':count,'baseline':a,'candidate':b,
                                  'regression':bool(loss or shorter or b['other_warnings']>a['other_warnings'])})
        a=inserted_scene(points,height,'pedestrian',70,4,baseline,recipe,target=False,cancel=cancel)
        b=inserted_scene(points,height,'pedestrian',70,4,candidate,recipe,target=False,cancel=cancel)
        cases.append({'background':name,'kind':'clear_corridor_control','baseline':a,'candidate':b,'regression':b['other_warnings']>a['other_warnings']})
    return {'passed':not any(c['regression'] for c in cases),'cases':cases,
            'baseline_target_misses':sum(not c['baseline']['caught'] for c in cases if c['kind']!='clear_corridor_control'),
            'candidate_target_misses':sum(not c['candidate']['caught'] for c in cases if c['kind']!='clear_corridor_control'),
            'baseline_detection_misses':sum(not c['baseline']['track_observed'] for c in cases if c['kind']!='clear_corridor_control'),
            'candidate_detection_misses':sum(not c['candidate']['track_observed'] for c in cases if c['kind']!='clear_corridor_control'),
            'metric_note':'Target warnings and detections are separate. Road/background is fixed in the ego frame while inserted targets approach. This isolates warning logic; background motion, return physics and real-road accuracy are not modeled.'}

def bus_exam(root,manifest,baseline,candidate):
    rows=[]
    for case in manifest['hk_mems']['cases']:
        if case['role']!='bus_occlusion':continue
        p=np.load(Path(root)/case['file'],allow_pickle=False)['points'];m=Mount(xyz=[0,0,case['height_m']],vehicle_height_m=case['vehicle_height_m'],measured=True,alignment_verified=True)
        body=transform(p,m);g=Ground().fit(body)
        # Exercise the engine's actual ground selection, including cold-start
        # occlusion. This static frame tests detection, not warning lead time.
        ctx={'at_ns':100_000_000_000,'speed_mps':0.,'yaw_rate_rps':0.,'ready':True,'motion_ready':True,
             'path':np.column_stack([np.linspace(0,120,61),np.zeros(61)]),'reasons':[]}
        q={'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':.001}
        ar=Engine(m,baseline).update(p,ctx,q);br=Engine(m,candidate).update(p,ctx,q)
        a=point_mask(body,ar['ground'],m,baseline);b=point_mask(body,br['ground'],m,candidate)
        # Coarse, independently inspected near-bus region; retain the known
        # baseline failure rather than presenting zero detections as a success.
        bus=(body[:,0]>2.3)&(body[:,0]<4.)&(abs(body[:,1])<1.6)&(body[:,2]>.3)&(body[:,2]<3.5)
        rows.append({'case':case['file'],'baseline_bus_points':int((a&bus).sum()),'candidate_bus_points':int((b&bus).sum()),
                     'baseline_ground_source':(ar['ground'] or {}).get('source'),'candidate_ground_source':(br['ground'] or {}).get('source'),
                     'vehicle_height_m':m.vehicle_height_m,
                     'ground_available':g is not None,'lost_baseline_bus_points':int((a&bus&~b).sum())})
    return {'passed':bool(rows) and all(r['lost_baseline_bus_points']==0 and r['candidate_bus_points']>0 for r in rows),'cases':rows,
            'known_limitation':'Static bus-frame detection with a dataset-specific height envelope; this does not measure lead time or establish Odyssey field performance.'}

def same_object(position,track):
    if track.get('misses',0):return False
    return np.linalg.norm(np.asarray(position)-np.asarray(track.get('position',[1e9,1e9])))<=1.

def protect_alerts(required,proposed,new_states):
    missing=[];shorter=[]
    for warning in required:
        t=warning['at_ns'];a=warning['alert'];tracks={v['id']:v for v in new_states.get(t,{}).get('tracks',[])}
        matching=[v for v in proposed if t-5_000_000_000<=v['at_ns']<=t and v['alert'].get('track_id') in tracks and same_object(a['position_m'],tracks[v['alert']['track_id']])]
        if not matching:missing.append(warning);continue
        if max(v['alert'].get('range_m',-1) for v in matching)+.01<a.get('range_m',0):shorter.append(warning)
    return missing,shorter

def causal_frames(folder,cancel=None):
    tele=Telemetry();diagnostics=Counter();raw=iter(r for _,r in read_jsonl(Path(folder)/'comma.jsonl',diagnostics,cancel));next_raw=next(raw,None)
    for points,stamps,original,q in frames(folder,cancel,strict=True):
        cutoff=q.get('fusion_cutoff_ns',q.get('last_receipt_ns'))
        if cutoff is None:raise ValueError('Missing causal frame cutoff')
        while next_raw is not None and next_raw.get('laptop_receipt_ns',0)<=cutoff:
            check_work(cancel)
            r=next_raw;receipt=r.get('laptop_receipt_ns',0)
            if r.get('kind')=='pong' and 'laptop_send_ns' in r:tele.clock.add(r['laptop_send_ns'],r['remote_receive_ns'],r['remote_send_ns'],receipt)
            else:tele.accept(r,receipt)
            next_raw=next(raw,None)
        if diagnostics:raise ValueError('Corrupt comma log; exam coverage cannot be established')
        yield points,stamps,tele.context(original['at_ns']),q

def own_sessions_exam(inventory,baseline,candidate,progress=None,cancel=None):
    reports=[];backgrounds=[];benefit=0
    for item in inventory:
        check_work(cancel)
        folder=Path(item['folder']);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
        manual=[r for _,r in read_jsonl(folder/'labels.jsonl')]
        autos=[r for _,r in read_jsonl(folder/'auto-labels.jsonl') if r.get('label')=='probable_false_positive']
        if progress:progress('Own-session exam: '+folder.name)
        mount=Mount(**cfg['mount']);old=Engine(mount,baseline);new=Engine(mount,candidate)
        required=[];new_alerts=[];states={};count=0;first_background=None
        original=list(read_jsonl(folder/'decisions.jsonl'))
        for label in manual:
            if label.get('label')!='false_warning':continue
            t=label.get('at_ns',0)
            matches=[(r['at_ns'],a) for _,r in original for a in r.get('alerts',[]) if 0<=t-r['at_ns']<=10_000_000_000]
            if matches:
                stamp,a=max(matches,key=lambda v:v[0]);autos.append({'at_ns':stamp,'track_id':a['track_id'],'label':'manual_false_warning'})
        for p,stamps,ctx,q in causal_frames(folder,cancel):
            count+=1;a=old.update(p,ctx,q,(stamps-stamps.max())/1e9);b=new.update(p,ctx,q,(stamps-stamps.max())/1e9)
            required.extend({'at_ns':a['at_ns'],'alert':v} for v in a['alerts'])
            new_alerts.extend({'at_ns':b['at_ns'],'alert':v} for v in b['alerts'])
            states[b['at_ns']]={'tracks':b['tracks'],'candidates':b['candidates']}
            if first_background is None and q.get('completeness',0)>=.98 and Ground().fit(transform(p,mount)):
                first_background=(folder.name,p.copy(),asdict(mount))
        if count!=len(original):raise ValueError('Incomplete raw replay for '+folder.name)
        if first_background:backgrounds.append(first_background)
        # Preserve every baseline catch in each manually labeled session, plus
        # original useful alerts explicitly tied to a recent manual useful mark.
        unresolved_useful=0
        for label in manual:
            if label.get('label')!='useful_warning':continue
            t=label.get('at_ns',0);old_audio=label.get('last_audio',{}) or {}
            matching=[{'at_ns':r['at_ns'],'alert':v} for _,r in original for v in r.get('alerts',[]) if
                      r['at_ns']<=t and t-r['at_ns']<=10_000_000_000 and
                      (not old_audio or v.get('track_id')==old_audio.get('track_id'))]
            if matching:required.extend(matching)
            else:unresolved_useful+=1
        missing,shorter=protect_alerts(required,new_alerts,states)
        reduced=0
        for label in autos:
            r=next((r for _,r in original if r['at_ns']==label['at_ns']),None)
            if not r:continue
            old_candidate=next((v for v in r.get('candidates',[])+r.get('alerts',[]) if v.get('track_id')==label.get('track_id')),None)
            if not old_candidate:continue
            xy=np.asarray(old_candidate['position_m']);state=states.get(label['at_ns'],{})
            trackmap={v['id']:v for v in state.get('tracks',[])};still_present=False
            for c in state.get('candidates',[]):
                tr=trackmap.get(c['track_id'],{});bounds=tr.get('bounds_m')
                if bounds:still_present|=bool(np.all(xy>=np.array(bounds['lo'][:2])-.5) and np.all(xy<=np.array(bounds['hi'][:2])+.5))
                else:still_present|=bool(np.linalg.norm(xy-np.array(c['position_m']))<2)
            reduced+=not still_present
        benefit+=reduced
        reports.append({'session':folder.name,'frames':count,'manual_labels':len(manual),'baseline_catches':len(required),
                        'lost_baseline_catches':len(missing),'shortened_warnings':len(shorter),'unresolved_useful_labels':unresolved_useful,
                        'manual_missed_obstacles_unresolved':sum(r.get('label')=='missed_obstacle' for r in manual),
                        'probable_false_evidence_reduced':reduced,'passed':not missing and not shorter and not unresolved_useful})
    return {'passed':bool(backgrounds) and all(r['passed'] for r in reports),'sessions':reports,'probable_false_evidence_reduced':benefit},backgrounds

def run_exam(asset_root,record_root,baseline,candidate,progress=None,cancel=None):
    baseline=canonical_policy(baseline);candidate=canonical_policy(candidate)
    result={'schema_version':1,'suite_id':SUITE_ID,'baseline_policy':baseline,'candidate_policy':candidate,
            'baseline_digest':policy_digest(baseline),'candidate_digest':policy_digest(candidate),'status':'incomplete','eligible':False,'categories':{},
            'warning':'Passing this finite regression exam does not establish real-world safety or zero missed hazards.'}
    try:
        check_work(cancel)
        manifest,asset_signature=validate_assets(asset_root,cancel=cancel);inventory=session_inventory(record_root)
        result.update(asset_signature=asset_signature,session_inventory=inventory)
        result['categories']['bikescenes']=bike_exam(asset_root,manifest,baseline,candidate,progress,cancel)
        backgrounds=[]
        for case in manifest['hk_mems']['cases']:
            if case['role']=='insertion_background':backgrounds.append((case['file'],np.load(Path(asset_root)/case['file'],allow_pickle=False)['points'],case['height_m']))
        own,own_backgrounds=own_sessions_exam(inventory,baseline,candidate,progress,cancel)
        result['categories']['own_sessions']=own
        result['categories']['hk_insertions']=injection_exam(backgrounds,baseline,candidate,manifest['insertion_recipe'],progress,cancel)
        result['categories']['own_insertions']=injection_exam(own_backgrounds,baseline,candidate,manifest['insertion_recipe'],progress,cancel) if own_backgrounds else {'passed':False,'reason':'No complete own M1 session is available for insertion cases'}
        result['categories']['bus_occlusion']=bus_exam(asset_root,manifest,baseline,candidate)
        from adversarial_exam import run_corpus
        corpus=json.loads((Path(asset_root)/manifest['adversarial_corpus']).read_text(encoding='utf-8'))
        result['categories']['moving_geometry_and_sparse_targets']=run_corpus(asset_root,corpus,baseline,candidate,progress,cancel)
        passed=all(v.get('passed') is True for v in result['categories'].values())
        result['status']='passed' if passed else 'rejected'
        result['eligible']=bool(passed and own['probable_false_evidence_reduced']>0 and baseline!=candidate)
        if not result['eligible']:result['reason']='Regression, missing coverage, or no demonstrated improvement in linked false-alarm evidence. Warning count alone is not a benefit.'
    except WorkPaused:raise
    except Exception as e:result['error']=str(e);result['reason']='Required exam category unavailable or invalid; adoption prohibited.'
    result['exam_digest']=signature_digest({k:v for k,v in result.items() if k!='exam_digest'})
    return result

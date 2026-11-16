"""Offline own-drive atlas pipeline. Source evidence is never modified.

Extraction, revisit decisions, graph solving, static reconstruction and activation
are separate restartable stages. Legacy maps remain on disk in their old format.
"""
from collections import Counter
from dataclasses import asdict
import bisect, json, math, hashlib, sqlite3
import runtime_clock as time
from pathlib import Path
import numpy as np
from fcw_engine import Ground,Mount,transform,deskew,ego_delta
from recording import frames
from offline_jobs import check_work
from route_map import features,rotation_z
from .contracts import Keyframe,Constraint,identity,checked_pose
from .store import AtlasStore
from .places import DescriptorIndex,PlaceMatcher,SequenceRecognizer,ICPRegistration,transform_matrix,angle_between
from .graph import PoseGraph,inverse
from .static_map import StaticMap,VERSION as MAP_VERSION
from .backends import BackendProfile
from .calibration_replay import checked_timing,RetimedFrames

EXTRACTION_VERSION='m1-own-drive-keyframes-v3-physical-lineage'


def drive_lineage(folder,data_root,*,allow_synthetic=False):
    """Conservative physical-drive/mount lineage; never infer it from session IDs.

    Actual group evidence comes from a fresh completed SourceCatalog revision and
    source-bound DatasetStore driving reviews. A known group with unknown mount
    period is useful provenance but cannot establish independent map persistence.
    Synthetic identities require BOTH explicit fixture metadata and test opt-in.
    """
    folder=Path(folder).resolve();data_root=Path(data_root).resolve()
    cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    raw_epoch=cfg.get('mount_epoch');epoch=raw_epoch.get('id') if isinstance(raw_epoch,dict) else raw_epoch
    known_epoch=isinstance(epoch,str) and 0<len(epoch)<=200 and not epoch.startswith('legacy-unknown:')
    result={'schema_version':1,'verified':False,'physical_drive_group':None,
            'mount_epoch':epoch if known_epoch else None,'mount_epoch_known':known_epoch,
            'source_route_ids':[],'persistence_eligible':False,'basis':'unverified',
            'synthetic_allowed':False,'evidence':{},'reason':'Physical driving episode has not been independently attributed'}
    if cfg.get('synthetic_fixture'):
        group=cfg.get('synthetic_drive_group')
        if allow_synthetic and isinstance(group,str) and 0<len(group)<=256:
            result.update(verified=True,physical_drive_group=group,basis='synthetic_fixture',synthetic_allowed=True,
                          persistence_eligible=known_epoch,
                          reason='Explicit synthetic test group; not real driving evidence')
        return result
    path=data_root/'learning/catalog.sqlite'
    if not path.is_file():return result
    try:
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            rows=db.execute('''SELECT r.result FROM catalog_sessions s JOIN catalog_revisions r ON r.id=s.active_revision
                WHERE r.state='complete' AND (s.folder=? OR s.id=?)''',(str(folder),cfg.get('session_id',''))).fetchall()
        finally:db.close()
        if len(rows)!=1:result['reason']='No unique completed source catalog entry';return result
        catalog=json.loads(rows[0][0])
        from learning.catalog import inventory_cache_key
        if catalog.get('inventory_cache_key')!=inventory_cache_key(folder):
            result['reason']='Source catalog is stale for this recording';return result
        routes=catalog.get('comma_route_ids')
        if (not isinstance(routes,list) or not routes or any(not isinstance(r,str) or not r for r in routes) or
            len(routes)!=len(set(routes)) or catalog.get('capture_mode')!='live' or catalog.get('synthetic_fixture')):
            result['reason']='Recording routes are unknown or nonphysical';return result
        result['source_route_ids']=sorted(routes)
        result['evidence']={'catalog_revision':catalog.get('catalog_revision'),
                            'catalog_input_signature':catalog.get('input_signature')}
        if (not catalog.get('mount_epoch_known') or catalog.get('mount_epoch')!=result['mount_epoch']):
            result['mount_epoch_known']=False;result['mount_epoch']=None
        if not (data_root/'learning/radar-datasets/registry.sqlite').is_file():return result
        from learning.datasets import DatasetStore
        sources=DatasetStore(data_root).sources();groups=set();reviews=[]
        for route in routes:
            matched=[s for s in sources if s.get('source_route_id')==route]
            if not matched or any(s.get('qualification_eligible') is not True for s in matched):
                result['reason']='A route lacks complete verified physical-drive evidence';return result
            for source in matched:
                review=source.get('provenance',{}).get('driving_review',{})
                group=review.get('physical_drive_group')
                if (review.get('verified') is not True or review.get('is_driving') is not True or
                    not isinstance(group,str) or not 0<len(group)<=256 or review.get('issues')):
                    result['reason']='Physical driving episode review is unknown or invalid';return result
                groups.add(group);reviews.append({'source_id':source['id'],'review_digest':identity(review)})
        if len(groups)!=1:result['reason']='Recording spans different or ambiguous physical drives';return result
        result.update(verified=True,physical_drive_group=next(iter(groups)),basis='reviewed_physical_drive',
                      persistence_eligible=result['mount_epoch_known'],
                      reason='Verified physical driving episode and mount period' if result['mount_epoch_known'] else 'Physical drive known; mount period remains unknown')
        result['evidence']['driving_reviews']=sorted(reviews,key=lambda r:r['source_id'])
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as error:
        result.update(verified=False,physical_drive_group=None,persistence_eligible=False,
                      reason='Lineage evidence unavailable: '+type(error).__name__)
    return result

def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def write_atlas_trace(folder,store,extraction,revision):
    poses=store.revision_poses(revision);segments={}
    for frame in store.extraction_keyframes(extraction):
        if frame.id not in poses:continue
        entry=poses[frame.id];T=np.asarray(entry['map_from_body']);key=(entry['component'],frame.segment_id)
        segments.setdefault(key,[]).append({'at_ns':frame.at_ns,'position_m':T[:3,3].tolist(),'yaw_rad':float(math.atan2(T[1,0],T[0,0])),'keyframe_id':frame.id})
    write_json(Path(folder)/'atlas-route.json',{'schema_version':1,'revision':revision,'source':'corrected_map_keyframes','not_ground_truth':True,
               'segments':[{'component':key[0],'source_segment':key[1],'poses':value} for key,value in segments.items()],
               'note':'Different components have separate origins. A graph correction is an estimate, not surveyed ground truth.'})

def source_signature(folder):
    names={'session.json','final-state.json','frame-contexts.jsonl','decisions.jsonl','comma.jsonl','raw-capture-index.json'}
    paths=[p for p in Path(folder).iterdir() if p.is_file() and (p.name in names or p.name.startswith(('lidar-','route-journal-')) and p.suffix in ('.pcap','.jsonl'))]
    return {p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(paths)}

def session_identity(folder,cfg,frame_provider=None):
    if cfg.get('session_id'):return cfg['session_id']
    # Older recordings lack a UUID. Use content/time-bearing source prefixes,
    # not a folder path that changes when a recording is copied or restored.
    evidence={}
    for pattern in ('lidar-*.pcap','comma.jsonl','frame-contexts.jsonl'):
        paths=sorted(Path(folder).glob(pattern))
        if paths:
            with paths[0].open('rb') as f:digest=hashlib.sha256(f.read(1024*1024)).hexdigest()
            evidence[pattern]={'prefix_sha256':digest,'bytes':paths[0].stat().st_size}
    if frame_provider is not None:evidence['provider']=frame_provider.signature(folder)
    return identity({'legacy_config':cfg,'source_evidence':evidence})

def visibility_cloud(points,origin):
    # Preserve the nearest return in each angular bin BEFORE reducing density.
    # Dropping a foreground return and retaining a farther one could otherwise
    # turn an occlusion into false disappearance evidence.
    delta=np.asarray(points)[:,:3]-origin;valid=np.isfinite(delta).all(1)&(np.linalg.norm(delta,axis=1)>.1)
    delta=delta[valid]
    if not len(delta):return None
    ranges=np.linalg.norm(delta,axis=1);step=math.radians(.5)
    bins=np.floor(np.column_stack((np.arctan2(delta[:,1],delta[:,0]),np.arctan2(delta[:,2],np.hypot(delta[:,0],delta[:,1]))))/step).astype(int)
    order=np.argsort(ranges,kind='stable');_,indices=np.unique(bins[order],axis=0,return_index=True);selected=order[indices]
    angles=(bins[selected]+.5)*step;distance=ranges[selected]
    direction=np.column_stack((np.cos(angles[:,1])*np.cos(angles[:,0]),np.cos(angles[:,1])*np.sin(angles[:,0]),np.sin(angles[:,1])))
    return origin+direction*distance[:,None]

class TrajectoryLookup:
    def __init__(self,folder):
        self.path=Path(folder)/'route-trajectory.jsonl';self.offset=0;self.current=None;self.peek=None;self.last_query=None
    def at(self,at_ns):
        if not self.path.exists():return None
        if self.last_query is not None and at_ns<self.last_query:self.offset=0;self.current=None;self.peek=None
        self.last_query=at_ns
        # A byte-offset cursor avoids retaining hours of 20Hz pose dictionaries.
        # The short-lived file handle also closes cleanly on cancellation.
        with self.path.open('rb') as f:
            f.seek(self.offset)
            while True:
                if self.peek is None:
                    line=f.readline()
                    if not line:break
                    self.offset=f.tell()
                    try:self.peek=json.loads(line)
                    except ValueError:continue
                if self.peek['at_ns']>at_ns:break
                self.current=self.peek;self.peek=None
        if self.current is None or not 0<=at_ns-self.current['at_ns']<=200_000_000:return None
        return self.current

def motion_exclusions(folder):
    path=Path(folder)/'decisions.jsonl';result={}
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:r=json.loads(line)
                except ValueError:continue
                boxes=[t['bounds_m'] for t in r.get('tracks',[]) if 'bounds_m' in t and (np.linalg.norm(t.get('velocity',[0,0]))>1 or t.get('association_ambiguous') or t.get('hits',3)<3)]
                if boxes:result[r['at_ns']]=boxes
    return result

def import_session(folder,store,cancel=None,calibration_override=None,frame_provider=None,registration=None,timing_override=None,profile=None,allow_synthetic=False):
    folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'));original=cfg['mount']
    mount=Mount(**(calibration_override or original));mount.validate()
    timing=checked_timing(cfg,timing_override);profile=profile or BackendProfile()
    if not (mount.measured and mount.alignment_verified and timing['verified']):raise ValueError('Verified mount, alignment and timing required before atlas extraction')
    lineage=drive_lineage(folder,store.root.parent.parent,allow_synthetic=allow_synthetic)
    calibration_id=identity({'mount':asdict(mount),'timing':timing,'mount_epoch':lineage['mount_epoch']})
    session=session_identity(folder,cfg,frame_provider)
    store.register_session(session,{'recorded_config':cfg});store.locate_session(session,folder)
    registration=registration or profile.registration_factory();provider={'version':'recorded-original-M1'}
    if timing_override is not None:frame_provider=RetimedFrames(timing,frame_provider)
    if frame_provider is not None:
        if not getattr(frame_provider,'version',None) or not callable(getattr(frame_provider,'signature',None)):raise ValueError('An alternate frame provider must declare its version and source signature')
        provider={'version':frame_provider.version,'source_signature':frame_provider.signature(folder)}
    manifest={'version':EXTRACTION_VERSION,'calibration_id':calibration_id,'source_signature':source_signature(folder),'frame_provider':provider,'registration_backend':registration.version,'backend_profile':profile.declaration(store),'drive_lineage':lineage}
    extraction=store.begin_extraction(session,manifest)
    import_report=folder/'atlas-import.json'
    if store.extraction_state(extraction)=='complete':
        try:prior=json.loads(import_report.read_text(encoding='utf-8'))
        except (OSError,ValueError):prior={}
        if prior.get('extraction')==extraction:return {**prior,'already_imported':True}
        return {'session_id':session,'extraction':extraction,'already_imported':True,'keyframes':len(store.extraction_keyframes(extraction))}
    index=profile.index_factory(store);extractor=profile.feature_factory();lookup=TrajectoryLookup(folder);exclusions=motion_exclusions(folder)
    counts=Counter();last_frame=None;last_key=None;segment=0;odom=np.eye(4);previous_cloud=None;first_wall=None;last_route_segment=None
    for raw,stamps,ctx,q in (frame_provider or frames)(folder,cancel):
        check_work(cancel);counts['recorded_frames']+=1;at=ctx['at_ns']
        dt=(at-last_frame)/1e9 if last_frame is not None else None
        reliable=(ctx.get('motion_ready') and q.get('timestamp_valid') is True and q.get('completeness',0)>=.98 and q.get('timing_bound_s',1)<=.05 and
                  math.isfinite(ctx.get('speed_mps',float('nan'))) and 0<=ctx['speed_mps']<70 and math.isfinite(ctx.get('yaw_rate_rps',float('nan'))))
        if not reliable:
            counts['invalid_frame_or_motion']+=1;last_frame=None;last_key=None;previous_cloud=None;segment+=1;continue
        row=lookup.at(at)
        route_segment=(row['stream_epoch'],row['segment_id']) if row else None
        if route_segment!=last_route_segment and last_frame is not None:
            counts['route_motion_discontinuities']+=1;last_key=None;previous_cloud=None;segment+=1;odom=np.eye(4)
        last_route_segment=route_segment
        if dt is None or not 0<dt<=.25:
            if dt is not None:counts['frame_gaps']+=1
            segment+=1;last_key=None;previous_cloud=None;odom=np.eye(4)
        elif row is None:
            movement,angle=ego_delta(ctx['speed_mps'],ctx['yaw_rate_rps'],dt);odom[:3,3]+=odom[:3,:3]@np.r_[movement,0.];odom[:3,:3]=odom[:3,:3]@rotation_z(angle)
        if row is not None:
            odom=transform_matrix(rotation_z(row['yaw_rad']),np.array([row['x_m'],row['y_m'],0.]))
        last_frame=at
        if last_key and at-last_key.at_ns<1_000_000_000:
            prior_pose=np.asarray(last_key.odom_from_body)
            if np.linalg.norm(odom[:3,3]-prior_pose[:3,3])<5. and angle_between(odom,prior_pose)<math.radians(5):continue
        body=transform(raw,mount)
        if len(stamps)==len(raw):body=deskew(body,(stamps-stamps.max())/1e9,ctx['speed_mps'],ctx['yaw_rate_rps'])
        # Exclusion boxes are expressed in the original calibrated body frame.
        old_mount=Mount(**original);boxes=[]
        for box in exclusions.get(q.get('recorded_fusion_at_ns',at),[]):
            lo,hi=np.asarray(box['lo'])-.3,np.asarray(box['hi'])+.3
            if calibration_override:
                corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                from telemetry import rotation
                sensor=(corners-old_mount.xyz)@rotation(np.radians(old_mount.rpy_deg));corners=transform(sensor,mount);lo,hi=corners.min(0),corners.max(0)
            boxes.append({'lo':lo,'hi':hi})
        extracted=extractor.extract(body,boxes);cloud=extracted.get('points')
        if cloud is None:counts[extracted.get('reason','feature_extraction_unavailable')]+=1;continue
        if len(cloud)<80:counts['insufficient_static_structure']+=1;continue
        edge_data=None
        if last_key:
            elapsed=(at-last_key.at_ns)/1e9
            if elapsed>2.5:
                segment+=1;last_key=None;previous_cloud=None;counts['registration_gap']+=1
            else:
                seed=inverse(np.asarray(last_key.odom_from_body))@odom
                fit=registration.align(cloud,previous_cloud,(seed[:3,:3],seed[:3,3]))
                if fit:
                    measured=transform_matrix(fit['R'],fit['t'])
                    if np.linalg.norm(measured[:3,3]-seed[:3,3])<=1.5 and angle_between(measured,seed)<=.12:
                        edge_data=(last_key,seed,measured,fit)
                if edge_data is None:
                    segment+=1;last_key=None;previous_cloud=None;counts['registration_rejected']+=1
        source={'context_line':q.get('context_line'),'first_device_ns':q.get('first_device_ns'),
                'raw_capture_completion_part':q.get('raw_capture_completion_part'),'frame_at_ns':at,'recorded_fusion_at_ns':q.get('recorded_fusion_at_ns',at),'sensor_origin_body':list(mount.xyz),
                'extraction_version':EXTRACTION_VERSION,'drive_lineage':lineage}
        vis=visibility_cloud(body,np.asarray(mount.xyz))
        if vis is not None:source['visibility_cloud_hash']=store.put_cloud(vis)
        wall=q.get('raw_capture_last_wall_ns') or cfg.get('capture_started_wall_ns',0)
        if first_wall is None:first_wall=wall
        identifier=identity({'extraction':extraction,'frame_at_ns':at,'first_device_ns':q.get('first_device_ns')})
        location=row.get('location') if row else ctx.get('location')
        frame=Keyframe(identifier,session,f'{extraction}:{segment}',at,store.put_cloud(cloud),calibration_id,odom.tolist(),source,
                       quality={'capture_wall_ns':wall,'completeness':q['completeness'],'timing_bound_s':q['timing_bound_s'],
                                'speed_mps':ctx['speed_mps'],'deskew':'planar-wheel-yaw','extraction':extraction},location=location,descriptor_version=index.version)
        store.add_keyframe(frame,extraction);index.add(frame,cloud);counts['keyframes']+=1
        if edge_data:
            previous,seed,measured,fit=edge_data
            for kind,T,std in (('odometry',seed,[.5,.5,.5,.08,.08,.08]),('scan_registration',measured,[.15,.15,.2,.04,.04,.04])):
                edge_id=identity({'source':previous.id,'target':frame.id,'kind':kind,'backend':registration.version})
                edge=Constraint(edge_id,previous.id,frame.id,T.tolist(),std,kind,
                                {'registration_backend':registration.version,'overlap':fit['overlap'],'rmse_m':fit['rmse_m'],'uncertainty_kind':'engineering_weights'},'accepted')
                store.add_constraint(edge)
        last_key=frame;previous_cloud=cloud
    if not counts['keyframes']:return {'session_id':session,'extraction':extraction,'status':'unavailable','reason':'No eligible registered keyframes; previous evidence retained','diagnostics':dict(counts)}
    store.complete_extraction(extraction)
    final=json.loads((folder/'final-state.json').read_text(encoding='utf-8'))
    processed=final.get('stats',{}).get('processed_frames')
    result={'session_id':session,'extraction':extraction,'status':'complete','keyframes':counts['keyframes'],'diagnostics':dict(counts),
            'drive_lineage':lineage,'processed_frames_in_session':processed,'raw_frame_coverage_fraction':min(1.,counts['recorded_frames']/processed) if processed else None,
            'recording_status':final.get('recording'),'coverage_note':'3D mapping uses preserved lidar frames. Compact GPS/motion logs can continue beyond the raw-data budget, but cannot replace missing lidar returns.'}
    write_json(import_report,result);return result

def discover_revisits(store,extraction,cancel=None,reference_revision=None,profile=None):
    reference_revision=reference_revision or store.current_revision()
    if not reference_revision:return {'confirmed_sequences':0,'constraints':0,'reason':'no reference poses available'}
    profile=profile or BackendProfile();matcher=PlaceMatcher(store,index=profile.index_factory(store),registration=profile.registration_factory());recognizer=SequenceRecognizer();created=set();confirmed=0;diagnostics=Counter();localized=set()
    existing={e['id'] for e in store.constraints(accepted_only=False)}
    queries=store.extraction_keyframes(extraction);all_frames=list(store.active_keyframes());revision=reference_revision;tracking=None;last_global=None
    for query in queries:
        check_work(cancel)
        excluded={f.id for f in all_frames if f.session_id==query.session_id and abs(f.at_ns-query.at_ns)<30_000_000_000}
        cloud=store.load_cloud(query.cloud_hash);result=None
        if tracking and tracking['segment']==query.segment_id and 0<query.at_ns-tracking['at_ns']<=2_500_000_000 and query.at_ns-last_global<10_000_000_000:
            predicted=tracking['map_from_odom']@np.asarray(query.odom_from_body)
            result=matcher.match(cloud,query.location,exclude_ids=excluded,cancel=cancel,revision=revision,map_from_query_seed=predicted,seed_component=tracking['component'])
        if result is None or result['status']!='candidate':
            result=matcher.match(cloud,query.location,exclude_ids=excluded,cancel=cancel,revision=revision);last_global=query.at_ns
        if result['status']=='candidate':
            match=result['matches'][0];tracking={'map_from_odom':np.asarray(match['map_from_query'])@inverse(np.asarray(query.odom_from_body)),
                                                'component':match['component'],'at_ns':query.at_ns,'segment':query.segment_id}
        else:tracking=None
        # This observation ID is kept with every match so a confirmation group
        # contains real separated query frames rather than repeated copies.
        for match in result['matches']:match['query_id']=query.id
        confirmation=recognizer.observe(query.at_ns,query.segment_id,query.odom_from_body,result)
        diagnostics[confirmation['status']]+=1
        store.event('place_match',{'query':query.id,'revision':revision,'status':confirmation['status'],'candidate_status':result['status'],
                                   'matches':result['matches'][:8],'rejected':result['rejected'],'confirmations':confirmation.get('confirmations',0)})
        if confirmation['status']!='confirmed':continue
        chosen=next(h for h in recognizer.hypotheses if h['component']==confirmation['component'])
        window=chosen['matches'];matches=[window[i] for i in (0,(len(window)-1)//2,len(window)-1)]
        query_frames=[store.keyframe(m['query_id']) for m in matches]
        if len({f.id for f in query_frames})<3:continue
        positions=np.array([np.asarray(f.odom_from_body)[:3,3] for f in query_frames]);baseline=float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
        if baseline<2 or query_frames[-1].at_ns-query_frames[0].at_ns<1_000_000_000:continue
        group=identity({'queries':[m['query_id'] for m in matches],'references':[m['reference_id'] for m in matches],'backend':matcher.registration.version})
        identifiers=[identity({'pair':sorted([m['query_id'],m['reference_id']]),'backend':matcher.registration.version}) for m in matches]
        localized.update(m['query_id'] for m in matches)
        if any(identifier in existing for identifier in identifiers):recognizer.reset();continue
        for match,edge_id in zip(matches,identifiers):
            if edge_id in created:continue
            constraint=Constraint(edge_id,match['reference_id'],match['query_id'],match['reference_from_query'],[.2,.2,.25,.05,.05,.05],'revisit',
                                  {'confirmation_group':group,'confirmations':3,'baseline_m':baseline,'overlap':match['overlap'],
                                   'reciprocal_overlap':match['reciprocal_overlap'],'rmse_m':match['rmse_m'],'reference_revision':revision},'accepted')
            store.add_constraint(constraint);created.add(edge_id)
            existing.add(edge_id)
        confirmed+=1
        recognizer.reset()
    return {'confirmed_sequences':confirmed,'confirmed_keyframes':len(localized),'constraints':len(created),'diagnostics':dict(diagnostics)}

def update_atlas(folder,map_root,cancel=None,calibration_override=None,frame_provider=None,allow_synthetic=False,timing_override=None,profile=None):
    folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    report={'schema_version':2,'status':'unavailable','warning_override':False,'backend':'own-drive-atlas-v2'}
    def finish():write_json(folder/'map-review.json',report);return report
    if not (folder/'final-state.json').exists():report['reason']='completed recording required';return finish()
    if not cfg.get('map_enabled',True):report['reason']='mapping disabled';return finish()
    if cfg.get('capture_mode')!='live' or cfg.get('synthetic_fixture') and not allow_synthetic:
        report['reason']='real live recording required';return finish()
    mount=calibration_override or cfg.get('mount',{});timing=checked_timing(cfg,timing_override)
    if not (mount.get('measured') and mount.get('alignment_verified') and timing['verified']):
        report['reason']='verified mounting, alignment and timing required';return finish()
    store=AtlasStore(Path(map_root)/'atlas-v2')
    profile=profile or BackendProfile();graph=profile.graph_factory();maps=profile.map_factory(store);profile_id=profile.fingerprint(store)
    imported=import_session(folder,store,cancel,calibration_override,frame_provider,timing_override=timing_override,profile=profile,allow_synthetic=allow_synthetic);report['extraction']=imported
    if imported.get('status')=='unavailable':report['reason']=imported['reason'];return finish()
    # Reference poses for every active keyframe allow a revisit within the first
    # recording to be found too. This draft is never exposed to live map readers.
    initial_generation=store.generation();initial_frames=list(store.active_keyframes());initial_ids={f.id for f in initial_frames}
    initial_edges=[e for e in store.constraints() if e['source_id'] in initial_ids and e['target_id'] in initial_ids]
    initial=graph.solve(initial_frames,initial_edges,cancel=cancel)
    if not initial['validated']:report.update(status='rejected',reason=initial['reason']);return finish()
    initial_generation=store.apply_decisions(initial_generation,initial['rejected_constraints'])
    draft_manifest={k:v for k,v in initial.items() if k not in ('poses','rejected_constraints')};draft_manifest.update(purpose='localization_reference_draft',map_backend=maps.version,extractor_version=EXTRACTION_VERSION,backend_profile=profile_id,allow_synthetic=bool(allow_synthetic))
    reference=store.publish_revision(graph.version,initial_generation,initial['poses'],draft_manifest,activate=False)
    revisits=discover_revisits(store,imported['extraction'],cancel,reference,profile);report['revisits']=revisits
    generation=store.generation();active=list(store.active_keyframes());ids={f.id for f in active}
    current=store.current_revision();old=store.revision_manifest(current) if current else {}
    if old.get('source_generation')==generation and old.get('algorithm')==graph.version and old.get('map_backend')==maps.version and old.get('extractor_version')==EXTRACTION_VERSION and old.get('backend_profile')==profile_id and old.get('allow_synthetic') is bool(allow_synthetic):
        static=json.loads(store.product(current,'map_report').read_text(encoding='utf-8'))
        report.update(status='complete',revision=current,already_processed=True,static_map=static,graph={k:v for k,v in old.items() if k!='edge_diagnostics'},
                      reason='Current evidence already has a checked map revision; repeated processing adds no visits.',
                      promoted_static_voxels=static['promoted_static_voxels'],pending_voxels=static['pending_voxels'],forgotten_voxels=static['forgotten_voxels'],
                      eligible_keyframes=imported['keyframes'],localized_keyframes=revisits.get('confirmed_keyframes',0),
                      coverage_percent=100*revisits.get('confirmed_keyframes',0)/max(1,imported['keyframes']),new_submaps=0)
        write_atlas_trace(folder,store,imported['extraction'],current);return finish()
    constraints=[e for e in store.constraints() if e['source_id'] in ids and e['target_id'] in ids]
    solved=graph.solve(active,constraints,cancel=cancel);report['graph']={k:v for k,v in solved.items() if k not in ('poses','edge_diagnostics')}
    if not solved['validated']:report['status']='rejected';report['reason']=solved['reason'];return finish()
    generation=store.apply_decisions(generation,solved['rejected_constraints'])
    manifest={k:v for k,v in solved.items() if k not in ('poses','rejected_constraints')};manifest.update(map_backend=maps.version,extractor_version=EXTRACTION_VERSION,backend_profile=profile_id,allow_synthetic=bool(allow_synthetic))
    revision=store.publish_revision(graph.version,generation,solved['poses'],manifest,activate=False)
    static=maps.rebuild(revision,cancel,allow_synthetic=allow_synthetic);check_work(cancel)
    store.activate_revision(revision,'checked own-drive map rebuild',require_map=True,require_current_generation=True)
    write_atlas_trace(folder,store,imported['extraction'],revision)
    report.update(status='complete',revision=revision,static_map=static,reason='Own-drive evidence accumulated; map remains advisory.',
                  promoted_static_voxels=static['promoted_static_voxels'],pending_voxels=static['pending_voxels'],forgotten_voxels=static['forgotten_voxels'],
                  eligible_keyframes=imported['keyframes'],localized_keyframes=revisits.get('confirmed_keyframes',0),
                  coverage_percent=100*revisits.get('confirmed_keyframes',0)/max(1,imported['keyframes']),new_submaps=len(solved['components']),
                  coordinate_system='versioned local 3D components; horizontal GNSS anchoring only when independently consistent')
    return finish()

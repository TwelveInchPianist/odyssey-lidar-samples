"""Bounded read-only extraction of empirical sensor-calibration evidence.

Raw lidar geometry is never seeded or deskewed by comma motion. Comma rows are
independent comparison observations, not truth. Outputs are unapplied candidate
measurements; no flags, configuration, recordings or maps are changed here.
"""
from collections import Counter
import copy, hashlib, json, math
import runtime_clock as time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from frames import FrameAssembler,decode_with_time
from m1_protocol import read_pcap,valid_msop,metadata
from offline_jobs import check_work
from radar import RadarBuffer,normalize_radar_record
from telemetry import Telemetry
from fcw_engine import point_components
from mapping.journal import journal_parts,MOTION_SERVICES

VERSION='sensor-selfcal-capture-v6-central-road-temporal-association'
DEFAULT_BUDGET={'max_frames':1200,'max_wall_s':1200.,'window_frames':100,'stride_frames':100,
                'max_radar_pairs':3000,'max_context_rows':200000,'max_comma_rows':1000000,
                'max_points':30000}
MAX_LINE_BYTES=2*1024*1024

def _finite(value):return type(value) in (int,float) and math.isfinite(value)
def _stamp(value):return type(value) is int and value>0
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _epoch(value):return value.get('id') if isinstance(value,dict) else value if isinstance(value,str) and value else None

def _safe_file(path):
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('Linked sources are not supported')
    if not path.is_file():raise ValueError('Missing source: '+path.name)

def _fingerprints(folder,cancel):
    paths=[folder/'session.json',folder/'final-state.json',folder/'frame-contexts.jsonl']
    paths+=[folder/name for name in ('comma.jsonl','raw-capture-index.json') if (folder/name).is_file()]
    paths+=journal_parts(folder)
    paths+=sorted(folder.glob('lidar-*.pcap'));result={}
    for path in paths:
        check_work(cancel);_safe_file(path);stat=path.stat()
        with path.open('rb') as stream:
            if path.suffix=='.json' and stat.st_size<=MAX_LINE_BYTES:
                content=stream.read();scope='full_sha256'
            else:
                content=stream.read(65536);stream.seek(max(0,stat.st_size-65536));content+=stream.read(65536);scope='bounded_edge_sha256_64KiB_each'
        result[path.name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':hashlib.sha256(content).hexdigest(),'digest_scope':scope}
    return result

class _Contexts:
    def __init__(self,path,limit,cancel,deadline):
        self.stream=path.open('rb');self.offsets=[];self.metadata=[]
        while True:
            check_work(cancel)
            if time.monotonic()>deadline:raise TimeoutError('Context indexing exhausted the analysis time budget')
            offset=self.stream.tell();line=self.stream.readline(MAX_LINE_BYTES+1)
            if not line:break
            if len(line)>MAX_LINE_BYTES:raise ValueError('Frame context exceeds supported line size')
            self.offsets.append((offset,len(line)))
            if len(self.offsets)>limit:raise ValueError('Frame-context index exceeds this session budget')
            try:
                row=json.loads(line);ctx=row['context'];quality=row['quality'];key=quality['first_device_ns']
                if type(key) is not int:raise ValueError('device key invalid')
                self.metadata.append({'line':len(self.offsets),'key':key,'speed':ctx.get('speed_mps'),'yaw':ctx.get('yaw_rate_rps'),
                                      'stationary':ctx.get('stationary_confirmed') is True,
                                      'receipt_start':quality.get('first_receipt_ns'),'receipt_end':quality.get('last_receipt_ns')})
            except (ValueError,TypeError,KeyError,AttributeError):self.metadata.append(None)
    def get(self,line):
        if type(line) is not int or not 1<=line<=len(self.offsets):return {}
        offset,length=self.offsets[line-1];self.stream.seek(offset)
        try:value=json.loads(self.stream.read(length))
        except (ValueError,TypeError):return {}
        return value if isinstance(value,dict) else {}
    def close(self):self.stream.close()

    def select(self,limits):
        """Choose contiguous windows across the full drive before any decoding.

        Motion metadata chooses informative windows only; it is never supplied
        as a seed or deskew reference for independent lidar registration.
        """
        width=limits['window_frames'];step=limits['stride_frames'];candidates=[]
        counts=Counter(row['key'] for row in self.metadata if row)
        for start in range(0,len(self.metadata),step):
            rows=[row for row in self.metadata[start:start+width] if row and counts[row['key']]==1]
            if not rows:continue
            speed=[row['speed'] for row in rows if _finite(row['speed'])]
            yaw=[row['yaw'] for row in rows if _finite(row['yaw'])]
            score=(float(np.std(speed)) if speed else 0)+20*(float(np.std(yaw)) if yaw else 0)
            parked=sum(row['stationary'] and _finite(row['speed']) and 0<=row['speed']<.2 for row in rows)/len(rows)
            steady=sum(_finite(row['speed']) and _finite(row['yaw']) and 5<=row['speed']<=35 and abs(row['yaw'])<=.035 for row in rows)/len(rows)
            candidates.append({'start':start,'rows':rows,'score':score,'parked_fraction':parked,'steady_fraction':steady})
        number=min(len(candidates),max(1,math.ceil(limits['max_frames']/width)))
        parked=next((item for item in candidates if item['start']<=300 and item['parked_fraction']>=.8),None)
        selected=[parked] if parked is not None and number else []
        remaining=[item for item in candidates if item is not parked]
        slots=min(number-len(selected),len(remaining))
        for slot,group in enumerate(np.array_split(np.arange(len(remaining)),slots) if slots else []):
            options=[remaining[int(i)] for i in group]
            steady=[item for item in options if item['steady_fraction']>=.7]
            if slot%2==0 and steady:
                choice=max(steady,key=lambda item:(item['steady_fraction'],-item['score'],-item['start']))
                choice['purpose']='steady_geometry'
            else:
                choice=max(options,key=lambda item:(item['score'],-item['start']));choice['purpose']='motion_excitation'
            selected.append(choice)
        output={}
        for window,item in enumerate(sorted(selected,key=lambda item:item['start'])):
            for row in item['rows']:output[row['key']]=(row['line'],window)
        return output,{'selection':'temporally_spread_balanced_geometry_and_motion_windows',
            'total_context_rows':len(self.metadata),'candidate_windows':len(candidates),'selected_windows':len(selected),
            'selected_context_rows':len(output),'selected_context_ranges':[[item['rows'][0]['line'],item['rows'][-1]['line']] for item in selected],
            'parked_window_reserved':parked is not None and parked in selected,
            'moving_window_budget':len(selected)-int(parked is not None and parked in selected),
            'window_purposes':[item.get('purpose','parked_reference_candidate') for item in sorted(selected,key=lambda item:item['start'])],
            'duplicate_device_keys_excluded':sum(count>1 for count in counts.values()),
            'metadata_only_selection_not_registration_seed':True}

def _selected_capture_parts(folder,index,selected):
    """Skip minute-sized PCAP parts only with a complete, compatible index.

    Selected frame receipt intervals include both sides of a part boundary.
    Missing/invalid index metadata always falls back to original full traversal.
    """
    paths=sorted(folder.glob('lidar-*.pcap'));info={'pcap_parts_total':len(paths),'pcap_parts_selected':len(paths),
        'pcap_parts_skipped':0,'pcap_selection':'full_traversal'}
    metadata=folder/'raw-capture-index.json'
    try:
        if not metadata.is_file() or metadata.stat().st_size>MAX_LINE_BYTES:raise ValueError('missing or excessive capture index')
        contents=json.loads(metadata.read_text(encoding='utf-8-sig'))
        if not isinstance(contents,dict):raise ValueError('capture index must be an object')
        entries=contents.get('segments')
        if not isinstance(entries,list) or len(entries)!=len(paths):raise ValueError('capture index is incomplete')
        by_name={row['file']:row for row in entries if isinstance(row,dict) and isinstance(row.get('file'),str)}
        if set(by_name)!={path.name for path in paths}:raise ValueError('capture filenames differ from index')
        intervals=[]
        for row in index.metadata:
            if row is None or row['key'] not in selected:continue
            first,last=row['receipt_start'],row['receipt_end']
            if not _stamp(first) or not _stamp(last) or first>last:raise ValueError('selected frame receipt interval unavailable')
            intervals.append((first,last))
        if len(intervals)!=len(selected):raise ValueError('selected frame index ambiguous')
        chosen=[];previous=None
        for path in paths:
            row=by_name[path.name];first=row.get('first_receipt_ns');last=row.get('last_receipt_ns')
            if (not _stamp(first) or not _stamp(last) or first>last or type(row.get('bytes')) is not int or row['bytes']!=path.stat().st_size or
                    (previous is not None and first<previous)):
                raise ValueError('capture interval or byte size invalid')
            previous=last
            if any(first<=end and start<=last for start,end in intervals):chosen.append(path)
        # Every selected interval must overlap some indexed capture. Otherwise
        # the metadata may describe a different clock/source and cannot prune.
        if intervals and not chosen:raise ValueError('selected intervals do not overlap indexed captures')
        for start,end in intervals:
            if not any(by_name[p.name]['first_receipt_ns']<=end and start<=by_name[p.name]['last_receipt_ns'] for p in chosen):
                raise ValueError('selected interval outside capture index')
        info.update(pcap_parts_selected=len(chosen),pcap_parts_skipped=len(paths)-len(chosen),pcap_selection='validated_original_receipt_intervals')
        return chosen,info
    except (ValueError,TypeError,KeyError,OSError) as error:
        info['pcap_index_fallback_reason']=str(error);return paths,info

def selected_frames(folder,index,selected,cfg,cancel,deadline,diagnostics):
    """Scan raw packets once; decode only selected whole continuous windows."""
    assembler=None;context=None;seen=set();completed=set()
    parts,part_selection=_selected_capture_parts(folder,index,selected)
    for key,value in part_selection.items():diagnostics[key]=value
    for path in parts:
        for wall,source,port,payload in read_pcap(path):
            check_work(cancel)
            if time.monotonic()>deadline:raise TimeoutError('Raw scan traversal exhausted analysis time budget')
            diagnostics['raw_packets_scanned']+=1
            if port!=cfg['msop_port'] or source[0]!=cfg['sensor_ip'] or not valid_msop(payload):continue
            sequence=int.from_bytes(payload[4:6],'big')
            if sequence==1:
                diagnostics['raw_scans_seen']+=1
                key=metadata(payload)['lidar_time_ns'];entry=selected.get(key)
                assembler=None;context=None
                if entry is not None and key not in seen:
                    line,window=entry;context=index.get(line);assembler=FrameAssembler();seen.add(key)
                    context['_selection_window']=window;context['_context_line']=line
            if assembler is None:continue
            for packets,q in assembler.push(payload,wall):
                points,stamps=decode_with_time(packets)
                original=context.get('context',{});quality={**context.get('quality',{}),**q,
                    'context_line':context['_context_line'],'selection_window':context['_selection_window'],
                    'raw_capture_completion_part':path.name}
                # Stored receipt timestamps are monotonic; PCAP wall time must
                # never overwrite them with a different clock domain.
                for name in ('first_receipt_ns','last_receipt_ns'):
                    if name in context.get('quality',{}):quality[name]=context['quality'][name]
                completed.add(q['first_device_ns'])
                yield points,stamps,original,quality
                assembler=None
    diagnostics['selected_contexts_without_raw_frames']=len(selected)-len(completed)

def _journal_probe(parts,cancel,deadline):
    """Bounded structural probe, not a claim that sensor measurements are valid."""
    services=set();clock=False;read=0;bytes_read=0
    for path in parts:
        _safe_file(path)
        with path.open('rb') as stream:
            while read<4096 and bytes_read<8*1024*1024:
                check_work(cancel)
                if time.monotonic()>deadline:raise TimeoutError('Journal probe exceeded analysis time budget')
                line=stream.readline(MAX_LINE_BYTES+1)
                if not line:break
                read+=1;bytes_read+=len(line)
                if len(line)>MAX_LINE_BYTES:return False,'journal line exceeds supported size'
                try:event=json.loads(line)
                except (ValueError,TypeError):continue
                if not isinstance(event,dict) or event.get('schema_version')!=1 or not isinstance(event.get('data'),dict):continue
                if event.get('kind')=='clock_exchange':clock=True
                if event.get('kind')=='sample':services.add(event['data'].get('service'))
                if clock and {'carState','liveCalibration'}<=services and services&{'livePose','liveLocationKalman'}:
                    return True,None
        if read>=4096 or bytes_read>=8*1024*1024:break
    return False,'bounded journal prefix lacks clock, carState, calibration or pose records'

def _journal_records(parts,session_id,diagnostics,cancel,deadline):
    """Reconstruct original clock/motion records, retaining explicit gap events.

    This follows mapping.trajectory.route_records' field convention, additionally
    retaining corrupt/missing sequence and invalid-source-time discontinuities
    needed for calibration. GPS/model payloads are not comparison observations.
    """
    previous=-1;gap=False;known_session=session_id
    for path in parts:
        _safe_file(path)
        with path.open('rb') as stream:
            number=0
            while True:
                check_work(cancel)
                if time.monotonic()>deadline:raise TimeoutError('Journal parsing exhausted analysis time budget')
                line=stream.readline(MAX_LINE_BYTES+1)
                if not line:break
                number+=1;diagnostics['journal_records_scanned']+=1
                if len(line)>MAX_LINE_BYTES:raise ValueError('Journal record exceeds supported line size')
                if not line.endswith(b'\n'):diagnostics['journal_truncated_records']+=1;gap=True;continue
                try:
                    event=json.loads(line)
                    if not isinstance(event,dict) or event.get('schema_version')!=1 or type(event.get('seq')) is not int or event['seq']<0 or not isinstance(event.get('data'),dict):raise ValueError('invalid journal schema')
                except (ValueError,TypeError):diagnostics['journal_invalid_records']+=1;gap=True;continue
                source_session=event.get('session_id')
                if not isinstance(source_session,str) or not source_session:raise ValueError('Journal source session identity missing')
                if known_session is None:known_session=source_session
                if source_session!=known_session:raise ValueError('Journal belongs to a different source session')
                sequence=event['seq']
                if sequence<=previous:diagnostics['journal_non_increasing_sequence']+=1;gap=True;continue
                if previous>=0 and sequence!=previous+1:diagnostics['journal_missing_sequence_records']+=sequence-previous-1;gap=True
                previous=sequence;receipt=event.get('receipt_ns');kind=event.get('kind');data=event['data']
                reference={'part':path.name,'line':number,'sequence':sequence}
                if gap:
                    yield {'kind':'offline_evidence_gap','laptop_receipt_ns':receipt,'source_ref':reference}
                    gap=False
                if kind in ('journal_open','stream_restart','source_clock_discontinuity'):
                    yield {'kind':'hello','laptop_receipt_ns':receipt,'commit':data.get('commit'),'source_ref':reference,'route_epoch':event.get('epoch')}
                elif kind=='invalid_source_time':
                    yield {'kind':'offline_evidence_gap','laptop_receipt_ns':receipt,'source_ref':reference}
                elif kind in ('sample','clock_exchange','comma_recording_reference'):
                    if kind=='sample' and data.get('service') not in MOTION_SERVICES:continue
                    item={k:v for k,v in data.items() if k!='location'}
                    original=item.get('laptop_receipt_ns')
                    if _stamp(original) and _stamp(receipt) and original!=receipt:
                        diagnostics['journal_conflicting_receipts']+=1
                        yield {'kind':'offline_evidence_gap','laptop_receipt_ns':receipt,'source_ref':reference};continue
                    if not _stamp(original):
                        # This is the actual receipt recorded by the journal,
                        # not a guessed native/source timestamp.
                        item['laptop_receipt_ns']=receipt;item['receipt_basis']='recorded_journal_envelope'
                    else:item['receipt_basis']='original_recorded_payload'
                    item.update(kind='sample' if kind=='sample' else 'pong' if kind=='clock_exchange' else 'route_reference',source_ref=reference)
                    diagnostics['journal_motion_clock_records']+=1;yield item
    if gap:yield {'kind':'offline_evidence_gap','laptop_receipt_ns':None,'reason':'unknown-time corrupt journal tail'}

class _Comma:
    def __init__(self,path,limit,cancel,deadline):
        self.limit=limit;self.cancel=cancel;self.deadline=deadline;self.count=0;self.diag=Counter()
        self.stream=None;self._journal_iterator=None
        parts=journal_parts(path.parent);supported=False;fallback=None
        if parts:supported,fallback=_journal_probe(parts,cancel,deadline)
        if parts and (supported or not path.is_file()):
            config=json.loads((path.parent/'session.json').read_text(encoding='utf-8-sig'))
            self._journal_iterator=iter(_journal_records(parts,config.get('session_id'),self.diag,cancel,deadline))
            self.source_scope={'kind':'compact_route_journal','files':[p.name for p in parts],
                'structural_probe_supported':supported,'note':fallback or 'Original receipt/source clocks and retained motion/calibration payloads; no modelV2 parsing',
                'raw_bulk_parsed':False,'record_budget_scope':'retained motion/clock records; full traversal also bounded by time'}
        else:
            self.stream=path.open('rb') if path.is_file() else None
            self.source_scope={'kind':'raw_comma' if self.stream else 'unavailable','files':[path.name] if self.stream else [],
                'fallback_reason':fallback or 'compact route journal absent','raw_bulk_parsed':self.stream is not None}
        self.telemetry=Telemetry();self.radar=RadarBuffer();self.epoch=0;self.last_receipt=None;self.last_query=None
        self.next=self._read()
    def reset(self):self.telemetry=Telemetry();self.radar=RadarBuffer();self.epoch+=1;self.diag['epochs']+=1
    def _read(self):
        if self.stream is None and self._journal_iterator is None:return None
        check_work(self.cancel)
        if self.count>=self.limit:raise ValueError('Comma source exceeds analysis record budget')
        if time.monotonic()>self.deadline:raise TimeoutError('Comma parsing exhausted analysis time budget')
        if self._journal_iterator is not None:
            record=next(self._journal_iterator,None)
            if record is not None:self.count+=1
            return record
        line=self.stream.readline(MAX_LINE_BYTES+1)
        if not line:return None
        self.count+=1
        if len(line)>MAX_LINE_BYTES:raise ValueError('Comma record exceeds supported line size')
        try:record=json.loads(line)
        except ValueError:self.diag['malformed']+=1;return {'kind':'invalid'}
        return record if isinstance(record,dict) else {'kind':'invalid'}
    def at(self,at):
        if not _stamp(at):return {},None
        if self.last_query is not None and at<self.last_query:
            self.reset();self.diag['frame_clock_reversals']+=1;self.last_query=at;return {},None
        self.last_query=at
        while self.next is not None:
            item=self.next;receipt=item.get('laptop_receipt_ns')
            if _stamp(receipt) and receipt>at:break
            self.next=self._read()
            if not _stamp(receipt):self.reset();self.diag['missing_receipts']+=1;continue
            if self.last_receipt is not None and receipt<self.last_receipt:self.reset();self.diag['receipt_reversals']+=1
            self.last_receipt=receipt
            try:
                kind=item.get('kind')
                if kind=='offline_evidence_gap':self.reset();self.diag['explicit_evidence_gaps']+=1;continue
                if kind=='hello':self.reset()
                if kind=='pong':
                    values=[item.get(k) for k in ('laptop_send_ns','remote_receive_ns','remote_send_ns','laptop_receipt_ns')]
                    if not all(_stamp(v) for v in values):raise ValueError('invalid clock exchange')
                    self.telemetry.clock.add(*values)
                elif kind in ('sample','hello'):
                    if kind=='sample' and (not isinstance(item.get('service'),str) or not isinstance(item.get('data'),dict)):raise ValueError('malformed sample')
                    self.telemetry.accept(item,receipt)
                self.radar.accept(item,receipt)
            except (ValueError,TypeError,KeyError,AttributeError,OverflowError):self.reset();self.diag['invalid_payloads']+=1
        try:
            context=self.telemetry.context(at);sync=self.telemetry.clock.estimate(at)
            radar=self.radar.latest(at,sync)
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError):self.diag['context_errors']+=1;return {},None
        context['source_epoch']=self.epoch
        return context,radar
    def close(self):
        if self.stream:self.stream.close()
        if self._journal_iterator is not None:self._journal_iterator.close()

def sensor_ground(points):
    """Fit raw sensor-coordinate road plane without a supplied mounting height."""
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]<3:return None
    p=p[:,:3];p=p[np.isfinite(p).all(1)&(p[:,0]>4)&(p[:,0]<40)&(abs(p[:,1])<4)&(p[:,2]>-5)&(p[:,2]<1)]
    if len(p)<100:return None
    p=p[np.argsort(p[:,2])];cells=np.floor(p[:,:2]/.75).astype(int)
    _,indices=np.unique(cells,axis=0,return_index=True);q=p[indices]
    if len(q)<40:return None
    q=q[::max(1,math.ceil(len(q)/1800))];A=np.c_[q[:,:2],np.ones(len(q))]
    central=np.flatnonzero(abs(q[:,1])<2.)
    if len(central)<30 or np.ptp(q[central,0])<10:return None
    rng=np.random.default_rng(91);best=None;count=0
    for _ in range(50):
        chosen=rng.choice(central,3,replace=False)
        if abs(np.linalg.det(A[chosen]))<.1:continue
        abc=np.linalg.solve(A[chosen],q[chosen,2])
        if np.linalg.norm(abc[:2])>.45 or not -4<abc[2]<-.4:continue
        mask=abs(A@abc-q[:,2])<.12;n=int(mask.sum())
        if np.mean(mask[central])>=.65 and n>count:best=mask;count=n
    if best is None or count<40 or count/len(q)<.5:return None
    support=q[best]
    if np.ptp(support[:,0])<10 or np.ptp(support[:,1])<4:return None
    abc=np.linalg.lstsq(A[best],support[:,2],rcond=None)[0]
    rms=float(np.sqrt(np.mean((A[best]@abc-support[:,2])**2)))
    if rms>.1:return None
    normal=np.r_[-abc[:2],1.];normal/=np.linalg.norm(normal)
    # Closest proper rotation mapping observed road normal to sensor leveled up.
    target=np.array([0.,0.,1.]);cross=np.cross(normal,target);c=float(normal@target)
    skew=np.array([[0.,-cross[2],cross[1]],[cross[2],0.,-cross[0]],[-cross[1],cross[0],0.]])
    rotation=np.eye(3)+skew+skew@skew/(1+c)
    return {'road_reference_method':'central_visible_road_v2','lateral_search_half_width_m':4.,'central_support_fraction':float(np.mean(best[central])), 'sensor_abc':abc.tolist(),'sensor_normal':normal.tolist(),'plane_offset_m':float(-abc[2]/np.linalg.norm(np.r_[-abc[:2],1.])),
        'support_points':count,'support_fraction':count/len(q),'fit_rmse_m':rms,'span_x_m':float(np.ptp(support[:,0])),
        'span_y_m':float(np.ptp(support[:,1])),'_rotation':rotation}

def _objects(points,stamps,ground,query_at,max_points,rotation=None):
    p=np.asarray(points,float)[:,:3];times=np.asarray(stamps)
    if times.shape!=(len(p),) or not len(times):return []
    level=p@(ground['_rotation'] if rotation is None else rotation).T
    height=p@np.asarray(ground['sensor_normal'])+ground['plane_offset_m']
    keep=np.isfinite(level).all(1)&(level[:,0]>4)&(level[:,0]<85)&(abs(level[:,1])<18)&(height>.3)&(height<3.8)
    indices=np.flatnonzero(keep)
    indices=indices[::max(1,math.ceil(len(indices)/max_points))]
    if len(indices)<6:return []
    chosen=level[indices];result=[]
    for component in point_components(chosen,.4):
        if len(component)<6:continue
        ids=indices[component];group=level[ids];lo=group.min(0);hi=group.max(0);extent=hi-lo
        if not (.03<=extent[0]<=6 and .25<=extent[1]<=3 and .3<=extent[2]<=3.5):continue
        stamp=int(np.median(times[ids]));span=(int(times[ids].max())-int(times[ids].min()))/1e9
        result.append({'position':np.median(group[:,:2],axis=0),'position_xyz':np.median(group,axis=0),'lo':lo,'hi':hi,'points':len(ids),
            'at_ns':int(query_at+stamp-int(times.max())),'device_ns':stamp,'point_span_s':span})
    return sorted(result,key=lambda item:float(np.linalg.norm(item['position'])))[:64]

class _Objects:
    def __init__(self):self.previous=[];self.next_id=0
    def reset(self):self.previous=[]
    def update(self,objects,motion,at):
        valid=motion.get('valid') is True;increment=motion.get('ego_increment_se2',{})
        if not valid:self.reset()
        angle=float(increment.get('yaw_rad',0));translation=np.asarray(increment.get('translation_m',[0.,0.]),float)
        rotation=np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
        full=motion.get('ego_increment_se3')
        use3d=bool(full and all('position_xyz' in obj for obj in self.previous+objects))
        if use3d:
            rotation3=np.asarray(full['rotation'],float).T;translation3=np.asarray(full['translation_m'],float)
            predicted3=[rotation3@(old['position_xyz']-translation3) for old in self.previous]
            predicted=[p[:2] for p in predicted3]
        else:predicted=[rotation@(old['position']-translation) for old in self.previous]
        options={i:[j for j,old in enumerate(self.previous) if np.linalg.norm(obj['position']-predicted[j])<.65] for i,obj in enumerate(objects)}
        reverse=Counter(j for values in options.values() for j in values)
        for i,obj in enumerate(objects):
            matches=options[i]
            old=self.previous[matches[0]] if len(matches)==1 and reverse[matches[0]]==1 else None
            if old:
                dt=(at-old['last_at'])/1e9
                residual=(obj['position_xyz']-predicted3[matches[0]]) if use3d else obj['position']-predicted[matches[0]]
                obj.update(id=old['id'],first_at=old['first_at'],hits=old['hits']+1,residual_m=float(np.linalg.norm(residual)),
                    ground_velocity=residual[:2]/max(.01,dt))
            else:
                self.next_id+=1;obj.update(id=self.next_id,first_at=at,hits=1,residual_m=None,ground_velocity=None)
            obj.update(last_at=at,age_s=(at-obj['first_at'])/1e9)
        self.previous=objects
        return objects

def _snapshot(snapshot,at):
    if not isinstance(snapshot,dict) or snapshot.get('service')!='liveTracks' or snapshot.get('kind')!='raw_oem_tracks' or snapshot.get('available') is not True:return None
    publication=snapshot.get('publication_at_ns');receipt=snapshot.get('receipt_ns');bound=snapshot.get('clock_bound_s')
    if not _stamp(publication) or not _stamp(receipt) or not _finite(bound) or not 0<=bound<=.04:return None
    if receipt>at or publication>at+bound*1e9 or (at-publication)/1e9+bound>.25:return None
    normalized=normalize_radar_record({'kind':'sample','service':'liveTracks','valid':snapshot.get('valid'),
        'logMonoTime':snapshot.get('source_ns'),'laptop_receipt_ns':receipt,
        'data':{'points':snapshot.get('points'),'errors':snapshot.get('errors')}})
    return snapshot if normalized['valid'] else None

def _velocity_evidence(motion,obj):
    result={'lidar_independent_ego_velocity_mps':None,'lidar_independent_yaw_rps':None,
            'lidar_velocity_error_bound_mps':None,'lidar_yaw_error_bound_rps':None}
    try:
        dt=motion['dt_s'];increment=motion['ego_increment_se2'];angle=increment['yaw_rad']
        trans=np.asarray(increment['translation_m'],float);quality=motion['quality']
        if motion.get('valid') is not True or not _finite(dt) or not .035<=dt<=.25 or not _finite(angle) or trans.shape!=(2,) or not np.isfinite(trans).all():return result
        if abs(angle)<1e-7:velocity=trans/dt
        else:
            a=math.sin(angle)/angle;b=(1-math.cos(angle))/angle
            velocity=np.linalg.solve(np.array([[a,-b],[b,a]]),trans/dt)
        te=quality.get('translation_error_m');ye=quality.get('yaw_error_rad');residual=obj.get('residual_m')
        if not all(_finite(v) and v>=0 for v in (te,ye,residual)):return result
        if motion.get('ego_increment_se3'):
            velocity=np.asarray(motion['velocity_lidar_xyz_mps'],float)[:2]
        # The optional 2D radar velocity model cannot silently discard a 3D
        # pitch/roll lever-arm contribution. Carry a conservative height bound.
        tilt=np.linalg.norm(np.asarray(motion.get('angular_velocity_lidar_rps',[0.,0.,0.]),float)[:2])
        bound=(te+residual)/dt+5.*tilt
        result.update(lidar_independent_ego_velocity_mps=velocity.tolist(),lidar_independent_yaw_rps=angle/dt,
                      lidar_velocity_error_bound_mps=[bound,bound],lidar_yaw_error_bound_rps=ye/dt)
    except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):pass
    return result

def _motion_comparison(ctx,at,provenance,source_kind,seen,diagnostics):
    """Retain actual asynchronous native channels selected at a recorded query.

    Telemetry.context already enforces each channel's freshness, clock bound,
    sensor validity and receipt cutoff. Pairing speed at the gyro timestamp
    would discard valid staggered publications; the fitter models each native
    channel at its own source time and separately checks causal availability.
    """
    if (ctx.get('motion_ready') is not True or not _stamp(ctx.get('motion_source_ns')) or
        not _stamp(ctx.get('speed_source_ns')) or
        not all(_finite(ctx.get(key)) for key in ('speed_mps','yaw_rate_rps','clock_bound_s'))):return None
    identity=(provenance['block_id'],ctx.get('source_epoch'),ctx['speed_source_ns'],ctx['motion_source_ns'])
    if identity in seen:
        diagnostics['duplicate_comma_source_samples_skipped']+=1;return None
    seen.add(identity)
    return {**provenance,**{key:ctx[key] for key in ('speed_mps','yaw_rate_rps','clock_bound_s','motion_ready','motion_source_ns','speed_source_ns')},
        **{key:ctx[key] for key in ('speed_native_source_ns','motion_native_source_ns','speed_receipt_ns','motion_receipt_ns') if _stamp(ctx.get(key))},
        'source_epoch':ctx.get('source_epoch'),'at_ns':at,'query_at_ns':at,
        'provenance':'causal_'+source_kind,'reference_kind':'comparison_not_truth'}

def _association_seed(prior,epoch):
    """A rough installation seed changes matching only, never evidence bounds.

    Raw observations stay in their original frames. All fit/holdout gates must
    still pass, and a mismatch in mounting period is an error, not a fallback.
    """
    if prior is None:return np.eye(2),None
    if (not isinstance(prior,dict) or prior.get('mount_epoch')!=epoch or
        prior.get('reference_frame')!='lidar_to_radar_native_flu' or
        prior.get('source_kind')!='rough_user_measurement' or prior.get('verified') is not False):
        raise ValueError('Invalid or wrong-mount association search prior')
    xy=prior.get('translation_m');yaw=prior.get('yaw_rad')
    if (not isinstance(xy,list) or len(xy)!=2 or not all(_finite(x) for x in xy) or
        not -5<=xy[0]<=0 or abs(xy[1])>2 or not _finite(yaw) or abs(yaw)>.175):
        raise ValueError('Implausible association search prior')
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s],[s,c]]),np.asarray(xy,float)


def analyze_session(folder,cancel=None,progress=None,*,budget=None,native_comma_path=None,association_search_prior=None,reference_override=None):
    """Extract candidate evidence from one completed session; writes nothing.

    budget keys are DEFAULT_BUDGET. Windows are contiguous so scan registration
    never connects sparsely sampled frames. Unknown mount epochs remain usable
    diagnostically, but persistence/activation is never authorized here. Native
    rlogs require a separate clock/epoch adapter and are not silently imported.
    """
    if native_comma_path is not None:raise ValueError('Native/CSV sources require an explicit clock adapter; captured comma.jsonl or saved sensor observations are supported')
    limits=dict(DEFAULT_BUDGET)
    if budget is not None:
        if not isinstance(budget,dict) or set(budget)-set(limits):raise ValueError('Unknown analysis budget')
        limits.update(budget)
    for key,value in limits.items():
        if not _finite(value) or value<=0 or (key!='max_wall_s' and type(value) is not int):raise ValueError('Invalid analysis budget: '+key)
    if limits['window_frames']>limits['stride_frames'] or limits['max_frames']>20000 or limits['max_points']>100000:raise ValueError('Analysis budget exceeds supported bounds')
    folder=Path(folder).resolve();started=time.monotonic();deadline=started+limits['max_wall_s']
    check_work(cancel);sources=_fingerprints(folder,cancel)
    cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    epoch=_epoch(cfg.get('mount_epoch'));session=str(cfg.get('session_id') or folder.name)
    search_rotation,search_translation=_association_seed(association_search_prior,epoch)
    identity=hashlib.sha256(_canonical({'session':session,'mount_epoch':epoch,'sources':sources})).hexdigest()
    report={'schema_version':1,'algorithm':VERSION,'status':'complete','source_id':identity,'session_id':session,
        'mount_epoch':epoch,'mount_epoch_known':epoch is not None,'source_files':sources,'ground_observations':[],
        'radar_pairs':[],'motion_rows':[],'comma_rows':[],'coverage':{},'eligible_for_activation':False,
        'synthetic_fixture':bool(cfg.get('synthetic_fixture')),'evidence_kind':'sensor_derived_empirical_candidates',
        'limitations':['Road-plane normal includes road grade and suspension motion; it is not an independently surveyed mounting angle.',
            'Nominal radar origin and stored lidar time mapping remain priors, not verified extrinsics or sensor latency.',
            'Raw scan motion is independent of comma, but can fail on moving/degenerate scenes.',
            'Only raw frames with saved contexts are decoded by this adapter; missing/processing-skipped contexts are reported.',
            'Large-source fingerprints cover metadata and bounded file edges, not full-file cryptographic identity.']}
    from selfcal_motion3d import IndependentMotion3D as IndependentMotion
    from selfcal_reference import choose_reference
    from selfcal_association import TemporalPairs,extent_distances
    associations=TemporalPairs()
    motion=IndependentMotion(max_points=1600);tracker=_Objects();index=None;comma=None;diagnostics=Counter();raw_diag=Counter();last_selected=None;window=None;seen_comma=set();fixed_level=None;actual_windows={}
    report['association_search_prior']=association_search_prior
    report['association_prior_is_calibration']=False
    origin=np.array([float(cfg.get('mount',{}).get('bumper_x_m',2.)),0.])
    if not np.isfinite(origin).all() or not 0<=origin[0]<=5:origin=np.array([2.,0.])
    try:
        index=_Contexts(folder/'frame-contexts.jsonl',limits['max_context_rows'],cancel,deadline)
        selected,selection=index.select(limits);report['selection']=selection
        reference_rows=[];reference_diag=Counter()
        if progress:progress({'stage':'road_reference','message':'Selecting a fixed reference from qualified fitting blocks; held-out blocks are excluded'})
        for points,stamps,stored,quality in selected_frames(folder,index,selected,cfg,cancel,deadline,reference_diag):
            check_work(cancel)
            if quality.get('timestamp_valid') is not True or quality.get('completeness',0)<.98 or not _stamp(stored.get('at_ns')):continue
            ground=sensor_ground(points)
            if ground is not None:
                effective=int(stored['at_ns']+int(np.median(stamps))-int(np.max(stamps))) if len(stamps) else stored['at_ns']
                reference_rows.append({**ground,'at_ns':effective})
        fixed_level,reference_quality=choose_reference(reference_rows)
        if reference_override is not None:
            from selfcal_alignment import _reference_rotation
            if (not isinstance(reference_override,dict) or reference_override.get('mount_epoch')!=epoch or
                not reference_override.get('source_session_id') or reference_override['source_session_id']==session):
                raise ValueError('Frozen reference requires another explicit session in the same mounting period')
            fixed_level,error=_reference_rotation([],{'fixed_level_rotation':reference_override.get('rotation')})
            if error or fixed_level is None:raise ValueError('Invalid frozen reference rotation')
            reference_quality={'eligible':False,'method':'frozen_cross_session_diagnostic_only',
                'source_session_id':reference_override['source_session_id'],'reason':'Frozen reference evaluation cannot directly qualify a new profile.',
                'rotation':fixed_level.tolist(),'heldout_used_for_selection':False}

        report['reference_quality']=reference_quality;report['reference_prepass']=dict(reference_diag)
        if fixed_level is None:
            fixed_level=np.eye(3)
            report['limitations'].append('No qualified road reference; raw-frame motion remains diagnostic and cannot qualify a profile.')
        report['fixed_level_rotation']=fixed_level.tolist()
        report['level_reference']='fit_only_block_robust_reference' if reference_quality['eligible'] else 'identity_diagnostic_only_unqualified_reference'
        comma=_Comma(folder/'comma.jsonl',limits['max_comma_rows'],cancel,deadline)
        for points,stamps,stored,quality in selected_frames(folder,index,selected,cfg,cancel,deadline,raw_diag):
            check_work(cancel)
            if time.monotonic()>deadline:report['status']='budget_limited';break
            diagnostics['raw_frames_seen']+=1
            if diagnostics['analyzed_frames']>=limits['max_frames']:report['status']='budget_limited';break
            at=stored.get('at_ns')
            if not _stamp(at) or quality.get('timestamp_valid') is not True or quality.get('completeness',0)<.98:
                diagnostics['invalid_frames']+=1;motion=IndependentMotion(max_points=1600);tracker.reset();continue
            current_window=quality.get('selection_window',0)
            if window!=current_window or (last_selected is not None and not .04<=(at-last_selected)/1e9<=.25):
                motion=IndependentMotion(max_points=1600);tracker.reset();associations.reset();diagnostics['independent_windows']+=1
            window=current_window;last_selected=at;diagnostics['analyzed_frames']+=1
            actual=actual_windows.setdefault(str(current_window),{'first_at_ns':at,'last_at_ns':at,'frames':0,'first_context_line':quality.get('context_line'),'last_context_line':quality.get('context_line')})
            actual.update(last_at_ns=at,last_context_line=quality.get('context_line'));actual['frames']+=1
            if progress and diagnostics['analyzed_frames']%10==0:progress({'stage':'sensor_calibration_extraction',
                'message':'Analyzing continuous raw-lidar windows','frames':diagnostics['analyzed_frames'],
                'total':min(len(selected),limits['max_frames']),**dict(diagnostics),'radar_pairs':len(report['radar_pairs'])})
            ctx,radar=comma.at(at)
            block=session+':window:'+str(current_window)+':epoch:'+str(comma.epoch)
            provenance={'session_id':session,'source_id':identity,'mount_epoch':epoch,'block_id':block,
                'registration_window_id':block,'alignment_block_id':session+':'+str(at//2_000_000_000),
                'context_line':quality.get('context_line'),'first_device_ns':quality.get('first_device_ns')}
            saved=index.get(quality.get('context_line')).get('sensor_observations',{})
            saved_radar=_snapshot(saved.get('radar') if isinstance(saved,dict) else None,at)
            radar=saved_radar or _snapshot(radar,at)
            diagnostics['radar_snapshot_frames' if saved_radar else 'raw_comma_radar_frames' if radar else 'radar_unavailable_frames']+=1
            for key in ('speed_mps','yaw_rate_rps','clock_bound_s'):
                if not _finite(ctx.get(key)):ctx[key]=None
            comparison=_motion_comparison(ctx,at,provenance,comma.source_scope['kind'],seen_comma,diagnostics)
            if comparison is not None:report['comma_rows'].append(comparison)
            effective=int(at+int(np.median(stamps))-int(np.max(stamps))) if len(stamps) else at
            ground=sensor_ground(points)
            provenance.update(reference_level_rotation=fixed_level.tolist(),point_frame='fixed_lidar_ground_level',seed_source='lidar_only')
            measured=motion.update(np.asarray(points)[:,:3]@fixed_level.T,effective,provenance=provenance)
            report['motion_rows'].append({**provenance,**measured})
            if measured.get('valid') is not True:diagnostics['motion_invalid: '+str(measured.get('reason','unknown'))]+=1
            if ground is None:diagnostics['ground_unavailable_frames']+=1;tracker.reset();continue
            row={**provenance,**{k:v for k,v in ground.items() if not k.startswith('_')},'at_ns':effective,
                'speed_mps':ctx.get('speed_mps'),'yaw_rate_rps':ctx.get('yaw_rate_rps'),'independent_of_comma':True}
            report['ground_observations'].append(row)
            objects=tracker.update(_objects(points,stamps,ground,at,limits['max_points'],fixed_level),measured,effective)
            if not radar or ctx.get('speed_mps') is None or not measured.get('valid'):continue
            stable=[obj for obj in objects if obj['age_s']>=.3 and obj['hits']>=4 and obj['ground_velocity'] is not None and np.linalg.norm(obj['ground_velocity'])<=1. and obj['residual_m']<=.2]
            radar_points=[p for p in radar['points'] if p.get('measured') is True and abs(p['vRel']+ctx['speed_mps'])<=2.]
            diagnostics['static_lidar_object_observations']+=len(stable);diagnostics['static_radar_object_observations']+=len(radar_points)
            if not stable or not radar_points:continue
            positions=np.array([[p['dRel'],p['yRel']] for p in radar_points])+origin
            search_positions=np.array([obj['position'] for obj in stable])
            if search_translation is not None:search_positions=search_positions@search_rotation.T+search_translation+origin
            distances=extent_distances(stable,positions,radar_points,radar['publication_at_ns'],
                search_rotation if search_translation is not None else None,
                search_translation+origin if search_translation is not None else None)
            choices=distances<2.;row_counts=choices.sum(1);column_counts=choices.sum(0)
            for i,obj in enumerate(stable):
                candidates=np.flatnonzero(choices[i])
                if len(candidates)!=1 or column_counts[candidates[0]]!=1:
                    diagnostics['ambiguous_or_unmatched_lidar_observations']+=1;continue
                j=int(candidates[0]);point=radar_points[j]
                rivals=np.r_[np.delete(distances[i],j),np.delete(distances[:,j],i)]
                margin=float(np.min(rivals)-distances[i,j]) if len(rivals) else 2.
                delta=(radar['publication_at_ns']-obj['at_ns'])/1e9
                clock_bound=radar['clock_bound_s']
                # These are post-drive fitting candidates, not live fusion.
                # Preserve scan spread and unknown sensor delay separately;
                # do not conflate them with the observed NTP clock bound.
                if margin<.5 or abs(delta)>.12 or clock_bound>.04 or obj['point_span_s']>.12:
                    diagnostics['association_margin_or_timing_rejections']+=1;continue
                temporal=associations.confirm(obj['id'],point['trackId'],obj['at_ns'],radar['source_ns'])
                if not temporal['confirmed']:
                    diagnostics['association_temporal_confirmation_pending']+=1;continue
                if len(report['radar_pairs'])>=limits['max_radar_pairs']:diagnostics['radar_pair_budget_exhausted']+=1;break
                report['radar_pairs'].append({**provenance,'at_ns':obj['at_ns'],'lidar_device_ns':obj['device_ns'],'association_evidence':temporal,'association_method':'time_compensated_extent_and_temporal_unique_v2',
                    **_velocity_evidence(measured,obj),
                    'lidar_track_id':obj['id'],'radar_track_id':point['trackId'],'lidar_xy':obj['position'].tolist(),
                    'lidar_lo':obj['lo'].tolist(),'lidar_hi':obj['hi'].tolist(),'radar_xy':positions[j].tolist(),
                    'radar_xy_native':[point['dRel'],point['yRel']],'radar_origin_prior_m':origin.tolist(),
                    'radar_origin_prior_bound_m':[1.,.35],'radar_origin_prior_source':'configured_bumper_distance_provisional',
                    'radar_vrel':point['vRel'],'ego_speed':ctx['speed_mps'],'ego_yaw_rate_rps':ctx.get('yaw_rate_rps'),'time_delta_s':delta,
                    'time_delta_convention':'radar_publication_minus_lidar',
                    'timing_error_bound_s':clock_bound,'includes_lidar_alignment':False,'includes_radar_latency':False,
                    'lidar_point_time_half_span_s':obj['point_span_s']/2,
                    'lidar_ground_velocity_mps':obj['ground_velocity'].tolist(),'unique_match':True,'association_margin_m':margin,
                    'ground_quality':{'fit_rmse_m':ground['fit_rmse_m'],'span_x_m':ground['span_x_m'],'span_y_m':ground['span_y_m']},
                    'static_quality':{'residual_m':obj['residual_m'],'age_s':obj['age_s'],'hits':obj['hits'],'measured':True},
                    'radar_publication_at_ns':radar['publication_at_ns'],'radar_source_ns':radar['source_ns'],
                    'temporal_usage':'postdrive_calibration_not_live_replay','coordinate_frame':'sensor_ground_leveled_to_nominal_camera_flu',
                    'candidate_only':True})
    except TimeoutError as error:report.update(status='budget_limited',budget_reason=str(error))
    finally:
        if index:index.close()
        if comma:comma.close()
    check_work(cancel)
    after=_fingerprints(folder,cancel)
    if sources!=after:raise ValueError('Source changed during read-only calibration analysis')
    report['coverage']={**dict(diagnostics),'recording_iterator':raw_diag,'comma':dict(comma.diag) if comma else {},
        'comma_source':copy.deepcopy(comma.source_scope) if comma else {'kind':'unavailable'},
        'ground_observations':len(report['ground_observations']),'radar_pairs':len(report['radar_pairs']),
        'valid_motion_rows':sum(row.get('valid') is True for row in report['motion_rows']),
        'comma_rows':len(report['comma_rows']),'duration_s':time.monotonic()-started,'budget':limits,
        'analyzed_windows':actual_windows,
        'all_session_frames_analyzed':bool(index and report['status']=='complete' and diagnostics['analyzed_frames']==len(index.metadata) and not diagnostics['invalid_frames'])}
    ranges=[float(np.linalg.norm(row['lidar_xy'])) for row in report['radar_pairs']]
    report['coverage']['paired_lidar_range_m']={'min':min(ranges),'max':max(ranges),'span':max(ranges)-min(ranges)} if ranges else None
    if not epoch:report['limitations'].append('Mount epoch is unknown; these rows cannot qualify the current installation.')
    return report

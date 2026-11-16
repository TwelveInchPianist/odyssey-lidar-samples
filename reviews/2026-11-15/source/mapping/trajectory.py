"""Replay compact route evidence into local odometry and geographic trace.

Local wheel/yaw trajectories and GPS traces are distinct products. Neither is
labeled ground truth or silently converted into a globally corrected map pose.
"""
from collections import Counter
import json,math,os
from pathlib import Path
from telemetry import Telemetry
from offline_jobs import check_work
from .journal import journal_parts,read_journal
from .location import decode_fix,FixContinuity,best_fix

VERSION='route-trajectory-v1'

def route_records(folder,diagnostics,cancel=None):
    if journal_parts(folder):
        for ref,event in read_journal(folder,diagnostics,cancel):
            kind=event['kind'];data=event['data'];receipt=event.get('receipt_ns')
            if kind in ('journal_open','stream_restart','source_clock_discontinuity'):
                yield ref,{'kind':'hello','laptop_receipt_ns':receipt,'commit':data.get('commit'),'route_epoch':event['epoch']}
            elif kind in ('sample','clock_exchange','comma_recording_reference'):
                item={k:v for k,v in data.items() if k!='location'}
                item['kind']='sample' if kind=='sample' else 'pong' if kind=='clock_exchange' else 'route_reference'
                yield ref,item
        return
    # Legacy recordings remain importable, with honest coverage limits.
    diagnostics['legacy_bulk_comma_source']=1
    path=Path(folder)/'comma.jsonl'
    if not path.exists():diagnostics['no_telemetry_source']=1;return
    with path.open(encoding='utf-8') as f:
        for number,line in enumerate(f,1):
            check_work(cancel)
            try:
                record=json.loads(line)
                if not isinstance(record,dict):raise ValueError('not a record')
            except ValueError:
                diagnostics['invalid_legacy_records']+=1
                # Make the missing evidence visible to the motion consumer;
                # skipping it silently can join samples around a corrupt row.
                yield {'part':path.name,'line':number},{'kind':'offline_evidence_gap'}
                continue
            yield {'part':path.name,'line':number},record

def samples(folder,diagnostics,cancel=None):
    tele=Telemetry();fix_gate=FixContinuity();locations={};last=None;segment=-1;epoch=0;x=y=yaw=distance=0.;last_receipt=None
    for ref,item in route_records(folder,diagnostics,cancel):
        check_work(cancel);receipt=item.get('laptop_receipt_ns');kind=item.get('kind')
        if kind=='offline_evidence_gap':
            last=None;epoch+=1;tele.messages.clear();tele.history.clear();tele.selected_times.clear()
            fix_gate.reset();locations.clear();diagnostics['malformed_record_gaps']+=1;continue
        if type(receipt) is not int:
            diagnostics['missing_receipt_time']+=1;last=None
            service=item.get('service')
            if isinstance(service,str):
                tele.messages.pop(service,None);tele.history.pop(service,None);tele.selected_times.pop(service,None)
            else:
                tele.messages.clear();tele.history.clear();tele.selected_times.clear()
            continue
        if kind!='sample' and item.get('service') is not None:
            diagnostics['invalid_motion_record_kind']+=1;last=None
            service=item.get('service')
            if isinstance(service,str):
                tele.messages.pop(service,None);tele.history.pop(service,None);tele.selected_times.pop(service,None)
            else:
                tele.messages.clear();tele.history.clear();tele.selected_times.clear()
            continue
        if kind=='hello' or (last_receipt is not None and receipt<last_receipt):
            tele=Telemetry();fix_gate.reset();locations.clear();last=None;epoch+=1;diagnostics['stream_segments']+=1
        last_receipt=receipt
        if kind=='pong':
            try:
                if any(type(item.get(k)) is not int for k in ('laptop_send_ns','remote_receive_ns','remote_send_ns')):raise ValueError('invalid clock timestamps')
                tele.clock.add(item['laptop_send_ns'],item['remote_receive_ns'],item['remote_send_ns'],receipt)
            except (KeyError,TypeError,ValueError,OverflowError):diagnostics['invalid_clock_exchange']+=1
            continue
        if kind!='sample':
            continue
        if kind=='sample' and (not isinstance(item.get('service'),str) or not isinstance(item.get('data'),dict)):
            diagnostics['malformed_telemetry_payloads']+=1;last=None
            if not isinstance(item.get('service'),str):continue
            item={**item,'data':{},'valid':False}
        try:
            tele.accept(item,receipt);sync=tele.clock.estimate(receipt)
            fix=decode_fix(item,sync) if kind=='sample' else None
        except (AttributeError,KeyError,TypeError,ValueError,OverflowError):
            diagnostics['malformed_telemetry_records']+=1;last=None;continue
        if fix is not None:
            fix=fix_gate.accept(fix);fix['source_ref']=ref;fix['stream_epoch']=epoch;locations[fix['provider']]=fix
            diagnostics['usable_location_observations' if fix['usable_for_fusion'] else 'coarse_location_observations' if fix.get('usable_for_trace') else 'unusable_location_observations']+=1
            yield {'kind':'location','fix':fix}
        if item.get('service')!='carState':continue
        try:ctx=tele.context(receipt)
        except (AttributeError,KeyError,TypeError,ValueError,OverflowError):
            diagnostics['malformed_motion_contexts']+=1;last=None;continue
        source=item.get('logMonoTime')
        valid=bool(ctx.get('motion_ready') and sync is not None and sync[1]<=40_000_000 and type(source) is int and
                   item.get('valid') and ctx.get('motion_source_ns') is not None)
        if not valid:
            last=None;diagnostics['invalid_motion_samples']+=1;continue
        at=int(source+sync[0]);dt=(at-last['at_ns'])/1e9 if last else None
        if last and dt is not None and dt<=0:
            last=None;diagnostics['non_increasing_motion_time']+=1;continue
        speed=ctx['speed_mps'];rate=ctx['yaw_rate_rps']
        if last is None or dt>.25:
            segment+=1;x=y=yaw=distance=0.;diagnostics['odometry_segments']+=1
            if last:diagnostics['motion_gaps']+=1
        else:
            v=(last['speed_mps']+speed)/2;omega=(last['yaw_rate_rps']+rate)/2
            angle=yaw+omega*dt/2;x+=v*dt*math.cos(angle);y+=v*dt*math.sin(angle);yaw+=omega*dt;distance+=v*dt
        row={'kind':'motion','schema_version':1,'algorithm':VERSION,'at_ns':at,'stream_epoch':epoch,'segment_id':segment,
             'x_m':x,'y_m':y,'yaw_rad':yaw,'distance_m':distance,'speed_mps':speed,'yaw_rate_rps':rate,
             'location':best_fix(locations.values(),at,allow_hints=True),'source_ref':ref,
             'uncertainty':{'kind':'unbounded_drift_local_dead_reckoning','clock_bound_s':sync[1]/1e9}}
        last=row;diagnostics['odometry_samples']+=1;yield row

def build_trajectory(folder,cancel=None):
    folder=Path(folder);diagnostics=Counter();geo={};seen_fixes=set();segments={};fix_count=0;available_fixes={};fix_epoch=None
    local_path=folder/'route-trajectory.jsonl';location_path=folder/'route-location-fixes.jsonl'
    local_tmp=local_path.with_suffix('.jsonl.tmp');location_tmp=location_path.with_suffix('.jsonl.tmp')
    with local_tmp.open('w',encoding='utf-8') as local,location_tmp.open('w',encoding='utf-8') as fixes:
        for row in samples(folder,diagnostics,cancel):
            if row['kind']=='location':
                fix=row['fix'];fixes.write(json.dumps(fix,allow_nan=False)+'\n');fix_count+=1
                if fix_epoch!=fix['stream_epoch']:available_fixes.clear();fix_epoch=fix['stream_epoch']
                available_fixes[fix['provider']]=fix
                selected=best_fix(available_fixes.values(),fix['at_ns'],allow_trace=True) if fix.get('at_ns') is not None else None
                if selected:
                    identity=(selected['stream_epoch'],selected['at_ns'],selected['provider'])
                    if identity not in seen_fixes:
                        seen_fixes.add(identity);p=selected['position'];gkey=(selected['stream_epoch'],selected['geographic_segment'])
                        geo.setdefault(gkey,[]).append((selected['at_ns'],[p['longitude_deg'],p['latitude_deg']],selected['horizontal_accuracy_m']))
                continue
            local.write(json.dumps(row,allow_nan=False)+'\n');key=(row['stream_epoch'],row['segment_id'])
            segments[key]=max(segments.get(key,0),row['distance_m'])
        local.flush();os.fsync(local.fileno());fixes.flush();os.fsync(fixes.fileno())
    check_work(cancel)
    features=[]
    for key,rows in sorted(geo.items()):
        rows.sort(key=lambda x:x[0]);coordinates=[r[1] for r in rows]
        features.append({'type':'Feature','properties':{'stream_epoch':key[0],'geographic_segment':key[1],'source':'selected_receiver_positions',
                         'max_reported_horizontal_accuracy_m':max((r[2] for r in rows if r[2] is not None),default=None),'accuracy_missing_for_some_points':any(r[2] is None for r in rows),'not_ground_truth':True},
                         'geometry':{'type':'LineString','coordinates':coordinates} if len(coordinates)>1 else {'type':'Point','coordinates':coordinates[0]}})
    geo_path=folder/'route-gps.geojson';geo_tmp=geo_path.with_suffix('.geojson.tmp')
    geo_tmp.write_text(json.dumps({'type':'FeatureCollection','features':features},allow_nan=False),encoding='utf-8')
    for source,target in ((local_tmp,local_path),(location_tmp,location_path),(geo_tmp,geo_path)):os.replace(source,target)
    result={'schema_version':1,'algorithm':VERSION,'status':'complete','diagnostics':dict(diagnostics),'local_segments':len(segments),
            'valid_local_distance_m':sum(segments.values()),'geographic_segments':len(features),'location_observations':fix_count,
            'gps_state':('available' if diagnostics['usable_location_observations'] else 'coarse_trace') if features else 'unavailable_or_unusable','source':'compact_route_journal' if journal_parts(folder) else 'legacy_comma_log',
            'limitations':'Local odometry drifts. GPS trace uses reported receiver accuracy; no cross-drive/global pose correction is claimed here. Invalid intervals are not connected.'}
    path=folder/'route-review.json';temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(temporary,path)
    return result

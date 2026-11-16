"""Exact, disposable raw-replay checkpoints; no live policy changes.

run_replay(folder, work_root, cancel=None, ...) reads immutable recordings and
writes ONLY into a separate owned work_root. Context indexing and replay chunks
commit to SQLite together with explicit Engine/Telemetry/FrameAssembler/
ScanMotion state and source cursors. A canceled chunk rolls back; resume seeks
its last committed packet/Comma offsets. No tracker reset or re-warm heuristic
substitutes for saved continuous state. Exported derived JSONL can be regenerated
from committed rows after a torn/truncated tail. Old outputs without this DB and
fingerprint are never trusted as resumable checkpoints.

Codec: bounded JSON + base64 typed arrays/packets, never pickle/eval/dynamic class
deserialization. Unknown object fields reject the checkpoint. Ground.fit creates
its own seeded RNG(17) per call; no advancing/global RNG state is used by the
reviewed engine. Code digests reject a future implementation that changes this.

Fingerprints include algorithm/code/runtime versions, exact small metadata,
mount/policy config, and bulk file stat + bounded edge digests. Raw bytes are
hashed as consumed. Normal source changes reject continuation; a same-stat edit
in an unsampled middle region needs the separate full integrity audit. Resuming
does not rehash/replay the entire historical 11GB prefix. This is explicit, not
a claim of cryptographic immutability against a file editor preserving metadata.

Scope matches runtime.replay_and_review: only raw scans with original recorded
contexts are recomputed. Unmatched raw scans remain counted, not silently treated
as warning-eligible. Timing proposals are returned separately; caller owns report
publication/summarize. Runtime timing/freshness limits are never relaxed.
"""
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, fields
import base64
import hashlib
import importlib
import json
import marshal
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sqlite3
import struct
import sys
import runtime_clock as time
import types
import uuid

import numpy as np
import scipy
from fcw_engine import Engine, Mount, Ground, Tracker, Track, transform
from frames import FrameAssembler, decode_with_time
from m1_protocol import metadata, valid_msop
from telemetry import Telemetry
from registration import ScanMotion
from offline_jobs import WorkPaused, check_work

VERSION = 'exact-raw-replay-v5-point-ownership'
CODEC_VERSION = 7  # Older disposable checkpoints lack point ownership and must be recomputed.
MINIMUM_FREE_BYTES = 10_000_000_000 + 16 * 1024 * 1024
MAX_STATE_BYTES = 64 * 1024 * 1024
MAX_JSON_LINE = 8 * 1024 * 1024
MAX_ARRAY_BYTES = 16 * 1024 * 1024
MAX_CODEC_NODES = 1_000_000
APPLICATION_ID = 0x4f464357
# Forced policy is reconstructed only from fingerprint-bound session config,
# never deserialized as an arbitrary object from checkpoint data.
ENGINE_FIELDS = {'mount','policy','ground','tracker','last_ns','held','last_alert','last_any_alert','last_alert_urgent','last_warning','forced_experiment',
                 '_far_accumulator','_lead_history','_radar_streaks','_radar_publication_times','_radar_last_update_ns','_longitudinal_evidence','_settled_road_reference','_warning_surface'}
GROUND_FIELDS = {'plane','stable','last_reason','miss_state'}
ASSEMBLER_FIELDS = {'packets','arrivals','expected','started','bad_packets','duplicate_packets','partial_frames'}
TELEMETRY_FIELDS = {'clock','messages','history','commit','rejected','selected_times','selected_records'}
MOTION_FIELDS = {'previous','previous_ns','last_transform'}


class CheckpointRejected(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _fields(obj, expected):
    if set(vars(obj)) != set(expected):
        raise CheckpointRejected('Unsupported state fields for ' + type(obj).__name__)


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.str not in ('<f4','<f8','<i4','<i8','<u4','<u8','|u1','|b1') or value.nbytes > MAX_ARRAY_BYTES:
            raise CheckpointRejected('Unsupported/oversized numeric state array')
        return {'t':'array','dtype':value.dtype.str,'shape':list(value.shape),'v':base64.b64encode(value.tobytes(order='C')).decode('ascii')}
    if isinstance(value, np.generic):
        return _pack(value.item())
    if isinstance(value, bytes):
        return {'t':'bytes','v':base64.b64encode(value).decode('ascii')}
    if isinstance(value, dict):
        return {'t':'dict','v':[[_pack(k),_pack(v)] for k,v in value.items()]}
    if isinstance(value, (list, tuple, deque)):
        return {'t':'tuple' if isinstance(value, tuple) else 'list','v':[_pack(v) for v in value]}
    if isinstance(value, float) and not math.isfinite(value):
        return {'t':'nonfinite_float','v':struct.pack('<d', value).hex()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise CheckpointRejected('Unsupported state type: ' + type(value).__name__)


def _unpack(value, budget=None, depth=0):
    budget = [MAX_CODEC_NODES] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 70:
        raise CheckpointRejected('State nesting/node budget exceeded')
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise CheckpointRejected('Nonfinite scalar must use explicit codec')
        return value
    if not isinstance(value, dict) or not isinstance(value.get('t'), str):
        raise CheckpointRejected('Malformed typed state')
    kind = value['t']
    if kind in ('dict','list','tuple'):
        if set(value) != {'t','v'} or not isinstance(value['v'], list):
            raise CheckpointRejected('Malformed container state')
        if kind == 'dict':
            out = {}
            for pair in value['v']:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise CheckpointRejected('Malformed dictionary pair')
                key = _unpack(pair[0], budget, depth+1)
                if type(key) not in (str,int) or key in out:
                    raise CheckpointRejected('Unsupported/duplicate dictionary key')
                out[key] = _unpack(pair[1], budget, depth+1)
            return out
        items = [_unpack(v, budget, depth+1) for v in value['v']]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'nonfinite_float':
        if set(value) != {'t','v'} or not isinstance(value['v'], str) or len(value['v']) != 16:
            raise CheckpointRejected('Malformed float state')
        return struct.unpack('<d', bytes.fromhex(value['v']))[0]
    if kind == 'bytes':
        if set(value) != {'t','v'} or not isinstance(value['v'], str) or len(value['v']) > 4 * MAX_ARRAY_BYTES // 3 + 4:
            raise CheckpointRejected('Oversized byte state')
        return base64.b64decode(value['v'], validate=True)
    if kind == 'array':
        if set(value) != {'t','dtype','shape','v'} or value['dtype'] not in ('<f4','<f8','<i4','<i8','<u4','<u8','|u1','|b1'):
            raise CheckpointRejected('Unsupported array dtype/schema')
        shape = value['shape']
        if not isinstance(shape, list) or len(shape) > 3 or any(type(v) is not int or not 0 <= v <= 1_000_000 for v in shape):
            raise CheckpointRejected('Invalid state array shape')
        size = math.prod(shape) * np.dtype(value['dtype']).itemsize
        if size > MAX_ARRAY_BYTES or not isinstance(value['v'], str) or len(value['v']) > 4 * MAX_ARRAY_BYTES // 3 + 4:
            raise CheckpointRejected('Oversized array state')
        raw = base64.b64decode(value['v'], validate=True)
        if len(raw) != size:
            raise CheckpointRejected('State array length mismatch')
        return np.frombuffer(raw, dtype=value['dtype']).reshape(shape).copy()
    raise CheckpointRejected('Unknown state tag')


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _far_state(acc):
    if acc is None:return None
    _fields(acc, {'frames','hist','frame','owner_ids','_pending','last_owner_stats'})
    result = {'frames':acc.frames,'hist':acc.hist,'frame':acc.frame,
              'owner_ids':acc.owner_ids,'pending':acc._pending,'last_owner_stats':acc.last_owner_stats}
    _restore_far(result)  # Apply identical bounds when saving and restoring.
    return result


def _restore_far(value):
    if value is None:return None
    from far_objects import FarAccumulator
    if (not isinstance(value,dict) or set(value)!={'frames','hist','frame','owner_ids','pending','last_owner_stats'} or type(value['frames']) is not int
            or not 1<=value['frames']<=16 or type(value['frame']) is not int or value['frame']<0
            or not isinstance(value['hist'],list) or len(value['hist'])>value['frames']):
        raise CheckpointRejected('Invalid far accumulator schema/bounds')
    acc=FarAccumulator(value['frames']);acc.frame=value['frame'];last=-1
    for pair in value['hist']:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise CheckpointRejected('Invalid far history row')
        points,frame=pair
        if (not isinstance(points,np.ndarray) or points.ndim!=2 or points.shape[1]!=3 or len(points)>200000
                or not np.isfinite(points).all() or type(frame) is not int or not last<frame<=acc.frame):
            raise CheckpointRejected('Invalid far history geometry/frame order')
        last=frame;acc.hist.append((points.copy(),frame))
    owners=value['owner_ids'];lengths={frame:len(points) for points,frame in acc.hist}
    if not isinstance(owners,dict) or set(owners)!=set(lengths):
        raise CheckpointRejected('Invalid far ownership frame identities')
    for frame,ids in owners.items():
        if (type(frame) is not int or not isinstance(ids,np.ndarray) or ids.ndim!=1 or
                ids.dtype.kind!='i' or len(ids)!=lengths[frame] or np.any(ids < -1)):
            raise CheckpointRejected('Invalid far ownership array')
        acc.owner_ids[frame]=ids.copy()
    pending=value['pending']
    if pending is not None:
        if (not isinstance(pending,dict) or set(pending)!={'frame','current_indices','layout','total'} or
                type(pending['frame']) is not int or pending['frame']!=acc.frame or
                type(pending['total']) is not int or not 0<=pending['total']<=200000*16 or
                not isinstance(pending['layout'],list) or len(pending['layout'])>=acc.frames):
            raise CheckpointRejected('Invalid far pending ownership schema')
        current=pending['current_indices'];layout=[];seen=set();end=None
        for row in pending['layout']:
            if (not isinstance(row,(list,tuple)) or len(row)!=3 or any(type(v)is not int for v in row)):
                raise CheckpointRejected('Invalid far pending layout')
            frame,start,stop=row
            if (frame not in lengths or frame==acc.frame or frame in seen or
                    not 0<=start<=stop<=pending['total'] or stop-start!=lengths[frame] or
                    (end is not None and start!=end)):
                raise CheckpointRejected('Invalid far pending layout bounds')
            seen.add(frame);layout.append((frame,start,stop));end=stop
        current_total=layout[0][1] if layout else pending['total']
        if (seen!=set(lengths)-{acc.frame} or (layout and end!=pending['total']) or
                acc.frame not in lengths or not isinstance(current,np.ndarray) or current.ndim!=1 or
                current.dtype.kind!='i' or len(current)!=lengths[acc.frame] or
                np.any(current<0) or np.any(current>=current_total) or np.any(np.diff(current)<=0)):
            raise CheckpointRejected('Invalid far current membership indices')
        acc._pending={'frame':acc.frame,'current_indices':current.copy(),'layout':layout,'total':pending['total']}
    stats=value['last_owner_stats']
    if not isinstance(stats,dict) or len(stats)>10000:raise CheckpointRejected('Invalid far ownership diagnostics')
    acc.last_owner_stats=_unpack(_pack(stats))
    return acc


def _lead_state(history):
    if not isinstance(history,dict) or len(history)>20000:raise CheckpointRejected('Invalid lead history map')
    rows=[]
    for key,samples in history.items():
        if type(key) is int:kind,tid='lidar',key
        elif isinstance(key,tuple) and len(key)==2 and key[0]=='r' and type(key[1]) is int:kind,tid='radar',key[1]
        else:raise CheckpointRejected('Invalid lead history identity')
        rows.append({'kind':kind,'id':tid,'samples':list(samples)})
    _restore_lead(rows)
    return rows


def _restore_lead(rows):
    if not isinstance(rows,list) or len(rows)>20000:raise CheckpointRejected('Invalid lead history rows')
    result={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!={'kind','id','samples'} or row['kind'] not in ('radar','lidar')
                or type(row['id']) is not int or not isinstance(row['samples'],list)):
            raise CheckpointRejected('Invalid lead history row')
        key=('r',row['id']) if row['kind']=='radar' else row['id'];limit=60 if row['kind']=='radar' else 40
        if key in result or len(row['samples'])>limit:raise CheckpointRejected('Duplicate/oversized lead history')
        values=[];last=-math.inf
        for pair in row['samples']:
            if (not isinstance(pair,(tuple,list)) or len(pair)!=2 or not all(_finite_number(x) for x in pair)
                    or pair[0]<last):raise CheckpointRejected('Invalid lead sample/time order')
            last=pair[0];values.append(tuple(pair))
        result[key]=deque(values,maxlen=limit)
    return result


def _validate_near_edge(value):
    if value is None:return
    if not isinstance(value,dict) or set(value) not in ({'near','agree_s','v_near_y'},{'near','agree_s','v_near_y','v_near_x'}):raise CheckpointRejected('Invalid near-edge schema')
    if not _finite_number(value['agree_s']) or value['agree_s']<0:raise CheckpointRejected('Invalid near-edge agreement duration')
    if value.get('v_near_x') is not None and not _finite_number(value['v_near_x']):raise CheckpointRejected('Invalid longitudinal near-edge velocity')
    if value['v_near_y'] is not None and not _finite_number(value['v_near_y']):raise CheckpointRejected('Invalid near-edge velocity')
    if value['near'] is not None:
        near=value['near']
        if (not isinstance(near,(tuple,list)) or len(near)!=2 or not isinstance(near[0],np.ndarray)
                or near[0].shape!=(2,) or not np.isfinite(near[0]).all() or type(near[1]) is not int or not 0<=near[1]<4):
            raise CheckpointRejected('Invalid near-edge corner')


def _validate_ground(value):
    if not isinstance(value,dict) or set(value)!=GROUND_FIELDS:raise CheckpointRejected('Invalid ground schema')
    miss=value['miss_state']
    if (not isinstance(miss,dict) or set(miss)!={'miss','tol'}
            or any(type(miss[k]) is not int or miss[k]<0 for k in ('miss','tol'))):
        raise CheckpointRejected('Invalid ground miss state')


def _validate_radar(streaks,times,last):
    for mapping in (streaks,times):
        if (not isinstance(mapping,dict) or len(mapping)>20000
                or any(type(k) is not int or type(v) is not int or v<0 for k,v in mapping.items())):
            raise CheckpointRejected('Invalid radar temporal map')
    if last is not None and (type(last) is not int or last<0):raise CheckpointRejected('Invalid radar update time')



def _validate_longitudinal(history):
    if not isinstance(history,dict) or len(history)>10000:raise CheckpointRejected('Invalid longitudinal history')
    required={'first_ns','at_ns','count','radar_id','publication','x','vrel'}
    for key,row in history.items():
        if type(key) is not int or not isinstance(row,dict) or set(row)!=required:raise CheckpointRejected('Invalid longitudinal row')
        if any(type(row[k]) is not int or row[k]<0 for k in ('first_ns','at_ns','count','radar_id','publication')):raise CheckpointRejected('Invalid longitudinal identity/time')
        if row['at_ns']<row['first_ns'] or not all(_finite_number(row[k]) for k in ('x','vrel')):raise CheckpointRejected('Invalid longitudinal measurements')

def _validate_recovery(reference,surface):
    def vector(v,n):
        return isinstance(v,list) and len(v)==n and all(_finite_number(x) for x in v)
    if reference is not None:
        if (not isinstance(reference,dict) or set(reference)!={'abc','at','travel_m','turn_rad'}
            or not vector(reference['abc'],3) or any(not _finite_number(reference[k]) or reference[k]<0 for k in ('at','travel_m','turn_rad'))):
            raise CheckpointRejected('Invalid settled road reference')
    if surface is not None:
        if (not isinstance(surface,dict) or set(surface)!={'track_id','direction','at','position','lo','hi'}
            or type(surface['track_id']) is not int or surface['direction'] not in ('LEFT','RIGHT','AHEAD')
            or not _finite_number(surface['at']) or surface['at']<0
            or any(not vector(surface[k],2) for k in ('position','lo','hi'))
            or any(a>b for a,b in zip(surface['lo'],surface['hi']))):raise CheckpointRejected('Invalid warning surface')


def encode_state(engine, telemetry, assembler, motion):
    """Return bounded canonical JSON bytes for reviewed exact runtime states."""
    _fields(engine, ENGINE_FIELDS); _fields(telemetry, TELEMETRY_FIELDS)
    _fields(assembler, ASSEMBLER_FIELDS); _fields(motion, MOTION_FIELDS)
    _fields(engine.ground, GROUND_FIELDS);_validate_ground(vars(engine.ground))
    _fields(engine.tracker, {'tracks','next_id'}); _fields(telemetry.clock, {'samples'})
    tracks=[]
    for key, track in engine.tracker.tracks.items():
        _fields(track, {f.name for f in fields(Track)})
        if key != track.id:raise CheckpointRejected('Track ID mismatch')
        _validate_near_edge(track.near_edge_state)
        tracks.append({f.name:getattr(track,f.name) for f in fields(Track)})
    _validate_radar(engine._radar_streaks,engine._radar_publication_times,engine._radar_last_update_ns)
    _validate_longitudinal(engine._longitudinal_evidence.history)
    _validate_recovery(engine._settled_road_reference,engine._warning_surface)
    value={'codec_version':CODEC_VERSION,'rng_contract':'Ground.fit resets default_rng(17) per call; no advancing/global RNG',
           'engine':{'mount':asdict(engine.mount),'policy':engine.policy,'last_ns':engine.last_ns,'held':engine.held,
                     'last_alert':engine.last_alert,'last_any_alert':engine.last_any_alert,'last_alert_urgent':engine.last_alert_urgent,'last_warning':engine.last_warning,'settled_road_reference':engine._settled_road_reference,'warning_surface':engine._warning_surface,
                     'ground':vars(engine.ground),'tracks':tracks,'next_id':engine.tracker.next_id,
                     'far_accumulator':_far_state(engine._far_accumulator),'lead_history':_lead_state(engine._lead_history),
                     'radar_streaks':engine._radar_streaks,'radar_publication_times':engine._radar_publication_times,
                     'radar_last_update_ns':engine._radar_last_update_ns,'longitudinal_history':engine._longitudinal_evidence.history},
           'telemetry':{'clock':list(telemetry.clock.samples),'messages':telemetry.messages,
                        'history':{k:list(v) for k,v in telemetry.history.items()},'commit':telemetry.commit,
                        'rejected':telemetry.rejected,'selected_times':telemetry.selected_times,'selected_records':telemetry.selected_records},
           'assembler':vars(assembler),'motion':vars(motion)}
    raw=canonical(_pack(value)).encode()
    if len(raw)>MAX_STATE_BYTES:raise CheckpointRejected('Checkpoint exceeds state memory budget')
    return raw


def decode_state(raw, *, expected_mount=None, expected_policy=None):
    """Restore only explicit known classes/fields; never execute checkpoint data."""
    if not isinstance(raw,bytes) or len(raw)>MAX_STATE_BYTES:raise CheckpointRejected('Invalid checkpoint size/type')
    try:value=_unpack(json.loads(raw))
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:raise CheckpointRejected('Invalid checkpoint encoding: '+str(exc)) from exc
    if set(value)!= {'codec_version','rng_contract','engine','telemetry','assembler','motion'} or value['codec_version']!=CODEC_VERSION:
        raise CheckpointRejected('Unsupported checkpoint schema')
    e=value['engine'];t=value['telemetry'];a=value['assembler'];m=value['motion']
    if set(e)!={'mount','policy','last_ns','held','last_alert','last_any_alert','last_alert_urgent','last_warning','ground','tracks','next_id',
                'far_accumulator','lead_history','radar_streaks','radar_publication_times','radar_last_update_ns','longitudinal_history','settled_road_reference','warning_surface'}:
        raise CheckpointRejected('Unsupported engine state schema')
    if expected_mount is not None and canonical(e['mount'])!=canonical(asdict(Mount(**expected_mount))):raise CheckpointRejected('Checkpoint mount differs from source calibration')
    if expected_policy is not None and canonical(e['policy'])!=canonical(expected_policy):raise CheckpointRejected('Checkpoint policy differs from source configuration')
    if e['last_warning'] is not None and (not isinstance(e['last_warning'],dict) or set(e['last_warning'])!={'track_id','direction'} or type(e['last_warning']['track_id']) is not int or e['last_warning']['direction'] not in ('AHEAD','LEFT','RIGHT')):raise CheckpointRejected('Invalid warning identity')
    _validate_recovery(e['settled_road_reference'],e['warning_surface'])
    engine=Engine(Mount(**e['mount']),e['policy'])
    engine._settled_road_reference=e['settled_road_reference'];engine._warning_surface=e['warning_surface']
    _validate_ground(e['ground'])
    if not isinstance(e['tracks'],list) or len(e['tracks'])>10000:raise CheckpointRejected('Invalid tracker schema')
    engine._far_accumulator=_restore_far(e['far_accumulator']);engine._lead_history=_restore_lead(e['lead_history'])
    _validate_radar(e['radar_streaks'],e['radar_publication_times'],e['radar_last_update_ns'])
    engine._radar_streaks=e['radar_streaks'];engine._radar_publication_times=e['radar_publication_times']
    engine._radar_last_update_ns=e['radar_last_update_ns']
    _validate_longitudinal(e['longitudinal_history']);engine._longitudinal_evidence.history=e['longitudinal_history']
    for key in ('last_ns','held','last_alert','last_any_alert','last_alert_urgent','last_warning'):setattr(engine,key,e[key])
    for key,val in e['ground'].items():setattr(engine.ground,key,val)
    engine.tracker.tracks={}
    for row in e['tracks']:
        if set(row)!={f.name for f in fields(Track)} or type(row['id']) is not int or row['id'] in engine.tracker.tracks:raise CheckpointRejected('Invalid track schema/identity')
        for field,shape in [('position',(2,)),('velocity',(2,)),('lo',(3,)),('hi',(3,))]:
            if not isinstance(row[field],np.ndarray) or row[field].shape!=shape or not np.isfinite(row[field]).all():raise CheckpointRejected('Invalid track geometry')
        for name in ('observed_position','measured_velocity'):
            if row[name] is not None and (not isinstance(row[name],np.ndarray) or row[name].shape!=(2,) or not np.isfinite(row[name]).all()):raise CheckpointRejected('Invalid motion observation state')
        if type(row['motion_samples']) is not int or row['motion_samples']<0:raise CheckpointRejected('Invalid motion sample count')
        for name in ('strong_motion_s','stable_motion_s','motion_edge_error_m'):
            if type(row[name]) not in (int,float) or not math.isfinite(row[name]) or row[name]<0:raise CheckpointRejected('Invalid motion consistency state')
        _validate_near_edge(row['near_edge_state'])
        engine.tracker.tracks[row['id']]=Track(**row)
    engine.tracker.next_id=e['next_id']
    if type(e['next_id']) is not int or e['next_id']<=max(engine.tracker.tracks,default=0):raise CheckpointRejected('Invalid next track ID')
    if set(t)!={'clock','messages','history','commit','rejected','selected_times','selected_records'} or len(t['clock'])>120 or len(t['history'])>128:raise CheckpointRejected('Invalid telemetry schema/bounds')
    if not isinstance(t['selected_records'],dict) or len(t['selected_records'])>128 or any(not isinstance(k,str) or not isinstance(v,dict) for k,v in t['selected_records'].items()):raise CheckpointRejected('Invalid selected native telemetry records')
    telemetry=Telemetry();telemetry.clock.samples=deque(t['clock'],maxlen=120)
    telemetry.messages=t['messages'];telemetry.commit=t['commit'];telemetry.rejected=t['rejected'];telemetry.selected_times=t['selected_times']
    telemetry.selected_records=t['selected_records']
    for service,rows in t['history'].items():
        if not isinstance(service,str) or not isinstance(rows,list) or len(rows)>200:raise CheckpointRejected('Invalid telemetry history bounds')
        telemetry.history[service]=deque(rows,maxlen=200)
    if set(a)!=ASSEMBLER_FIELDS or a['expected'] not in (0,630,1260) or len(a['packets'])>1260 or set(a['packets'])!=set(a['arrivals']):raise CheckpointRejected('Invalid assembler schema')
    for seq,packet in a['packets'].items():
        if type(seq) is not int or not valid_msop(packet) or metadata(packet)['sequence']!=seq:raise CheckpointRejected('Invalid checkpoint packet')
    assembler=FrameAssembler()
    for key,val in a.items():setattr(assembler,key,val)
    if set(m)!=MOTION_FIELDS:raise CheckpointRejected('Invalid motion registration schema')
    motion=ScanMotion()
    for key,val in m.items():setattr(motion,key,val)
    if motion.previous is not None and (not isinstance(motion.previous,np.ndarray) or motion.previous.ndim!=2 or motion.previous.shape[1]!=2 or len(motion.previous)>1800):raise CheckpointRejected('Invalid registration cloud')
    return engine,telemetry,assembler,motion


def _code_digest(name):
    module=importlib.import_module(name)
    path=Path(getattr(module,'__file__',''))
    if path.is_file():return hashlib.sha256(path.read_bytes()).hexdigest()
    loader=getattr(module,'__loader__',None)
    code=loader.get_code(name) if loader and hasattr(loader,'get_code') else None
    if code is None:raise CheckpointRejected('Cannot fingerprint replay implementation: '+name)
    # Hash trusted already-installed code; never marshal-load external state.
    # Frozen loaders may relocate co_filename into a fresh _MEI directory on
    # every launch. Normalize only filenames, not executable instructions/data.
    def stable(value):
        if not isinstance(value,types.CodeType):return value
        return value.replace(co_filename=name.replace('.','/')+'.py',co_consts=tuple(stable(v) for v in value.co_consts))
    return hashlib.sha256(marshal.dumps(stable(code))).hexdigest()


def source_fingerprint(folder,stage_version=VERSION):
    folder=Path(folder)
    paths=[folder/'session.json',folder/'final-state.json',folder/'frame-contexts.jsonl']
    paths += [folder/name for name in ('comma.jsonl','comma-device.json','raw-capture-index.json') if (folder/name).exists()]
    paths += sorted(folder.glob('lidar-*.pcap'))
    sources={}
    for path in paths:
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise CheckpointRejected('Source links are unsupported')
        st=path.stat()
        with path.open('rb') as stream:
            if path.suffix=='.json':
                if st.st_size>MAX_JSON_LINE:raise CheckpointRejected('Source metadata exceeds budget')
                checksum=hashlib.sha256(stream.read()).hexdigest();kind='full_sha256'
            else:
                head=stream.read(65536);stream.seek(max(0,st.st_size-65536));tail=stream.read(65536)
                checksum=hashlib.sha256(head+tail).hexdigest();kind='bounded_edge_sha256'
        sources[path.name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'edge_or_full_sha256':checksum,'digest_scope':kind}
    code={name:_code_digest(name) for name in ('fcw_engine','warning_arbitration','ground_recovery','warning_surface','telemetry','frames','m1_protocol','registration','calibration','policy','road_surface',
        'motion_trust','longitudinal_evidence','empirical_calibration','empirical_corridor','forced_experiment','corridor','runtime',
        'far_objects','lead_gate','near_edge_motion','radar_fusion','corridor_occupancy','limited_experiment','map_live',__name__)}
    return {'stage_version':stage_version,'codec_version':CODEC_VERSION,'sources':sources,'implementation':code,
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'machine':platform.machine(),'platform':sys.platform,
            'numerical_threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}


def _empirical_binding(config):
    """Bind historical effective settings to their frozen recorded profile.

    This does not consult or activate the current installation. The historical
    session marker records that live startup accepted the applied artifact;
    this check verifies its copied numeric/digest contract and exact effective
    mount/timing. It does not re-certify the underlying physical evidence.
    """
    profile=config.get('empirical_calibration')
    requested=(profile is not None or config.get('calibration_basis')=='sensor_derived_empirical')
    if not requested:return None
    binding={'valid':False,'profile':profile if isinstance(profile,dict) else None,'reasons':[]}
    reasons=binding['reasons']
    if not isinstance(profile,dict):
        reasons.append('recorded empirical calibration profile is missing or malformed');return binding
    try:
        from empirical_calibration import validate,MAX_HORIZON_S,MAX_TIMING_S,MAX_LATERAL_M,MAX_LONGITUDINAL_M
        check=validate(profile)
        if not check['eligible']:reasons.extend(check['reasons'])
        if config.get('calibration_basis')!='sensor_derived_empirical':
            reasons.append('recording has no accepted empirical runtime marker')
        if not profile.get('mount_epoch') or config.get('mount_epoch')!=profile['mount_epoch']:
            reasons.append('recorded empirical mounting period differs from capture configuration')
        if profile.get('mount',{}).get('measured') is not False or profile.get('mount',{}).get('alignment_verified') is not False:
            reasons.append('recorded empirical profile incorrectly claims surveyed mount flags')
        expected=asdict(Mount(**profile['mount']));expected.update(measured=True,alignment_verified=True)
        actual=asdict(Mount(**config['mount']))
        if expected!=actual:reasons.append('recorded effective mount differs from the empirical profile')
        if config.get('timing_verified') is not True:
            reasons.append('recording did not enable its empirical timing model')
        adjustment=config.get('lidar_time_adjustment_s');bound=config.get('lidar_timing_bound_s')
        if (type(adjustment) not in (int,float) or not math.isfinite(adjustment) or
                adjustment!=profile.get('lidar_time_adjustment_s')):
            reasons.append('recorded lidar time correction differs from its frozen empirical profile')
        if (type(bound) not in (int,float) or not math.isfinite(bound) or
                bound!=profile.get('timing',{}).get('error_bound_s')):
            reasons.append('recorded lidar timing bound differs from its frozen empirical profile')
        limits=profile.get('limits');expected_limits={'max_horizon_s':MAX_HORIZON_S,'max_timing_s':MAX_TIMING_S,
            'max_lateral_m':MAX_LATERAL_M,'max_longitudinal_m':MAX_LONGITUDINAL_M}
        if (not isinstance(limits,dict) or set(limits)!=set(expected_limits) or
                any(type(limits[k]) not in (int,float) or not math.isfinite(limits[k]) or not 0<limits[k]<=v
                    for k,v in expected_limits.items())):
            reasons.append('recorded empirical capability limits are malformed or unsupported')
        source=profile.get('source');report_digest=source.get('report_digest') if isinstance(source,dict) else None
        if not isinstance(report_digest,str) or len(report_digest)!=64 or any(c not in '0123456789abcdef' for c in report_digest):
            reasons.append('recorded applied report identity is missing')
    except (KeyError,ValueError,TypeError,AttributeError,OverflowError,RecursionError) as error:
        reasons.append('recorded empirical calibration contract invalid: '+type(error).__name__)
    binding['reasons']=list(dict.fromkeys(reasons));binding['valid']=not binding['reasons']
    return binding


def _empirical_replay_context(ctx,original,quality,binding,frame_issues):
    """Re-run only the recorded empirical lidar capability, never live fusion."""
    if binding is None:return ctx,None
    p=binding.get('profile') or {};issues=list(binding['reasons'])+list(frame_issues)
    if binding['valid']:
        if original.get('mount_epoch')!=p['mount_epoch'] or original.get('calibration_id')!=p['digest'] or original.get('calibration_verified') is not True:
            issues.append('recorded frame calibration identity differs from the session profile')
        if 'empirical_profile_id' in original and original['empirical_profile_id']!=p['id']:
            issues.append('recorded frame empirical profile identity differs')
        if 'lidar_time_adjustment_s' in original and original['lidar_time_adjustment_s']!=p['lidar_time_adjustment_s']:
            issues.append('recorded frame lidar time correction differs')
    profile_id=p.get('id') if isinstance(p.get('id'),str) and len(p['id'])<=256 else None
    profile_digest=p.get('digest') if isinstance(p.get('digest'),str) and len(p['digest'])==64 else None
    audit={'mode':'recorded_empirical_lidar_only','profile_id':profile_id,'profile_digest':profile_digest,
           'profile_bound_to_capture':binding['valid'],'radar_fusion_replayed':False,
           'camera_fusion_replayed':False,'map_prior_replayed':False,'surveyed_ground_truth':False,
           'timestamp_handling':'recorded corrected at_ns retained; no second time adjustment',
           'source_cutoff_handling':'original capture-time receipt cutoff, no future arrivals',
           'reasons':list(dict.fromkeys(issues))}
    if issues:
        out=dict(ctx);out.update(ready=False,motion_ready=False,empirical_profile_validated=False,motion_verified=False,
            reasons=list(dict.fromkeys(list(ctx.get('reasons',[]))+['empirical replay withheld: '+v for v in issues])),
            independent_corridor={'active':False,'eligible':False,'empirical':True,'coverage':'unavailable',
                                  'reasons':audit['reasons'],'surveyed_ground_truth':False})
        audit['status']='withheld_invalid_recorded_evidence'
        return out,audit
    from empirical_corridor import prepare_context
    out=prepare_context(ctx,p,quality)
    out.update(mount_epoch=p['mount_epoch'],calibration_id=p['digest'],calibration_verified=True)
    audit['status']='lidar_capability_recomputed'
    audit['empirical_corridor_active']=(out.get('independent_corridor') or {}).get('active') is True
    return out,audit


def _engine_quality(quality):
    """A malformed recorded field must withhold replay, not crash comparisons."""
    result=dict(quality)
    for key,fallback,maximum in (('completeness',0.,1.),('arrival_age_s',1.,1e6),('timing_bound_s',1.,1e6)):
        value=result.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=maximum:
            result[key]=fallback
    result['timestamp_valid']=result.get('timestamp_valid') is True
    return result


def _space(root,minimum,extra=4*MAX_STATE_BYTES):
    if shutil.disk_usage(root).free<=minimum+extra:raise OSError('Exact replay paused above disk reserve')


@contextmanager
def _db(path,immediate=False):
    db=sqlite3.connect(path,timeout=5);db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-4096')
    try:
        if immediate:db.execute('BEGIN IMMEDIATE')
        with db:yield db
    finally:db.close()


class PacketCursor:
    """Seekable equivalent of m1_protocol.read_pcap with explicit record offset."""
    def __init__(self,folder,names,index=0,offset=24):
        self.folder=Path(folder);self.names=names;self.index=index;self.offset=offset;self.stream=None

    def close(self):
        if self.stream:self.stream.close();self.stream=None

    def next(self,cancel=None):
        while self.index<len(self.names):
            check_work(cancel)
            if self.stream is None:
                self.stream=(self.folder/self.names[self.index]).open('rb');header=self.stream.read(24)
                kinds={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
                if len(header)!=24 or header[:4] not in kinds:raise CheckpointRejected('Unsupported PCAP header')
                self.endian,self.scale=kinds[header[:4]];self.link=struct.unpack(self.endian+'I',header[20:24])[0]
                if self.link not in (1,101):raise CheckpointRejected('Unsupported PCAP link type')
                if self.offset<24 or self.offset>os.fstat(self.stream.fileno()).st_size:raise CheckpointRejected('PCAP cursor outside source')
                self.stream.seek(self.offset)
            start=self.offset;record=self.stream.read(16)
            if not record:self.close();self.index+=1;self.offset=24;continue
            if len(record)!=16:raise CheckpointRejected('Truncated PCAP record header')
            sec,sub,length,original=struct.unpack(self.endian+'IIII',record)
            if length>1_000_000:raise CheckpointRejected('Invalid PCAP record length')
            ip=self.stream.read(length)
            if len(ip)!=length:raise CheckpointRejected('Truncated PCAP packet')
            self.offset=self.stream.tell();raw_digest=hashlib.sha256(record+ip).hexdigest()
            part=self.names[self.index]
            if self.link==1:
                if len(ip)<14:return part,start,raw_digest,None
                ethertype=int.from_bytes(ip[12:14],'big');off=14
                while ethertype in (0x8100,0x88a8) and len(ip)>=off+4:ethertype=int.from_bytes(ip[off+2:off+4],'big');off+=4
                if ethertype!=0x0800:return part,start,raw_digest,None
                ip=ip[off:]
            if len(ip)<28 or ip[0]>>4!=4 or ip[9]!=17:return part,start,raw_digest,None
            ihl=(ip[0]&15)*4
            if ihl<20 or len(ip)<ihl+8 or int.from_bytes(ip[6:8],'big')&0x3fff:return part,start,raw_digest,None
            srcport,dstport,size,_=struct.unpack('!HHHH',ip[ihl:ihl+8])
            if size<8 or len(ip)<ihl+size:return part,start,raw_digest,None
            return part,start,raw_digest,(sec*1_000_000_000+sub*self.scale,(socket.inet_ntoa(ip[12:16]),srcport),dstport,ip[ihl+8:ihl+size])
        return None


def _jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def _line(value):
    return json.dumps(value,default=_jsonable,separators=(',',':'))+'\n'


def _chain(previous,raw):
    return hashlib.sha256(bytes.fromhex(previous)+len(raw).to_bytes(8,'big')+raw).hexdigest()


def _check_sources(folder,fingerprint):
    for name,expected in fingerprint['sources'].items():
        stat=(folder/name).stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
            raise CheckpointRejected('Source changed during replay: '+name)
    if sorted(p.name for p in folder.glob('lidar-*.pcap'))!=sorted(n for n in fingerprint['sources'] if n.endswith('.pcap')):
        raise CheckpointRejected('Raw capture inventory changed during replay')


def _save(db,engine,telemetry,assembler,motion,cursor,space_check=None):
    state=encode_state(engine,telemetry,assembler,motion)
    c=canonical(_pack(cursor))
    if len(c)>MAX_STATE_BYTES:raise CheckpointRejected('Checkpoint cursor exceeds state memory budget')
    if space_check:space_check(3*(len(state)+len(c))+16*1024*1024)
    checksum=hashlib.sha256(state+c.encode()).hexdigest()
    db.execute('INSERT INTO checkpoint VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,cursor=excluded.cursor,sha256=excluded.sha256',
               (state,c,checksum))


def _load(db,config):
    row=db.execute('SELECT * FROM checkpoint WHERE id=1').fetchone()
    if row is None:raise CheckpointRejected('Checkpoint state is missing')
    raw=bytes(row['state']);c=row['cursor']
    if len(raw)>MAX_STATE_BYTES or len(c)>MAX_STATE_BYTES or hashlib.sha256(raw+c.encode()).hexdigest()!=row['sha256']:
        raise CheckpointRejected('Checkpoint digest mismatch')
    from policy import canonical_policy
    runtime=decode_state(raw,expected_mount=config['mount'],expected_policy=canonical_policy(config.get('perception_policy')))
    from forced_experiment import configured
    runtime[0].forced_experiment=configured(config)
    cursor=_unpack(json.loads(c))
    if cursor.get('phase') not in ('context_index','replay','frames_complete','complete'):
        raise CheckpointRejected('Invalid replay phase')
    for name in ('context_offset','context_line','pcap_index','pcap_offset','comma_offset','frames','packets','legacy_cutoffs','unmatched_frames'):
        if type(cursor.get(name)) is not int or cursor[name]<0:raise CheckpointRejected('Invalid replay cursor: '+name)
    if db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]!=cursor['frames']:
        raise CheckpointRejected('Committed decisions and checkpoint frame count differ')
    return (*runtime,cursor)


def _next_comma(stream,cursor,cancel):
    if stream is None:return None
    stream.seek(cursor['comma_offset'])
    while True:
        check_work(cancel)
        raw=stream.readline(MAX_JSON_LINE+1)
        if not raw:return None
        if len(raw)>MAX_JSON_LINE:raise CheckpointRejected('Comma record exceeds bounded replay line size')
        cursor['comma_offset']=stream.tell();cursor['comma_content_chain']=_chain(cursor['comma_content_chain'],raw)
        try:record=json.loads(raw)
        except (ValueError,UnicodeError):cursor['telemetry_diagnostics']['invalid_json_rows']=cursor['telemetry_diagnostics'].get('invalid_json_rows',0)+1;continue
        if isinstance(record,dict) and type(record.get('laptop_receipt_ns')) is int:return record
        cursor['telemetry_diagnostics']['invalid_record_rows']=cursor['telemetry_diagnostics'].get('invalid_record_rows',0)+1


def _export_rows(dbpath,work_root,cancel,minimum):
    products={}
    for column,name in (('decision','recomputed-decisions.jsonl'),('context','recomputed-contexts.jsonl')):
        target=work_root/name;temporary=target.with_name(name+'.'+uuid.uuid4().hex+'.tmp')
        h=hashlib.sha256();written=0
        try:
            with temporary.open('xb') as output,_db(dbpath) as db:
                for row in db.execute('SELECT decision,context,sha256 FROM decisions ORDER BY sequence'):
                    check_work(cancel)
                    if hashlib.sha256((row['decision']+row['context']).encode()).hexdigest()!=row['sha256']:
                        raise CheckpointRejected('Committed derived row digest mismatch')
                    raw=row[column].encode()
                    if written//(1024*1024)!=(written+len(raw))//(1024*1024) or not written:_space(work_root,minimum,len(raw)+16*1024*1024)
                    output.write(raw);h.update(raw);written+=len(raw)
                output.flush();os.fsync(output.fileno())
            check_work(cancel);os.replace(temporary,target)
            stat=target.stat();products[column]={'path':str(target),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':h.hexdigest()}
        finally:
            if temporary.exists():temporary.unlink()  # this call's owned derived temporary only
    return products


def run_replay(folder,work_root,cancel=None,*,stage_version=VERSION,checkpoint_frames=10,
               checkpoint_packets=25000,index_chunk_records=500,minimum_free_bytes=MINIMUM_FREE_BYTES,
               progress=None,max_timing_observations=1_000_000):
    """Resume exact raw replay and return a report plus owned derived products.

    Caller must own a single worker lease for work_root. No report is published
    into the source session. Cancellation throws WorkPaused with the last fully
    committed chunk intact. Timing summary can explicitly abstain at its bounded
    input budget; per-frame Engine decisions are still preserved exactly.
    """
    if type(checkpoint_frames) is not int or not 1<=checkpoint_frames<=100 or type(checkpoint_packets) is not int or not 1<=checkpoint_packets<=100000:
        raise ValueError('Replay checkpoint bounds are invalid')
    if type(index_chunk_records) is not int or not 1<=index_chunk_records<=5000:raise ValueError('Context-index chunk bound is invalid')
    if type(max_timing_observations) is not int or not 1<=max_timing_observations<=1_000_000:raise ValueError('Timing-observation memory bound is invalid')
    if not isinstance(minimum_free_bytes,(int,float)) or not math.isfinite(minimum_free_bytes) or minimum_free_bytes<0:raise ValueError('Invalid disk reserve')
    check_work(cancel)
    folder=Path(folder).resolve();work_root=Path(work_root).resolve()
    if folder==work_root:raise ValueError('Exact replay requires a separate owned work directory')
    if work_root.is_symlink() or getattr(work_root,'is_junction',lambda:False)():raise ValueError('Replay work directory may not be a link')
    anchor=work_root
    while not anchor.exists():anchor=anchor.parent
    _space(anchor,minimum_free_bytes)
    fingerprint=source_fingerprint(folder,stage_version);config=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    empirical_binding=_empirical_binding(config)
    work_root.mkdir(parents=True,exist_ok=True);dbpath=work_root/'replay.sqlite'
    existing_checkpoint=dbpath.exists()
    space_check=lambda needed:_space(work_root,minimum_free_bytes,needed)
    if existing_checkpoint:
        probe=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)
        try:
            if probe.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID:raise CheckpointRejected('Unrecognized replay database')
        finally:probe.close()
    with _db(dbpath,True) as db:
        db.execute('PRAGMA application_id='+str(APPLICATION_ID))
        db.executescript('''
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY CHECK(id=1),state BLOB NOT NULL,cursor TEXT NOT NULL,sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contexts(sequence INTEGER PRIMARY KEY,device_key TEXT NOT NULL,offset INTEGER NOT NULL,length INTEGER NOT NULL,sha256 TEXT NOT NULL,used INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS contexts_lookup ON contexts(device_key,used,sequence);
CREATE TABLE IF NOT EXISTS decisions(sequence INTEGER PRIMARY KEY,at_ns INTEGER NOT NULL,decision TEXT NOT NULL,context TEXT NOT NULL,sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS timing(sequence INTEGER NOT NULL,kind TEXT NOT NULL,at_s REAL NOT NULL,yaw REAL NOT NULL,PRIMARY KEY(sequence,kind));
''')
        old=db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if old and json.loads(old[0])!=fingerprint:raise CheckpointRejected('Replay source/version/calibration fingerprint changed; use a new work revision')
        if not old:
            db.execute('INSERT INTO meta VALUES(?,?)',('fingerprint',canonical(fingerprint)))
            from forced_experiment import configured
            engine=Engine(Mount(**config['mount']),config.get('perception_policy'),forced_experiment=configured(config))
            cursor={'phase':'context_index','context_offset':0,'context_line':0,'pcap_index':0,'pcap_offset':24,
                    'comma_offset':0,'next_raw':None,'comma_initialized':False,'frames':0,'packets':0,'legacy_cutoffs':0,
                    'unmatched_frames':0,'telemetry_diagnostics':{},'last_comma_time':None,
                    'pcap_content_chain':'0'*64,'comma_content_chain':'0'*64}
            if empirical_binding is not None:
                cursor.update(last_empirical_cutoff_ns=None,empirical_frames=0,empirical_withheld_frames=0)
            _save(db,engine,Telemetry(),FrameAssembler(),ScanMotion(),cursor,space_check)
        engine,telemetry,assembler,motion,cursor=_load(db,config)
        resume_start={'phase':cursor['phase'],'frames':cursor['frames'],'pcap_index':cursor['pcap_index'],'pcap_offset':cursor['pcap_offset']}
    # Context indexing persists byte offsets and content digests, never all paths.
    while cursor['phase']=='context_index':
        check_work(cancel);_space(work_root,minimum_free_bytes)
        with _db(dbpath,True) as db,(folder/'frame-contexts.jsonl').open('rb') as source:
            engine,telemetry,assembler,motion,cursor=_load(db,config);source.seek(cursor['context_offset'])
            consumed=0
            for _ in range(index_chunk_records):
                check_work(cancel);offset=source.tell();raw=source.readline(MAX_JSON_LINE+1)
                if not raw:cursor['phase']='replay';break
                if len(raw)>MAX_JSON_LINE:raise CheckpointRejected('Context record exceeds bounded line size')
                cursor['context_line']+=1;cursor['context_offset']=source.tell();consumed+=len(raw)
                try:
                    record=json.loads(raw);key=record['quality']['first_device_ns']
                    if type(key) is not int:raise ValueError()
                except (ValueError,KeyError,TypeError,UnicodeError) as exc:raise CheckpointRejected('Malformed frame context; replay coverage cannot be established') from exc
                db.execute('INSERT INTO contexts(sequence,device_key,offset,length,sha256) VALUES(?,?,?,?,?)',
                           (cursor['context_line'],str(key),offset,len(raw),hashlib.sha256(raw).hexdigest()))
                if consumed>=8*1024*1024:break
            _check_sources(folder,fingerprint);_save(db,engine,telemetry,assembler,motion,cursor,space_check)
        if progress:progress({'phase':cursor['phase'],'frames':cursor['frames'],'context_bytes':cursor['context_offset']})
    from runtime import ingest_telemetry_record,safe_telemetry_context,APP_VERSION
    captures=sorted(n for n in fingerprint['sources'] if n.endswith('.pcap'))
    while cursor['phase']=='replay':
        check_work(cancel);_space(work_root,minimum_free_bytes)
        packet_cursor=None;comma_stream=None
        try:
            with _db(dbpath,True) as db,(folder/'frame-contexts.jsonl').open('rb') as context_file:
                engine,telemetry,assembler,motion,cursor=_load(db,config)
                packet_cursor=PacketCursor(folder,captures,cursor['pcap_index'],cursor['pcap_offset'])
                comma_path=folder/'comma.jsonl';comma_stream=comma_path.open('rb') if comma_path.exists() else None
                if not cursor['comma_initialized']:
                    cursor['next_raw']=_next_comma(comma_stream,cursor,cancel);cursor['comma_initialized']=True
                initial_frames=cursor['frames'];initial_packets=cursor['packets']
                while cursor['frames']-initial_frames<checkpoint_frames and cursor['packets']-initial_packets<checkpoint_packets:
                    check_work(cancel);row=packet_cursor.next(cancel)
                    if row is None:cursor['phase']='frames_complete';break
                    part,offset,packet_digest,udp=row;cursor['packets']+=1
                    cursor['pcap_content_chain']=_chain(cursor['pcap_content_chain'],(part+':'+str(offset)+':'+packet_digest).encode())
                    if udp is None:continue
                    wall,source,port,payload=udp
                    if port!=config['msop_port'] or source[0]!=config['sensor_ip']:continue
                    for packets,q in assembler.push(payload,wall):
                        matched=db.execute('SELECT * FROM contexts WHERE device_key=? AND used=0 ORDER BY sequence LIMIT 1',(str(q['first_device_ns']),)).fetchone()
                        if matched is None:cursor['unmatched_frames']+=1;continue
                        context_file.seek(matched['offset']);raw=context_file.read(matched['length'])
                        if hashlib.sha256(raw).hexdigest()!=matched['sha256']:raise CheckpointRejected('Indexed context content changed')
                        record=json.loads(raw);points,stamps=decode_with_time(packets)
                        db.execute('UPDATE contexts SET used=1 WHERE sequence=?',(matched['sequence'],))
                        if not len(points):continue
                        original_context=record['context']
                        if (not isinstance(original_context,dict) or type(original_context.get('at_ns')) is not int or
                                not 0<original_context['at_ns']<2**63):
                            raise CheckpointRejected('Recorded frame query timestamp is missing or invalid')
                        quality={**record['quality'],'context_line':matched['sequence'],'raw_capture_completion_part':part,'raw_capture_last_wall_ns':wall}
                        cutoff=quality.get('fusion_cutoff_ns');frame_issues=[]
                        if cutoff is None:cutoff=quality.get('last_receipt_ns');cursor['legacy_cutoffs']+=1
                        if empirical_binding is None and (type(cutoff) is not int or cutoff<=0):
                            raise CheckpointRejected('Recorded source receipt cutoff is invalid')
                        if empirical_binding is not None:
                            recorded_cutoff=quality.get('fusion_cutoff_ns')
                            if type(recorded_cutoff) is not int or not 0<recorded_cutoff<2**63:
                                frame_issues.append('original empirical source receipt cutoff is missing or invalid')
                                fallback=quality.get('last_receipt_ns')
                                cutoff=fallback if type(fallback) is int and fallback>0 else 0
                            previous=cursor.get('last_empirical_cutoff_ns')
                            if previous is not None and cutoff<previous:
                                frame_issues.append('recorded empirical source cutoff regressed')
                            cursor['last_empirical_cutoff_ns']=max(previous or 0,cutoff)
                            receipt=quality.get('last_receipt_ns')
                            if type(receipt) is not int or receipt<=0 or cutoff<receipt:
                                frame_issues.append('source cutoff precedes the recorded lidar delivery')
                            completeness=quality.get('completeness');raw_completeness=q.get('completeness')
                            if (type(completeness) not in (int,float) or not math.isfinite(completeness) or not 0<=completeness<=1 or
                                    type(raw_completeness) not in (int,float) or not math.isfinite(raw_completeness) or
                                    raw_completeness+1e-9<completeness):
                                frame_issues.append('raw frame completeness does not support the recorded quality')
                            if q.get('timestamp_valid') is not True:
                                frame_issues.append('raw frame timestamps do not support empirical replay')
                            timing_profile=(empirical_binding.get('profile') or {}).get('timing',{})
                            base_bound=timing_profile.get('error_bound_s') if isinstance(timing_profile,dict) else None
                            recorded_bound=quality.get('timing_bound_s')
                            if (type(recorded_bound) not in (int,float) or not math.isfinite(recorded_bound) or recorded_bound<0 or
                                    (type(base_bound) in (int,float) and recorded_bound+1e-12<base_bound)):
                                frame_issues.append('recorded frame timing uncertainty understates the applied profile')
                        while cursor['next_raw'] is not None and cursor['next_raw']['laptop_receipt_ns']<=cutoff:
                            check_work(cancel);r=cursor['next_raw']
                            try:
                                if ingest_telemetry_record(telemetry,r,r['laptop_receipt_ns'],r.get('laptop_send_ns')):
                                    key='payloads_invalidated';cursor['telemetry_diagnostics'][key]=cursor['telemetry_diagnostics'].get(key,0)+1
                            except Exception:
                                key='records_rejected';cursor['telemetry_diagnostics'][key]=cursor['telemetry_diagnostics'].get(key,0)+1
                            cursor['next_raw']=_next_comma(comma_stream,cursor,cancel)
                        ctx,problem=safe_telemetry_context(telemetry,original_context['at_ns'])
                        if problem:
                            key='context_failures';cursor['telemetry_diagnostics'][key]=cursor['telemetry_diagnostics'].get(key,0)+1
                        ctx,replay_capabilities=_empirical_replay_context(ctx,original_context,quality,empirical_binding,frame_issues)
                        if engine.forced_experiment:
                            if not engine.forced_experiment.matches(original_context):raise CheckpointRejected('Forced experiment frame/model identity differs')
                            ctx=engine.forced_experiment.prepare(ctx)
                        if replay_capabilities is not None:
                            cursor['empirical_frames']=cursor.get('empirical_frames',0)+1
                            cursor['empirical_withheld_frames']=cursor.get('empirical_withheld_frames',0)+(replay_capabilities['status']=='withheld_invalid_recorded_evidence')
                        decision=engine.update(points,ctx,_engine_quality(quality) if empirical_binding is not None else quality,(stamps-stamps.max())/1e9)
                        decision['processing_timing_scope']='offline_replay; original live engine/decode cost is unknown here'
                        if replay_capabilities is not None:
                            decision['replay_capabilities']=replay_capabilities
                        normalized=_line({'context':ctx,'quality':quality,'selected_source_times':dict(telemetry.selected_times)})
                        encoded=_line(decision);sequence=cursor['frames']
                        _space(work_root,minimum_free_bytes,2*(len(encoded)+len(normalized))+3*MAX_STATE_BYTES)
                        db.execute('INSERT INTO decisions VALUES(?,?,?,?,?)',(sequence,int(ctx['at_ns']),encoded,normalized,hashlib.sha256((encoded+normalized).encode()).hexdigest()))
                        cursor['frames']+=1
                        check_work(cancel)
                        effective=int(ctx['at_ns']+np.median(stamps)-stamps.max())
                        registration=motion.update(transform(points,engine.mount),effective)
                        if registration.get('valid'):db.execute('INSERT INTO timing VALUES(?,?,?,?)',(sequence,'lidar',registration['at_ns']/1e9,registration['yaw_rate_rps']))
                        if ctx.get('motion_ready',ctx.get('ready')) and ctx.get('motion_source_ns'):
                            source_time=ctx['motion_source_ns']/1e9
                            if cursor['last_comma_time'] is None or source_time>cursor['last_comma_time']:
                                db.execute('INSERT INTO timing VALUES(?,?,?,?)',(sequence,'comma',source_time,ctx['yaw_rate_rps']));cursor['last_comma_time']=source_time
                cursor['pcap_index']=packet_cursor.index;cursor['pcap_offset']=packet_cursor.offset
                _check_sources(folder,fingerprint);_save(db,engine,telemetry,assembler,motion,cursor,space_check)
        finally:
            if packet_cursor:packet_cursor.close()
            if comma_stream:comma_stream.close()
        if progress:progress({'phase':cursor['phase'],'frames':cursor['frames'],'packets':cursor['packets'],
                              'pcap_index':cursor['pcap_index'],'pcap_offset':cursor['pcap_offset']})
    check_work(cancel)
    # Export is disposable and streaming: damage to a derived tail never forces
    # re-reading raw captures or resets committed engine/track state.
    products=_export_rows(dbpath,work_root,cancel,minimum_free_bytes)
    with _db(dbpath) as db:
        counts={r['kind']:r['n'] for r in db.execute('SELECT kind,COUNT(*) n FROM timing GROUP BY kind')}
        remaining=db.execute('SELECT COUNT(*) FROM contexts WHERE used=0').fetchone()[0]
        if max(counts.values(),default=0)>max_timing_observations:
            timing={'accepted':False,'reason':'timing summary exceeds explicit observation memory budget; frame replay is complete'}
        elif min(counts.get('lidar',0),counts.get('comma',0))>=100:
            check_work(cancel)
            rows={kind:np.asarray([(r['at_s'],r['yaw']) for r in db.execute('SELECT at_s,yaw FROM timing WHERE kind=? ORDER BY sequence',(kind,))],float) for kind in ('lidar','comma')}
            from calibration import timing_alignment
            timing=timing_alignment(rows['lidar'][:,0],rows['lidar'][:,1],rows['comma'][:,0],rows['comma'][:,1])
        else:timing={'accepted':False,'reason':'not enough independently registered turning data'}
    check_work(cancel);_check_sources(folder,fingerprint)
    device=json.loads((folder/'comma-device.json').read_text()) if (folder/'comma-device.json').exists() else {}
    report={'redecoded_frames':cursor['frames'],'unmatched_frames':cursor['unmatched_frames'],'replay_version':APP_VERSION,
            'recording_version':config.get('version','unknown'),'telemetry_record_diagnostics':cursor['telemetry_diagnostics'],
            'renormalized_frames':cursor['frames'],'legacy_conservative_cutoff_frames':cursor['legacy_cutoffs'],
            'replay_scope':'raw lidar + recorded comma messages; source normalization re-run without future messages',
            'contexts_without_raw_frames':remaining,'timing_proposal':timing,'session':str(folder),'mount':config['mount'],
            'comma_commit':device.get('commit'),'total_time_adjustment_s':config.get('lidar_time_adjustment_s',0)+timing.get('offset_s',0),
            'note':'No calibration or thresholds were automatically activated. Use held-out evidence to validate a proposal.',
            'checkpoint_algorithm':VERSION,'source_fingerprint':digest(fingerprint),'products':products,
            'raw_packet_records_read':cursor['packets'],'consumed_pcap_chain':cursor['pcap_content_chain'],
            'consumed_comma_prefix_chain':cursor['comma_content_chain'],
            'recovered_from_committed_state':existing_checkpoint,'resume_start':resume_start,'legacy_partial_outputs_reused':False}
    if empirical_binding is not None:
        p=empirical_binding.get('profile') or {}
        report['replay_scope']='raw lidar + causally recorded comma + frozen empirical lidar corridor; optional radar/camera/map fusion is NOT replayed'
        report['empirical_replay']={'profile_id':p.get('id') if isinstance(p.get('id'),str) and len(p['id'])<=256 else None,
            'profile_digest':p.get('digest') if isinstance(p.get('digest'),str) and len(p['digest'])==64 else None,
            'profile_bound_to_capture':empirical_binding['valid'],'binding_reasons':empirical_binding['reasons'],
            'frames':cursor.get('empirical_frames',0),'withheld_recorded_evidence_frames':cursor.get('empirical_withheld_frames',0),
            'radar_fusion_replayed':False,'camera_fusion_replayed':False,'map_prior_replayed':False,
            'timestamp_handling':'original corrected query timestamps reused, calibration offset never applied twice',
            'installation_changed':False,'physical_qualification_performed':False}
    with _db(dbpath,True) as db:
        cursor['phase']='complete';_save(db,engine,telemetry,assembler,motion,cursor,space_check)
        db.execute('INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',('report',canonical(report)))
    return report

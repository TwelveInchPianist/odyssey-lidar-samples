"""Bounded framing and continuity checks for read-only comma encoded video."""
import json
import struct

MAGIC = b'FCWVID1\0'
PREFIX = struct.Struct('!8sII')
MAX_META = 16 * 1024
MAX_PAYLOAD = 8 * 1024 * 1024
SERVICES = {'roadEncodeData', 'wideRoadEncodeData'}


def write_message(stream, metadata, payload=b''):
    meta = json.dumps(metadata,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(meta)>MAX_META or len(payload)>MAX_PAYLOAD:
        raise ValueError('Video message exceeds bound')
    stream.write(PREFIX.pack(MAGIC,len(meta),len(payload)))
    stream.write(meta)
    stream.write(payload)
    stream.flush()


def _read_exact(stream, size, *, allow_eof=False):
    parts=[]; left=size
    while left:
        data=stream.read(left)
        if not data:
            if allow_eof and left==size:return None
            raise EOFError('Truncated encoded video message')
        parts.append(data);left-=len(data)
    return b''.join(parts)


def read_message(stream):
    prefix=_read_exact(stream,PREFIX.size,allow_eof=True)
    if prefix is None:return None
    magic,meta_size,payload_size=PREFIX.unpack(prefix)
    if magic!=MAGIC or not 0<meta_size<=MAX_META or payload_size>MAX_PAYLOAD:
        raise ValueError('Invalid encoded video framing')
    meta=json.loads(_read_exact(stream,meta_size))
    if not isinstance(meta,dict):raise ValueError('Video metadata must be an object')
    payload=_read_exact(stream,payload_size)
    validate_metadata(meta,len(payload))
    return meta,payload


def validate_metadata(meta,payload_size):
    """Shared native-clock video contract for transport and saved evidence."""
    if meta.get('kind')=='video':
        if meta.get('service') not in SERVICES:raise ValueError('Unexpected camera stream')
        for key in ['encode_id','frame_id','source_ns','sent_ns','width','height','header_bytes']:
            if type(meta.get(key)) is not int or meta[key]<0:raise ValueError('Invalid video '+key)
        if (not 0<meta['source_ns']<=meta['sent_ns'] or not 0<meta['width']<=4096
                or not 0<meta['height']<=4096 or meta['header_bytes']>payload_size):
            raise ValueError('Invalid encoded video timing/shape/header')


class Continuity:
    """A dropped access unit invalidates the GOP until the next key frame."""
    def __init__(self):self.states={}
    def accept(self, meta):
        service=meta['service'];old=self.states.get(service)
        key=meta.get('keyframe') is True
        identity=(meta.get('writer'),meta.get('segment_num'))
        sequence=(old is not None and old['identity']==identity
                  and meta['encode_id']==old['encode_id']+1
                  and meta['source_ns']>old['source_ns'])
        reset=old is None or not sequence
        ready=key or (sequence and old['ready'])
        self.states[service]={'identity':identity,'encode_id':meta['encode_id'],
                              'source_ns':meta['source_ns'],'ready':ready}
        return {'ready':ready,'reset_decoder':reset or (key and not old['ready']),
                'reason':None if ready else 'waiting_for_keyframe_after_attach_or_gap'}

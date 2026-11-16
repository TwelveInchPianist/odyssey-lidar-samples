"""Read-only native encoded road video; run over a separate persistent SSH pipe.

No image re-encoding, subscribers, publishers, fork edits, or on-device files.
The source's existing encoder timestamps and keyframes remain intact. A slow
transport can lose a GOP; continuity and source-age checks expose that loss.
"""
import importlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import threading
import time
from comma_reader import HEADER_SIZE,DEFAULT_SIZE,RingSnapshot,ReadOnlyMessageReader,verify_native_layout
from video_transport import write_message,SERVICES


def source_clock_ns():
    # Inspected camerad processing time and common/timing.h use CLOCK_BOOTTIME.
    # Unlike CLOCK_MONOTONIC, this domain includes any intervening system suspend.
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def all_messages(data, require_wrap=False):
    at=0; messages=[]
    while at<len(data):
        if len(data)-at<8:raise ValueError('Truncated ring length')
        length=struct.unpack_from('<q',data,at)[0]
        if length==-1:
            if not require_wrap:raise ValueError('Unexpected ring wrap')
            return messages
        if length<=0 or length>len(data)-at-8:raise ValueError('Invalid ring payload length')
        end=(at+8+length+7)&~7
        if end>len(data):raise ValueError('Invalid ring alignment')
        messages.append(data[at+8:at+8+length]);at=end
    if require_wrap:raise ValueError('Missing ring wrap')
    return messages


class BatchRingSnapshot(RingSnapshot):
    def receive(self):
        pointer,writer=self.header()
        if not writer or writer!=self.writer or self.cursor is None:
            if self.writer is not None and writer!=self.writer:self.stats['writer_resets']+=1
            self.writer,self.cursor=writer,pointer
            return []
        if pointer==self.cursor:return []
        pos,old_pos=pointer&0xffffffff,self.cursor&0xffffffff
        cycles=((pointer>>32)-(self.cursor>>32))&0xffffffff
        if cycles>1 or (cycles==0 and pos<old_pos) or (cycles==1 and pos>=old_pos):
            self.stats['overruns']+=1;self.cursor=pointer;return []
        try:
            if cycles:
                tail=bytes(self.mapping[HEADER_SIZE+old_pos:])
                head=bytes(self.mapping[HEADER_SIZE:HEADER_SIZE+pos])
                payloads=all_messages(tail,True)+all_messages(head)
            else:
                payloads=all_messages(bytes(self.mapping[HEADER_SIZE+old_pos:HEADER_SIZE+pos]))
        except ValueError:
            self.stats['malformed']+=1;self.cursor=pointer;return []
        if self.header()!=(pointer,writer):
            self.stats['races']+=1;return []
        self.cursor=pointer;self.stats['received']+=len(payloads)
        return payloads


class EncodedReader(ReadOnlyMessageReader):
    def packets(self):
        for name in self.services:
            path=self.root/name if self.prefixed else Path(str(self.root)+name)
            if name not in self.rings:
                try:file=path.open('rb')
                except FileNotFoundError:continue
                expected=getattr(self.registry[name],'queue_size',0) or DEFAULT_SIZE
                if os.fstat(file.fileno()).st_size!=HEADER_SIZE+expected:
                    file.close();raise ValueError('Unsupported encoded ring size')
                mapping=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
                self.files[name]=(file,mapping);self.rings[name]=BatchRingSnapshot(mapping)
            file,mapping=self.files[name];stat=path.stat()
            if stat.st_size!=len(mapping) or stat.st_ino!=os.fstat(file.fileno()).st_ino:
                raise ValueError('Encoded ring changed while observing')
            for payload in self.rings[name].receive():
                event=self.messaging.log_from_bytes(payload)
                if event.which()!=name or not event.valid:continue
                if event.logMonoTime<=self.logMonoTime[name]:continue
                self.logMonoTime[name]=int(event.logMonoTime)
                yield name,getattr(event,name),self.rings[name].writer


def main():
    sys.dont_write_bytecode=True
    os.environ.pop('ZMQ',None)
    sys.path[:0]=['/data/openpilot','/data']
    for prefix in ('cereal','openpilot.cereal'):
        try:
            messaging=importlib.import_module(prefix+'.messaging')
            module=importlib.import_module(prefix+'.services')
            registry=getattr(module,'SERVICE_LIST',getattr(module,'service_list',{}))
            break
        except ImportError:continue
    else:raise RuntimeError('Installed cereal unavailable')
    import msgq.ipc_pyx
    digest=verify_native_layout(msgq.ipc_pyx)
    services=sorted(SERVICES.intersection(registry))
    reader=EncodedReader(services,messaging,registry)
    output=sys.stdout.buffer
    lock=threading.Lock();stop=threading.Event()
    def emit(meta,payload=b''):
        with lock:write_message(output,meta,payload)
    emit({'kind':'hello','schema_version':1,'services':services,
                          'native_sha256':digest,'subscriber_slots_used':0,
                          'source_clock':'linux_boottime',
                          'transport':'read_only_encoded_road_video'})
    def ping_reader():
        try:
            for line in sys.stdin:
                received=source_clock_ns()
                if len(line)>1000:continue
                try:item=json.loads(line)
                except ValueError:continue
                if isinstance(item,dict) and item.get('kind')=='ping' and type(item.get('id')) is int:
                    emit({'kind':'pong','id':item['id'],'remote_receive_ns':received,
                          'remote_send_ns':source_clock_ns()})
        finally:stop.set()
    threading.Thread(target=ping_reader,daemon=True).start()
    last_heartbeat=0
    try:
        while not stop.is_set():
            now=source_clock_ns()
            for service,data,writer in reader.packets():
                idx=data.idx
                stamp=int(idx.timestampEof)
                # Do not begin pushing already-old source frames into a queue.
                sent=source_clock_ns()
                if not 0<=sent-stamp<=250_000_000:continue
                header=bytes(data.header);body=bytes(data.data)
                meta={'kind':'video','service':service,'writer':writer,
                      'encode_id':int(idx.encodeId),'frame_id':int(idx.frameId),
                      'segment_num':int(idx.segmentNum),'source_ns':stamp,
                      'sent_ns':sent,'width':int(data.width),'height':int(data.height),
                      'codec':str(idx.type),'keyframe':bool(int(idx.flags)&8),
                      'header_bytes':len(header)}
                emit(meta,header+body)
            if now-last_heartbeat>1_000_000_000:
                emit({'kind':'heartbeat','sent_ns':now,
                    'rings':{name:ring.stats for name,ring in reader.rings.items()}})
                last_heartbeat=now
            time.sleep(.005)
    finally:reader.close()

if __name__=='__main__':main()

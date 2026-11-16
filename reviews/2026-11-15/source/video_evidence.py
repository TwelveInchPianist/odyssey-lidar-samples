"""Bounded native encoded-video evidence for submitted awareness advisories.

The receiver only keeps immutable compressed packets in a small GOP cache. A
separate writer saves the already-received decoding chain when audio is submitted.
No image is synthesized, re-encoded or substituted for a missing exact reference.
"""
from collections import Counter,deque
import hashlib,json,queue,re,threading
from pathlib import Path
from video_transport import Continuity,MAGIC,PREFIX,MAX_META,MAX_PAYLOAD,SERVICES,validate_metadata

CAMERAS={'roadEncodeData':'narrow','wideRoadEncodeData':'wide'}


class VideoEvidence:
    def __init__(self,folder,*,on_error=None,space_available=None,reserve_bytes=None,
                 max_buffer_bytes=64*1024*1024,max_queue_bytes=64*1024*1024):
        self.folder=Path(folder)/'camera-evidence';self.folder.mkdir(exist_ok=True)
        self.pending=self.folder/'writer.pending'
        with self.pending.open('x',encoding='utf-8') as stream:stream.write('Native camera evidence is still being captured or closed.\n')
        try:self.index=(self.folder/'index.jsonl').open('x',encoding='utf-8',buffering=1)
        except Exception:
            self.pending.unlink(missing_ok=True);raise
        self.on_error=on_error or (lambda _:None);self.space_available=space_available or (lambda _:True)
        self.reserve_bytes=reserve_bytes or (lambda _:True)
        self.max_buffer_bytes=max_buffer_bytes;self.max_queue_bytes=max_queue_bytes
        self.lock=threading.Lock();self.stop_event=threading.Event();self.jobs=queue.Queue(maxsize=32)
        self.continuity=Continuity();self.chains=deque();self.current={};self.buffer_bytes=0;self.queue_bytes=0
        self.next_chain=1;self.metrics=Counter();self.last_error=None;self.seen=set();self.seen_order=deque();self.thread=None

    def start(self):
        if self.thread is not None:raise RuntimeError('Camera evidence writer already started')
        self.thread=threading.Thread(target=self._run,daemon=True,name='camera-evidence-writer');self.thread.start()

    def _error(self,error):
        self.last_error=str(error);self.metrics['errors']+=1
        self.on_error('Camera evidence: '+self.last_error)

    def offer(self,metadata,payload):
        """Nonblocking receiver callback. Capture backpressure cannot stall inference."""
        if self.stop_event.is_set():return False
        if not self.lock.acquire(blocking=False):self.metrics['cache_lock_drops']+=1;return False
        try:
            if self.stop_event.is_set():return False
            service=metadata.get('service');native=metadata.get('native_source_ns')
            if (service not in SERVICES or metadata.get('kind')!='video' or metadata.get('codec')!='fullHEVC' or type(native) is not int or native<=0
                    or type(metadata.get('source_ns')) is not int or type(metadata.get('receipt_ns')) is not int
                    or not 0<metadata['source_ns']<=metadata['receipt_ns']
                    or not isinstance(payload,bytes) or len(payload)>MAX_PAYLOAD):
                raise ValueError('Native timestamped road-video packet required')
            # Transport framing validates source<=sent in the native clock. Keep
            # the mapped laptop clock in a distinct field instead of mixing clocks.
            recorded={**metadata,'source_ns':native,'mapped_source_ns':metadata['source_ns'],
                      'record_clock_schema':'native_plus_mapped-v1'}
            validate_metadata(recorded,len(payload))
            meta_bytes=json.dumps(recorded,separators=(',',':'),allow_nan=False).encode()
            if len(meta_bytes)>MAX_META:raise ValueError('Camera metadata exceeds bound')
            recorded=json.loads(meta_bytes);status=self.continuity.accept(recorded)
            if not status['ready']:
                self.current.pop(service,None);self.metrics['waiting_for_keyframe']+=1;return False
            if recorded.get('keyframe') is True or service not in self.current:
                chain={'id':self.next_chain,'service':service,'messages':[],'bytes':0};self.next_chain+=1
                self.chains.append(chain);self.current[service]=chain
            chain=self.current[service]
            if len(chain['messages'])>=256:
                self.chains.remove(chain);self.buffer_bytes-=chain['bytes'];self.current.pop(service,None)
                self.continuity.states[service]['ready']=False;self.metrics['oversized_gops']+=1;return False
            header=PREFIX.pack(MAGIC,len(meta_bytes),len(payload))+meta_bytes;size=len(header)+len(payload)
            chain['messages'].append((recorded,header,payload));chain['bytes']+=size;self.buffer_bytes+=size
            # Two GOPs per camera cover the measured1.5s native keyframe cadence.
            for candidate in list(self.chains):
                same=[c for c in self.chains if c['service']==candidate['service']]
                if len(same)>2 or self.buffer_bytes>self.max_buffer_bytes:
                    self.chains.remove(candidate);self.buffer_bytes-=candidate['bytes'];self.metrics['evicted_gops']+=1
                    if self.current.get(candidate['service']) is candidate:
                        self.current.pop(candidate['service'],None)
                        self.continuity.states[candidate['service']]['ready']=False
            self.metrics['cached_packets']+=1
            return self.current.get(service) is chain
        except (ValueError,KeyError,TypeError) as error:
            self.metrics['invalid_packets']+=1;self.last_error=str(error);return False
        finally:self.lock.release()

    def bookmark(self,event):
        identity=event.get('id');source=event.get('source_ns');submitted=event.get('submission_ns',event.get('at_ns'))
        if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',identity):return {'state':'invalid_event_identity'}
        if type(source) is not int or type(submitted) is not int or not 0<source<=submitted:return {'state':'invalid_event_time'}
        if not self.lock.acquire(blocking=False):self.metrics['bookmark_lock_drops']+=1;return {'state':'cache_busy','event_id':identity}
        try:
            if self.stop_event.is_set():return {'state':'stopped','event_id':identity}
            if self.thread is None or not self.thread.is_alive():return {'state':'writer_unavailable','event_id':identity}
            if identity in self.seen:return {'state':'already_requested','event_id':identity}
            clips={};targets=[];missing=[]
            for service,camera in CAMERAS.items():
                references=[o for o in event.get('evidence',[]) if o.get('camera')==camera or o.get('sensor')=='camera:'+camera]
                available=[(chain,i,meta) for chain in self.chains if chain['service']==service
                           for i,(meta,_,_) in enumerate(chain['messages']) if meta['receipt_ns']<=submitted]
                requests=references or [None]
                requested=set()
                for reference in requests:
                    if reference is not None:
                        fid=reference.get('frame_id');native=reference.get('native_source_ns');stream=reference.get('stream_identity')
                        key=(fid,native)
                        if key in requested:continue
                        requested.add(key)
                        matches=[row for row in available if row[2].get('frame_id')==fid and row[2].get('native_source_ns')==native
                                 and (not isinstance(stream,dict) or all(row[2].get(k)==v for k,v in stream.items()))]
                        match=matches[-1] if len(matches)==1 else None;kind='exact_inference_frame'
                    else:
                        candidates=[row for row in available if abs(row[2]['mapped_source_ns']-source)<=250_000_000]
                        match=min(candidates,key=lambda row:abs(row[2]['mapped_source_ns']-source)) if candidates else None
                        kind='nearest_context_frame'
                    if match is None:
                        missing.append({'camera':camera,'kind':kind,'reason':'matching_decodable_frame_not_in_cache'});continue
                    chain,_,meta=match
                    filename=identity+'-'+camera+'-'+str(chain['id'])+'.fcv'
                    if filename not in clips:
                        # Include every already-received access unit, allowing
                        # decoder reorder without borrowing future video.
                        messages=tuple(message for message in chain['messages'] if message[0]['receipt_ns']<=submitted)
                        clips[filename]={'file':filename,'camera':camera,'messages':messages,
                            'bytes':sum(len(h)+len(p) for _,h,p in messages)}
                    targets.append({'camera':camera,'kind':kind,'clip':filename,'frame_id':meta['frame_id'],
                        'native_source_ns':meta['native_source_ns'],'mapped_source_ns':meta['mapped_source_ns'],
                        'source_interval_ns':meta.get('source_interval_ns'),'offset_from_event_ns':meta['mapped_source_ns']-source,
                        'identity':{k:meta.get(k) for k in ['writer','segment_num','encode_id','frame_id']}})
            size=sum(c['bytes'] for c in clips.values())
            if self.jobs.full() or self.queue_bytes+size>self.max_queue_bytes:
                self.metrics['bookmark_backpressure']+=1;return {'state':'writer_backpressure','event_id':identity}
            if not self.reserve_bytes(size):
                clips={};targets=[];size=0;missing.append({'reason':'session_recording_limit_reached'})
            job={'event':{k:event.get(k) for k in ['id','phrase','label','source_ns','submission_ns','delivery','acknowledged','virtual_delivery']},
                 'targets':targets,'missing':missing,'clips':list(clips.values()),'bytes':size}
            self.jobs.put_nowait(job);self.queue_bytes+=size;self.seen.add(identity);self.seen_order.append(identity)
            while len(self.seen_order)>8192:self.seen.discard(self.seen_order.popleft())
            self.metrics['bookmarks_requested']+=1
            return {'state':'queued','event_id':identity,'camera_targets':len(targets),'missing':missing}
        finally:self.lock.release()

    def _save(self,job):
        if job['bytes'] and not self.space_available(job['bytes']):raise OSError('free-space reserve prevents camera evidence write')
        clips=[]
        for clip in job['clips']:
            final=self.folder/clip['file'];temporary=final.with_suffix('.fcv.partial')
            digest=hashlib.sha256()
            with temporary.open('xb') as stream:
                for _,header,payload in clip['messages']:
                    stream.write(header);stream.write(payload);digest.update(header);digest.update(payload)
            if final.exists():raise FileExistsError('Camera evidence already exists')
            temporary.replace(final)
            clips.append({'file':clip['file'],'camera':clip['camera'],'bytes':clip['bytes'],'sha256':digest.hexdigest()})
            self.metrics['written_bytes']+=clip['bytes']
        return {'event':job['event'],'targets':job['targets'],'missing':job['missing'],'clips':clips,
                'state':'saved' if clips else 'unavailable','native_encoded_evidence':True,
                'image_export_required':True,'audio_heard_verified':False}

    def _run(self):
        try:
            while not self.stop_event.is_set() or not self.jobs.empty():
                try:job=self.jobs.get(timeout=.1)
                except queue.Empty:continue
                try:
                    try:result=self._save(job)
                    except (OSError,ValueError,TypeError) as error:
                        self._error(error);result={'event':job['event'],'state':'write_failed','error':str(error),'targets':job['targets'],'missing':job['missing'],'clips':[]}
                    self.index.write(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
                    self.metrics['bookmarks_finished']+=1
                finally:
                    with self.lock:self.queue_bytes-=job['bytes']
                    self.jobs.task_done()
        except Exception as error:self._error(error)
        finally:
            self.index.close()
            try:self.pending.unlink(missing_ok=True)
            except OSError as error:self._error(error)

    def snapshot(self):
        with self.lock:return {'metrics':dict(self.metrics),'buffer_bytes':self.buffer_bytes,'queued_bytes':self.queue_bytes,
            'pending_bookmarks':self.jobs.qsize(),'writer_alive':bool(self.thread and self.thread.is_alive()),
            'last_error':self.last_error,'mode':'native_advisory_clips'}

    def stop(self):
        with self.lock:
            self.stop_event.set();self.chains.clear();self.current.clear();self.buffer_bytes=0
        if self.thread:self.thread.join(timeout=3)
        else:
            self.index.close();self.pending.unlink(missing_ok=True)
        return not self.thread or not self.thread.is_alive()

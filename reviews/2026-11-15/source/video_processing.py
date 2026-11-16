"""Bounded camera decoding/inference workers, independent of lidar reception.

Decode every available access unit; inference consumes only the newest decoded
image per camera. Compressed packet loss waits for the next keyframe. OpenCV
and PyAV are deliberately isolated in different spawned processes.

This component emits observations, never speech or vehicle commands. Runtime
adoption is gated separately by encounter and complete-device measurements.
"""
import copy
import collections
import concurrent.futures
import json
import math
import multiprocessing as mp
import queue
import threading
import time
import uuid

from awareness import MAX_AGE_NS
from video_transport import MAX_META
import runtime_clock as clock

CAMERAS={'roadEncodeData':'narrow','wideRoadEncodeData':'wide'}

class InferenceCadence:
    """One aggregate start-to-start budget; never catch up a delayed queue."""
    def __init__(self,hz=4.):
        if isinstance(hz,bool) or not isinstance(hz,(int,float)) or not math.isfinite(hz) or not 0<hz<=4:
            raise ValueError('Aggregate camera inference rate must be in (0,4] Hz')
        self.interval_ns=round(1e9/hz);self.next_ns=0
    def delay_s(self,now_ns):return max(0.,(self.next_ns-now_ns)/1e9)
    def started(self,now_ns):self.next_ns=now_ns+self.interval_ns



class LatestFrame:
    """One atomic shared image + identity, avoiding stale multi-MB pipe copies."""
    def __init__(self,ctx,width=1928,height=1208):
        self.width=width;self.height=height
        self.pixels=ctx.RawArray('B',width*height*3)
        self.identity=ctx.RawArray('q',3)  # sequence, decode duration, metadata bytes
        self.metadata=ctx.RawArray('B',MAX_META)
        self.lock=ctx.Lock()

    def write(self,frame):
        import numpy as np
        image=frame['bgr'];meta=frame['metadata']
        if image.shape!=(self.height,self.width,3) or image.dtype!=np.uint8:
            raise ValueError('Unexpected native image shape/type')
        encoded=json.dumps(meta,separators=(',',':'),allow_nan=False).encode('utf-8')
        if not 0<len(encoded)<=MAX_META:raise ValueError('Decoded frame metadata exceeds bound')
        if not self.lock.acquire(block=False):return False
        try:
            np.copyto(np.frombuffer(self.pixels,dtype=np.uint8).reshape(image.shape),image)
            self.metadata[:len(encoded)]=encoded
            self.identity[0]+=1
            self.identity[1:]=[int(frame['decode_ms']*1e6),len(encoded)]
            return True
        finally:self.lock.release()

    def take(self,after_sequence):
        import numpy as np
        if not self.lock.acquire(block=False):return None
        try:
            sequence,decode_ns,size=self.identity[:]
            if sequence<=after_sequence:return None
            if not 0<size<=MAX_META:raise ValueError('Invalid shared frame metadata length')
            metadata=json.loads(bytes(self.metadata[:size]))
            pixels=np.frombuffer(self.pixels,dtype=np.uint8).reshape(self.height,self.width,3).copy()
            return sequence,{'bgr':pixels,'decode_ms':decode_ns/1e6,
                            'metadata':metadata}
        finally:self.lock.release()


def put_latest(destination,item):
    """Replace a pending image/result, never block a recording/receiver thread."""
    try:
        destination.put_nowait(item)
        return True
    except queue.Full:
        try:destination.get_nowait()
        except queue.Empty:return False  # a multiprocessing feeder is in flight
        try:
            destination.put_nowait(item)
            return True
        except queue.Full:return False


def source_fresh(meta,now_ns):
    stamp=meta.get('source_ns')
    return type(stamp) is int and 0<stamp<=now_ns and now_ns-stamp<=MAX_AGE_NS


def _decoder(encoded,decoded,output,stop,camera,hw_device=None):
    # No OpenCV or model imports in this process.
    counts=collections.Counter();last_report=0.
    try:
        put_latest(output,{'kind':'decoder_starting','camera':camera})
        from video_decoder import VideoDecoder
        decoder=VideoDecoder(hw_device=hw_device)
        put_latest(output,{'kind':'decoder_ready','camera':camera})
        while not stop.is_set():
            try:meta,payload=encoded.get(timeout=.1)
            except queue.Empty:
                now=time.monotonic()
                if now-last_report>=1:
                    put_latest(output,{'kind':'decoder_status','camera':camera,'counts':dict(counts),'hardware':dict(decoder.hardware_active)})
                    last_report=now
                continue
            counts['packets']+=1
            if not source_fresh(meta,clock.monotonic_ns()):
                counts['expired_encoded_packets']+=1
                continue
            frames,reason=decoder.decode(meta,payload)
            if reason:counts[reason]+=1
            for frame in frames:
                counts['decoded_frames']+=1
                if not source_fresh(frame['metadata'],clock.monotonic_ns()):
                    counts['expired_decoded_frames']+=1
                    continue
                cam=CAMERAS[frame['metadata']['service']]
                if cam!=camera:raise ValueError('Packet was routed to the wrong camera worker')
                if not decoded.write(frame):counts['decoded_handoff_drops']+=1
            now=time.monotonic()
            if now-last_report>=1:
                put_latest(output,{'kind':'decoder_status','camera':camera,'counts':dict(counts),'hardware':dict(decoder.hardware_active)})
                last_report=now
    except Exception as error:
        put_latest(output,{'kind':'worker_error','worker':'decoder:'+camera,'error':type(error).__name__+': '+str(error)})
    finally:
        put_latest(output,{'kind':'decoder_final','camera':camera,'counts':dict(counts)})


def _inference(decoded,output,stop,model_config,paused=None,control_epoch=None):
    # No PyAV imports in this process. Model/runtime failure is observable.
    motion_pool=None;pending_motion=None;identity_pool=None;pending_identity=None
    try:
        # A pause requested before startup also postpones model allocation and
        # warmup; FCW should not pay an optional GPU startup cost while pressured.
        while paused is not None and paused.is_set():
            if stop.wait(.05):return
        import cv2
        from awareness_onnx import OnnxCameraDetector
        from awareness_vision import VisionTracks
        from camera_motion import CameraMotion
        from visual_identity import VisualIdentity
        from video_decoder import PIXEL_CONVERSION
        import numpy as np
        cadence=InferenceCadence(model_config.get('aggregate_inference_hz',4.))
        cv2.setNumThreads(1)
        detector=OnnxCameraDetector(model_config['weights'],model_config['sha256'],model_config['provider'])
        # Warm kernels before declaring readiness; these images never enter the
        # tracker or count as observations.
        for _ in range(2):
            while paused is not None and paused.is_set():
                if stop.wait(.05):return
            if stop.is_set():return
            detector.predict([np.zeros((1208,1928,3),dtype=np.uint8)])
        tracks={cam:VisionTracks(cam,fx=2648. if cam=='narrow' else 567.,cx=964.) for cam in ['narrow','wide']}
        motion={cam:CameraMotion() for cam in tracks}
        identity=VisualIdentity()
        identity_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='optional-camera-identity')
        motion_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='camera-image-motion')
        put_latest(output,{'kind':'inference_ready','provider':detector.provider,
                          'model_sha256':detector.sha256,'pixel_conversion':PIXEL_CONVERSION,
                          'aggregate_inference_hz':model_config.get('aggregate_inference_hz',4.),'precision':'float32',
                          'worker_epoch':model_config.get('worker_epoch')})
        counts=collections.Counter();turn=0;sequences={cam:0 for cam in decoded};was_paused=False
        while not stop.is_set():
            if paused is not None and paused.is_set():
                if not was_paused:
                    put_latest(output,{'kind':'inference_paused','at_ns':clock.monotonic_ns()})
                    was_paused=True
                stop.wait(.05);continue
            if was_paused:
                put_latest(output,{'kind':'inference_resumed','at_ns':clock.monotonic_ns()})
                was_paused=False
            loop_now=clock.monotonic_ns();delay=cadence.delay_s(loop_now)
            if delay>0:
                stop.wait(min(delay,.05));continue
            # Round-robin prevents an expensive or busier camera starving the other.
            cam=('narrow','wide')[turn];turn=1-turn
            item=decoded[cam].take(sequences[cam])
            if item is None:
                stop.wait(.002)
                continue
            sequences[cam],frame=item
            meta=frame['metadata'];began=loop_now
            if not source_fresh(meta,began):
                counts['expired_before_inference']+=1
                continue
            cadence.started(began)
            epoch=control_epoch.value if control_epoch is not None else 0
            put_latest(output,{'kind':'inference_started','at_ns':began,'camera':cam,
                               'frame_id':meta['frame_id'],'control_epoch':epoch})
            # Optical flow uses a CPU core while the neural model uses the GPU.
            # They see the same immutable image/source identity. No future frame
            # or result from a different camera can be used for association.
            future=None
            if pending_motion is None or pending_motion.done():
                future=motion_pool.submit(motion[cam].update,frame['bgr'],meta['source_ns'])
                pending_motion=future
            else:counts['image_motion_busy']+=1
            boxes=detector.predict([frame['bgr']])[0];model_finished=clock.monotonic_ns()
            try:
                if future is None:raise TimeoutError('Prior image motion still running; no queued motion job')
                remaining=max(0.,(meta['source_ns']+MAX_AGE_NS-model_finished)/1e9)
                image_motion=future.result(timeout=remaining)
            except Exception as error:
                image_motion={'valid':False,'source_ns':meta['source_ns'],
                              'reason':'image_motion_error','error':type(error).__name__+': '+str(error)}
                counts['image_motion_errors']+=1
            finished=clock.monotonic_ns()
            motion_finished=finished
            counts['inference_calls']+=1
            epoch_changed=control_epoch is not None and epoch!=control_epoch.value
            paused_result=paused is not None and paused.is_set()
            fresh=source_fresh(meta,finished) and not epoch_changed and not paused_result
            identity_ms=0.;identity_error=None
            if fresh:
                h,w=frame['bgr'].shape[:2]
                observations=tracks[cam].update(boxes,meta['source_ns'],finished,w,h,
                                                frame_id=meta['frame_id'],image_motion=image_motion)
                identity_started=clock.monotonic_ns()
                # Never let optional identity consume the primary observation's
                # remaining lifetime. One owned job, no queue and no second publish.
                margin_s=max(0.,(meta['source_ns']+MAX_AGE_NS-clock.monotonic_ns()-25_000_000)/1e9)
                if pending_identity is not None and pending_identity.done():
                    pending_identity=None
                if pending_identity is None and margin_s>.055:
                    def enrich(cam,obs,image,metadata):
                        try:return identity.update(cam,obs,image,metadata)
                        except Exception:
                            identity.reset();raise
                    pending_identity=identity_pool.submit(enrich,cam,copy.deepcopy(observations),frame['bgr'],dict(meta))
                    try:
                        observations=pending_identity.result(timeout=min(margin_s,.02))
                        pending_identity=None
                    except concurrent.futures.TimeoutError:
                        counts['identity_budget_exhausted']+=1
                    except Exception as error:
                        counts['visual_identity_errors']+=1;pending_identity=None
                        identity_error=type(error).__name__+': '+str(error)
                else:counts['identity_skipped_for_primary']+=1
                finished=clock.monotonic_ns()
                identity_ms=(finished-identity_started)/1e6
                epoch_changed=control_epoch is not None and epoch!=control_epoch.value
                paused_result=paused is not None and paused.is_set()
                fresh=source_fresh(meta,finished) and not epoch_changed and not paused_result
                if fresh:counts['fresh_results']+=1
                else:
                    observations=[];counts['expired_after_identity']+=1
            else:
                observations=[];counts['expired_after_inference']+=1
            row={'kind':'observations','camera':cam,'source_ns':meta['source_ns'],
                 'native_source_ns':meta.get('native_source_ns'),'frame_id':meta['frame_id'],
                 'stream_identity':{k:meta.get(k) for k in ('writer','segment_num','encode_id','frame_id')},
                 'source_interval_ns':meta.get('source_interval_ns'),'clock_bound_ns':meta.get('clock_bound_ns'),
                 'native_clock_domain':meta.get('native_clock_domain'),
                 'receipt_ns':meta.get('receipt_ns'),'finished_ns':finished,
                 'source_age_ms':(finished-meta['source_ns'])/1e6,
                 'inference_ms':(model_finished-began)/1e6,'decode_ms':frame['decode_ms'],
                 'inference_and_motion_ms':(motion_finished-began)/1e6,
                 'visual_identity_ms':identity_ms,'visual_identity_error':identity_error,
                 'processing_ms':(finished-began)/1e6,
                 'fresh':fresh,'observations':observations,'image_motion':image_motion,'counts':dict(counts)}
            row.update(control_epoch=epoch,worker_epoch=model_config.get('worker_epoch'),
                       paused_result=paused_result,control_epoch_changed=epoch_changed)
            if not put_latest(output,row):counts['output_backpressure']+=1
    except Exception as error:
        put_latest(output,{'kind':'worker_error','worker':'inference','error':type(error).__name__+': '+str(error)})
    finally:
        if motion_pool:motion_pool.shutdown(wait=False,cancel_futures=True)
        if identity_pool:identity_pool.shutdown(wait=False,cancel_futures=True)


class VideoProcessing:
    def __init__(self,model_config,on_result):
        ctx=mp.get_context('spawn')
        self.stop_event=ctx.Event();self.encoded={cam:ctx.Queue(maxsize=8) for cam in ['narrow','wide']}
        self.pause_event=ctx.Event();self.control_epoch=ctx.Value('q',0)
        self.worker_epoch=uuid.uuid4().hex;model_config={**model_config,'worker_epoch':self.worker_epoch}
        self.pause_reason=None;self.inference_ready=False;self.inflight_started_ns=None
        self.last_inference_finished_ns=None;self.last_health_state=None
        self.decoded={cam:LatestFrame(ctx) for cam in ['narrow','wide']}
        self.output=ctx.Queue(maxsize=64);self.on_result=on_result
        self.counts=collections.Counter();self.thread=None;self.last_error=None
        self.processes=[ctx.Process(target=_decoder,
            args=(self.encoded[cam],self.decoded[cam],self.output,self.stop_event,cam,model_config.get('decoder_hw_device')),
            name='awareness-decode-'+cam) for cam in ['narrow','wide']]
        self.processes.append(ctx.Process(target=_inference,args=(self.decoded,self.output,self.stop_event,dict(model_config),self.pause_event,self.control_epoch),name='awareness-inference'))

    def start(self):
        try:
            for process in self.processes:process.start()
            self.thread=threading.Thread(target=self._receive,daemon=True,name='awareness-results')
            self.thread.start()
        except Exception:
            self.stop()
            raise

    def on_packet(self,meta,payload):
        if self.stop_event.is_set() or meta.get('service') not in CAMERAS:return False
        if not source_fresh(meta,clock.monotonic_ns()):
            self.counts['expired_at_enqueue']+=1;return False
        try:
            self.encoded[CAMERAS[meta['service']]].put_nowait((meta,payload));self.counts['enqueued']+=1;return True
        except queue.Full:
            # Do not conflate arbitrary compressed packets; continuity will reset
            # at the decoder when a sequence gap is seen.
            self.counts['encoded_queue_full']+=1;return False

    def set_camera_paused(self,paused,reason=None):
        """Pause optional inference only; decoders retain the latest fresh frame.

        No tracker or external announcement history is reset. An epoch change
        invalidates in-flight/queued results even across a quick pause/resume.
        """
        if type(paused) is not bool:raise ValueError('Camera pause state must be boolean')
        with self.control_epoch.get_lock():
            if self.pause_event.is_set()==paused:return False
            self.control_epoch.value+=1
            if paused:self.pause_event.set()
            else:self.pause_event.clear()
            self.pause_reason=str(reason)[:200] if paused and reason is not None else None
        return True

    def snapshot(self):
        now=clock.monotonic_ns();p=self.processes[-1]
        dead=p.pid is not None and not p.is_alive() and not self.stop_event.is_set()
        stalled=(self.inflight_started_ns is not None and now-self.inflight_started_ns>1_000_000_000)
        state=('stopped' if self.stop_event.is_set() else 'failed' if dead else 'stalled' if stalled
               else 'paused' if self.pause_event.is_set() else 'running' if self.inference_ready else 'starting')
        return {'state':state,'paused':self.pause_event.is_set(),'pause_reason':self.pause_reason,
                'control_epoch':self.control_epoch.value,'worker_epoch':self.worker_epoch,
                'inference_pid':p.pid,'inference_exitcode':p.exitcode,'inflight_started_ns':self.inflight_started_ns,
                'last_inference_finished_ns':self.last_inference_finished_ns,'last_error':self.last_error,
                'automatic_restart':False,'counts':dict(self.counts)}

    def _health_change(self):
        status=self.snapshot()
        if status['state']==self.last_health_state:return
        self.last_health_state=status['state']
        try:self.on_result({'kind':'camera_health',**status,'at_ns':clock.monotonic_ns()})
        except Exception as error:
            self.counts['consumer_errors']+=1;self.last_error=type(error).__name__+': '+str(error)

    def _receive(self):
        while not self.stop_event.is_set():
            self._health_change()
            try:result=self.output.get(timeout=.1)
            except queue.Empty:continue
            if result.get('kind')=='inference_ready':self.inference_ready=True
            elif result.get('kind')=='inference_started':self.inflight_started_ns=result.get('at_ns')
            elif result.get('kind')=='worker_error':self.last_error=result.get('error')
            if result.get('kind')=='observations':
                self.inflight_started_ns=None;self.last_inference_finished_ns=result.get('finished_ns')
                received=clock.monotonic_ns()
                result={**result,'received_at_consumer_ns':received,'consumer_age_ms':(received-result['source_ns'])/1e6}
                if self.pause_event.is_set() or result.get('control_epoch')!=self.control_epoch.value:
                    result={**result,'fresh':False,'observations':[],'suppressed_by_camera_pause':True}
                elif not source_fresh(result,received):
                    result={**result,'fresh':False,'observations':[],'expired_at_consumer':True}
            try:self.on_result(result)
            except Exception as error:
                self.counts['consumer_errors']+=1
                # The caller can inspect this; never silently terminate the reader.
                self.last_error=type(error).__name__+': '+str(error)

    def stop(self):
        self.stop_event.set()
        for process in self.processes:
            if process.pid is None:continue
            process.join(timeout=2)
            if process.is_alive():process.terminate();process.join(timeout=2)
        if self.thread:self.thread.join(timeout=2)
        for q in [*self.encoded.values(),self.output]:
            q.cancel_join_thread();q.close()

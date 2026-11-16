"""Isolated inference with private bounded IPC; no sensor sockets or vehicle control."""
import os
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import pickle
import threading
import time

CAPACITY=16*1024*1024
RESPONSE_SOFT_S=.75
RESPONSE_HARD_S=1.5  # liveness only; warning freshness remains 150 ms

class WorkerFailure(RuntimeError):pass
class ResponseTimeout(WorkerFailure):pass


class _KillOnCloseJob:
    """Own the Windows inference child without throttling its CPU."""
    def __init__(self,process):
        self.handle=None
        if os.name!='nt':return
        from ctypes import wintypes as w
        S=ctypes.c_size_t
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',w.DWORD),
                      ('min_ws',S),('max_ws',S),('active_limit',w.DWORD),('affinity',S),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_ulonglong) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',S),('job_memory',S),('peak_process',S),('peak_job',S)]
        k=ctypes.WinDLL('kernel32',use_last_error=True);self.kernel=k
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
        k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];k.SetInformationJobObject.restype=w.BOOL
        k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
        k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
        self.handle=k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits=Extended();limits.basic.flags=0x2000|0x200|0x8
            limits.basic.active_limit=2;limits.job_memory=2*1024**3
            if not k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
            if not k.AssignProcessToJobObject(self.handle,w.HANDLE(process.sentinel)):raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:self.close();raise
    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle);self.handle=None


def _serve(conn,request_name,response_name,engine,live):
    request=shared_memory.SharedMemory(name=request_name)
    response=shared_memory.SharedMemory(name=response_name)
    try:
        conn.send(('ready',time.perf_counter_ns()))
        while True:
            command,seq,size=conn.recv()
            if command=='close':break
            if not isinstance(size,int) or not 0<size<=CAPACITY:raise ValueError('Invalid private request size')
            payload=pickle.loads(bytes(request.buf[:size]))
            try:
                if command=='update':
                    points,context,quality,relative,deadline,inputs=payload
                    if live:
                        quality=dict(quality);receipt=quality.get('last_receipt_ns');now=time.perf_counter_ns()
                        age=(now-receipt)/1e9 if type(receipt) is int and 0<receipt<=now else 1.
                        quality['arrival_age_s']=max(quality.get('arrival_age_s',0.),age)
                    start=time.perf_counter_ns();cpu_start=time.process_time_ns()
                    result=engine.update(points,context,quality,relative,processing_deadline_ns=deadline,fusion_inputs=inputs)
                    value=('ok',result,{'child_engine_ms':(time.perf_counter_ns()-start)/1e6,'child_cpu_ms':(time.process_time_ns()-cpu_start)/1e6})
                elif command=='reject':
                    engine.reject_delivery(*payload);value=('ok',None,{})
                else:raise ValueError('Unknown private command')
            except Exception as e:
                value=('error',type(e).__name__+': '+str(e),{})
            encoded=pickle.dumps(value,protocol=5)
            if len(encoded)>CAPACITY:raise ValueError('Private response exceeds bound')
            response.buf[:len(encoded)]=encoded
            conn.send(('result',seq,len(encoded)))
    except (EOFError,BrokenPipeError,OSError):pass
    finally:
        conn.close();request.close();response.close()


class EngineProcess:
    def __init__(self,engine,*,live=False,stop_event=None):
        self.mount=engine.mount;self.stop_event=stop_event
        self._lock=threading.RLock();self._closed=False;self._failed=False;self._ready=False;self._seq=0
        self.last_metrics={};self._job=None;self._started=time.perf_counter_ns()
        self._request=shared_memory.SharedMemory(create=True,size=CAPACITY)
        self._response=shared_memory.SharedMemory(create=True,size=CAPACITY)
        ctx=mp.get_context('spawn');self._conn,child=ctx.Pipe()
        self.process=ctx.Process(target=_serve,args=(child,self._request.name,self._response.name,engine,live),daemon=True)
        try:
            self.process.start()
            self._job=_KillOnCloseJob(self.process)
        except BaseException:
            if self.process.is_alive():self.process.terminate();self.process.join(1)
            if self._job:self._job.close()
            child.close();self._conn.close();self._release();raise
        child.close()

    def _wait(self,timeout):
        end=time.perf_counter()+timeout
        while time.perf_counter()<end:
            if self._closed or (self.stop_event is not None and self.stop_event.is_set()):raise WorkerFailure('Inference stopped')
            if self._conn.poll(min(.02,max(0,end-time.perf_counter()))):return self._conn.recv()
            if not self.process.is_alive():raise WorkerFailure('Inference child exited')
        raise ResponseTimeout('Inference child response timed out')

    def wait_ready(self):
        with self._lock:
            if self._ready:return
            msg=self._wait(8)
            now=time.perf_counter_ns()
            if len(msg)!=2 or msg[0]!='ready' or not self._started<=msg[1]<=now:raise WorkerFailure('Inference clock/handshake mismatch')
            self._ready=True;self.startup_ms=(now-self._started)/1e6

    def _call(self,command,payload):
        with self._lock:
            if self._failed or self._closed:raise WorkerFailure('Inference worker unavailable')
            try:
                self.wait_ready();self._seq+=1;seq=self._seq
                start=time.perf_counter_ns();encoded=pickle.dumps(payload,protocol=5)
                if len(encoded)>CAPACITY:raise WorkerFailure('Inference request exceeds bound')
                self._request.buf[:len(encoded)]=encoded
                self._conn.send((command,seq,len(encoded)))
                sent=time.perf_counter();late=False
                try:reply=self._wait(RESPONSE_SOFT_S)
                except ResponseTimeout:
                    if command!='update':raise
                    late=True
                    # Drain this exact request; never resubmit it or overwrite its shared buffer.
                    # Stop/death checks continue every 20 ms. Freshness checks below reject late speech.
                    try:reply=self._wait(max(0.,RESPONSE_HARD_S-(time.perf_counter()-sent)))
                    except ResponseTimeout as e:
                        raise ResponseTimeout('Inference child response timed out; command=%s seq=%d pid=%d alive=%s wait_ms=%.1f' %
                            (command,seq,self.process.pid,self.process.is_alive(),1000*(time.perf_counter()-sent))) from e
                if len(reply)!=3 or reply[0]!='result' or reply[1]!=seq or not 0<reply[2]<=CAPACITY:raise WorkerFailure('Inference response identity mismatch')
                status,value,metrics=pickle.loads(bytes(self._response.buf[:reply[2]]))
                if command=='update':self.last_metrics={**metrics,'roundtrip_ms':(time.perf_counter_ns()-start)/1e6,'late_response_recovered':late,'request_bytes':len(encoded),'response_bytes':reply[2]}
                if status!='ok':raise WorkerFailure(value)
                return value
            except BaseException as error:
                self._failed=True;self.close()
                if isinstance(error,Exception) and not isinstance(error,WorkerFailure):
                    raise WorkerFailure(type(error).__name__+': '+str(error)[:1000]) from error
                raise

    def update(self,points,context,quality=None,relative_times=None,processing_deadline_ns=None,fusion_inputs=None):
        with self._lock:
            result=self._call('update',(points,context,quality or {},relative_times,processing_deadline_ns,fusion_inputs))
            # Serialization/return transport is part of the caller's same deadline.
            # Late output never reaches speech or retains an undelivered cooldown.
            if processing_deadline_ns is not None and time.perf_counter_ns()>processing_deadline_ns:
                for alert in result.get('alerts',[]):self.reject_delivery(alert['track_id'],result['at_ns'])
                result['alerts']=[];result['status']='not_ready'
                result['reasons']=list(dict.fromkeys(result.get('reasons',[])+['inference exceeded 150 ms']))
            return result

    def reject_delivery(self,track_id,at_ns):return self._call('reject',(track_id,at_ns))

    def _release(self):
        for shm in (self._request,self._response):
            try:shm.close()
            except Exception:pass
            try:shm.unlink()
            except FileNotFoundError:pass

    def close(self):
        if self._closed:return
        self._closed=True
        try:
            if self.process.is_alive():
                try:self._conn.send(('close',0,0))
                except (OSError,BrokenPipeError):pass
                self.process.join(.25)
                if self.process.is_alive():self.process.terminate();self.process.join(1)
        finally:
            if self._job:self._job.close()
            if self.process.is_alive():self.process.join(1)
            self._conn.close();self._release()
        if self.process.is_alive():raise WorkerFailure('Inference child did not exit')

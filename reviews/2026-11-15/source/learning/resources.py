"""Windows-enforced budgets for a disposable post-drive worker process tree.

The worker is created suspended, assigned to a private Job Object, and only then
resumed. Its descendants therefore inherit the same memory/CPU limits. Failure
to establish limits leaves no unbounded worker running. Closing the owner kills
this job's descendants only, including after application shutdown or a crash.

Win32 definitions: learn.microsoft.com/windows/win32/api/winnt/
ns-winnt-jobobject_extended_limit_information and
ns-winnt-jobobject_cpu_rate_control_information. Tree quiescence uses
JOBOBJECT_BASIC_ACCOUNTING_INFORMATION.ActiveProcesses, and inheritance uses
the documented STARTUPINFOEX / PROC_THREAD_ATTRIBUTE_HANDLE_LIST API.

stop()/close() may wait up to timeout_s and must be called outside application
capture/UI locks. They raise if tree quiescence is not established. A signaled
root process alone is not evidence that descendants have stopped.
"""
import ctypes
import math
import os
from pathlib import Path
import re
import subprocess
import runtime_clock as time
import uuid


class ResourceControlUnavailable(RuntimeError):
    pass


def worker_environment():
    return dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')


def current_job_limits(job_name=None):
    """Query this process's actual Windows job budget; no caller assertion."""
    if os.name!='nt':raise ResourceControlUnavailable('No supported enforced worker job on this host')
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',w.DWORD),
                  ('minimum_ws',ctypes.c_size_t),('maximum_ws',ctypes.c_size_t),('processes',w.DWORD),
                  ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(ctypes.Structure):_fields_=[('values',ctypes.c_ulonglong*6)]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                  ('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]
    class CPU(ctypes.Structure):_fields_=[('flags',w.DWORD),('rate',w.DWORD)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenJobObjectW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];kernel.OpenJobObjectW.restype=w.HANDLE
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)];kernel.IsProcessInJob.restype=w.BOOL
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype=w.BOOL
    name=job_name or os.environ.get('ODYSSEY_LEARNING_JOB_NAME','')
    if not re.fullmatch(r'Local\\OdysseyFCW-learning-[0-9a-f]{32}',name):
        raise ResourceControlUnavailable('Expected owned worker job name unavailable')
    handle=kernel.OpenJobObjectW(4,False,name)  # JOB_OBJECT_QUERY only, not inheritable
    if not handle:raise ResourceControlUnavailable('Cannot query expected owned worker job')
    limits=Extended();cpu=CPU()
    try:
        member=w.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),handle,ctypes.byref(member)) or not member.value:
            raise ResourceControlUnavailable('This process is not a member of the expected worker job')
        for kind,value in ((9,limits),(15,cpu)):
            if not kernel.QueryInformationJobObject(handle,kind,ctypes.byref(value),ctypes.sizeof(value),None):
                raise ResourceControlUnavailable('Current process has no verifiable worker job budget')
    finally:kernel.CloseHandle(handle)
    return {'job_memory_limit_bytes':int(limits.job_memory),'job_memory_limit_enforced':bool(limits.basic.flags&0x200),
            'cpu_rate':int(cpu.rate),'cpu_hard_cap':bool(cpu.flags&4),'cpu_enabled':bool(cpu.flags&1),
            'active_process_limit':int(limits.basic.processes),'kill_on_owner_close':bool(limits.basic.flags&0x2000)}


class BoundedProcess:
    """Owned process tree; methods never locate/terminate unrelated processes."""
    def __init__(self,command,log_path,*,cwd=None,memory_mb=1024,cpu_rate=None,environment=None,job_name=None):
        if os.name!='nt':
            raise ResourceControlUnavailable('Enforced learning-worker budgets are not implemented on this host')
        if type(memory_mb) is not int or not 128<=memory_mb<=2048:
            raise ValueError('Worker memory budget must be 128–2048 MiB')
        cpu_rate=cpu_rate if cpu_rate is not None else min(2000,max(100,10000//max(1,os.cpu_count() or 1)))
        if type(cpu_rate) is not int or not 100<=cpu_rate<=3000:
            raise ValueError('CPU budget must be 1–30 percent of total system capacity')
        if not command or not Path(command[0]).is_file():raise ValueError('Worker executable does not exist')
        self.process=None;self.job=None;self.pid=None;self.closed=False;self._output=None;self._input=None
        self._exit_code=None;self.quiescence_confirmed=False;self.explicit_handle_allowlist=False
        self._member_handles={};self._members_proven=True
        self.memory_bytes=memory_mb*1024**2;self.cpu_rate=cpu_rate
        self.job_name=job_name or 'Local\\OdysseyFCW-learning-'+uuid.uuid4().hex
        if not re.fullmatch(r'Local\\OdysseyFCW-learning-[0-9a-f]{32}',self.job_name):raise ValueError('Invalid owned worker job name')
        self.command=list(map(str,command));self.log_path=Path(log_path)
        from ctypes import wintypes as w
        import msvcrt
        DWORD=w.DWORD;HANDLE=w.HANDLE;SIZE=ctypes.c_size_t;ULONGLONG=ctypes.c_ulonglong
        class Basic(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
                      ('LimitFlags',DWORD),('MinimumWorkingSetSize',SIZE),('MaximumWorkingSetSize',SIZE),
                      ('ActiveProcessLimit',DWORD),('Affinity',SIZE),('PriorityClass',DWORD),('SchedulingClass',DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(name,ULONGLONG) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                                  'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',SIZE),
                      ('JobMemoryLimit',SIZE),('PeakProcessMemoryUsed',SIZE),('PeakJobMemoryUsed',SIZE)]
        class CPU(ctypes.Structure):
            _fields_=[('ControlFlags',DWORD),('CpuRate',DWORD)]
        class Accounting(ctypes.Structure):
            _fields_=[('TotalUserTime',ctypes.c_longlong),('TotalKernelTime',ctypes.c_longlong),
                      ('ThisPeriodTotalUserTime',ctypes.c_longlong),('ThisPeriodTotalKernelTime',ctypes.c_longlong),
                      ('TotalPageFaultCount',DWORD),('TotalProcesses',DWORD),
                      ('ActiveProcesses',DWORD),('TotalTerminatedProcesses',DWORD)]
        class Startup(ctypes.Structure):
            _fields_=[('cb',DWORD),('lpReserved',w.LPWSTR),('lpDesktop',w.LPWSTR),('lpTitle',w.LPWSTR),
                      ('dwX',DWORD),('dwY',DWORD),('dwXSize',DWORD),('dwYSize',DWORD),('dwXCountChars',DWORD),
                      ('dwYCountChars',DWORD),('dwFillAttribute',DWORD),('dwFlags',DWORD),('wShowWindow',w.WORD),
                      ('cbReserved2',w.WORD),('lpReserved2',ctypes.POINTER(ctypes.c_ubyte)),
                      ('hStdInput',HANDLE),('hStdOutput',HANDLE),('hStdError',HANDLE)]
        class StartupEx(ctypes.Structure):
            _fields_=[('StartupInfo',Startup),('lpAttributeList',ctypes.c_void_p)]
        class ProcessInfo(ctypes.Structure):
            _fields_=[('hProcess',HANDLE),('hThread',HANDLE),('dwProcessId',DWORD),('dwThreadId',DWORD)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);self.kernel=kernel;self.Extended=Extended;self.CPU=CPU;self.Accounting=Accounting
        kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];kernel.CreateJobObjectW.restype=HANDLE
        kernel.SetInformationJobObject.argtypes=[HANDLE,ctypes.c_int,ctypes.c_void_p,DWORD];kernel.SetInformationJobObject.restype=w.BOOL
        kernel.QueryInformationJobObject.argtypes=[HANDLE,ctypes.c_int,ctypes.c_void_p,DWORD,ctypes.c_void_p];kernel.QueryInformationJobObject.restype=w.BOOL
        kernel.AssignProcessToJobObject.argtypes=[HANDLE,HANDLE];kernel.AssignProcessToJobObject.restype=w.BOOL
        kernel.CloseHandle.argtypes=[HANDLE];kernel.CloseHandle.restype=w.BOOL
        kernel.SetHandleInformation.argtypes=[HANDLE,DWORD,DWORD];kernel.SetHandleInformation.restype=w.BOOL
        kernel.CreateProcessW.argtypes=[w.LPCWSTR,w.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,w.BOOL,DWORD,ctypes.c_void_p,w.LPCWSTR,ctypes.POINTER(Startup),ctypes.POINTER(ProcessInfo)]
        kernel.CreateProcessW.restype=w.BOOL
        kernel.ResumeThread.argtypes=[HANDLE];kernel.ResumeThread.restype=DWORD
        kernel.WaitForSingleObject.argtypes=[HANDLE,DWORD];kernel.WaitForSingleObject.restype=DWORD
        kernel.GetExitCodeProcess.argtypes=[HANDLE,ctypes.POINTER(DWORD)];kernel.GetExitCodeProcess.restype=w.BOOL
        kernel.TerminateJobObject.argtypes=[HANDLE,w.UINT];kernel.TerminateJobObject.restype=w.BOOL
        kernel.TerminateProcess.argtypes=[HANDLE,w.UINT];kernel.TerminateProcess.restype=w.BOOL
        kernel.OpenProcess.argtypes=[DWORD,w.BOOL,DWORD];kernel.OpenProcess.restype=HANDLE
        kernel.IsProcessInJob.argtypes=[HANDLE,HANDLE,ctypes.POINTER(w.BOOL)];kernel.IsProcessInJob.restype=w.BOOL
        kernel.InitializeProcThreadAttributeList.argtypes=[ctypes.c_void_p,DWORD,DWORD,ctypes.POINTER(SIZE)]
        kernel.InitializeProcThreadAttributeList.restype=w.BOOL
        kernel.UpdateProcThreadAttribute.argtypes=[ctypes.c_void_p,DWORD,SIZE,ctypes.c_void_p,SIZE,ctypes.c_void_p,ctypes.POINTER(SIZE)]
        kernel.UpdateProcThreadAttribute.restype=w.BOOL
        kernel.DeleteProcThreadAttributeList.argtypes=[ctypes.c_void_p]
        kernel.DeleteProcThreadAttributeList.restype=None
        def checked(ok,label):
            if not ok:raise ResourceControlUnavailable(label+': '+str(ctypes.WinError(ctypes.get_last_error())))
        self._checked=checked
        thread=None;assigned=False
        try:
            ctypes.set_last_error(0)
            self.job=kernel.CreateJobObjectW(None,self.job_name);checked(self.job,'Create learning job')
            if ctypes.get_last_error()==183:
                kernel.CloseHandle(self.job);self.job=None
                raise ResourceControlUnavailable('Refusing an already-existing worker job identity')
            limits=Extended()
            limits.BasicLimitInformation.LimitFlags=0x2000|0x200|0x8|0x20  # kill-on-close, job memory, process count, priority
            limits.BasicLimitInformation.ActiveProcessLimit=4
            limits.BasicLimitInformation.PriorityClass=0x4000  # below normal
            limits.JobMemoryLimit=self.memory_bytes
            checked(kernel.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)),'Set worker memory/process limits')
            cpu=CPU(0x1|0x4,cpu_rate)  # enable + hard cap
            checked(kernel.SetInformationJobObject(self.job,15,ctypes.byref(cpu),ctypes.sizeof(cpu)),'Set worker CPU cap')
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            self._output=self.log_path.open('ab',buffering=0);self._input=open(os.devnull,'rb',buffering=0)
            output_handle=msvcrt.get_osfhandle(self._output.fileno());input_handle=msvcrt.get_osfhandle(self._input.fileno())
            checked(kernel.SetHandleInformation(output_handle,1,1),'Mark worker output handle')
            checked(kernel.SetHandleInformation(input_handle,1,1),'Mark worker input handle')
            startup=StartupEx();startup.StartupInfo.cb=ctypes.sizeof(startup);startup.StartupInfo.dwFlags=0x100
            startup.StartupInfo.hStdInput=input_handle;startup.StartupInfo.hStdOutput=output_handle;startup.StartupInfo.hStdError=output_handle
            info=ProcessInfo();cmd=ctypes.create_unicode_buffer(subprocess.list2cmdline(self.command))
            env=worker_environment();env.update(environment or {})
            env['ODYSSEY_LEARNING_JOB_NAME']=self.job_name
            # Thread constraints are binding even if caller provides other environment values.
            for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            block=ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k,v in sorted(env.items(),key=lambda item:item[0].upper()))+'\0\0')
            attribute_list=None
            try:
                # The sizing call is documented to fail with insufficient buffer.
                size=SIZE();ctypes.set_last_error(0)
                kernel.InitializeProcThreadAttributeList(None,1,0,ctypes.byref(size))
                if ctypes.get_last_error()!=122 or not 0<size.value<=1024*1024:
                    raise ResourceControlUnavailable('Could not size the worker handle allowlist')
                attribute_buffer=ctypes.create_string_buffer(size.value)
                checked(kernel.InitializeProcThreadAttributeList(attribute_buffer,1,0,ctypes.byref(size)),'Initialize worker handle allowlist')
                attribute_list=ctypes.cast(attribute_buffer,ctypes.c_void_p)
                inherited=(HANDLE*2)(input_handle,output_handle)
                checked(kernel.UpdateProcThreadAttribute(attribute_list,0,0x00020002,inherited,ctypes.sizeof(inherited),None,None),
                        'Set explicit worker stdin/stdout handle allowlist')
                startup.lpAttributeList=attribute_list
                checked(kernel.CreateProcessW(self.command[0],cmd,None,None,True,0x4|0x08000000|0x400|0x4000|0x00080000,
                                               block,str(cwd) if cwd else None,ctypes.cast(ctypes.byref(startup),ctypes.POINTER(Startup)),ctypes.byref(info)),
                        'Create suspended worker')
                self.explicit_handle_allowlist=True
            finally:
                if attribute_list:kernel.DeleteProcThreadAttributeList(attribute_list)
                kernel.SetHandleInformation(output_handle,1,0);kernel.SetHandleInformation(input_handle,1,0)
            self.process=info.hProcess;thread=info.hThread;self.pid=int(info.dwProcessId)
            checked(kernel.AssignProcessToJobObject(self.job,self.process),'Assign suspended worker to budget')
            assigned=True
            if kernel.ResumeThread(thread)==0xffffffff:raise ResourceControlUnavailable('Could not resume budgeted worker')
            kernel.CloseHandle(thread);thread=None
            self.started=time.monotonic()
        except Exception:
            if self.process:
                if assigned:kernel.TerminateJobObject(self.job,75)
                else:kernel.TerminateProcess(self.process,75)
                kernel.WaitForSingleObject(self.process,3000)
            if thread:kernel.CloseHandle(thread)
            self.close()
            raise

    def poll(self):
        if not self.process:return self._exit_code
        wait=self.kernel.WaitForSingleObject(self.process,0)
        if wait==258:return None
        self._checked(wait==0,'Inspect worker process')
        value=ctypes.c_ulong()
        self._checked(self.kernel.GetExitCodeProcess(self.process,ctypes.byref(value)),'Read worker exit status')
        self._exit_code=int(value.value)
        return self._exit_code

    def wait(self,timeout_s=5):
        if not self.process:return self._exit_code
        wait=self.kernel.WaitForSingleObject(self.process,int(max(0,timeout_s)*1000))
        if wait==258:return None
        return self.poll()

    def limits(self):
        limits=self.Extended();cpu=self.CPU()
        self._checked(self.kernel.QueryInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits),None),'Read worker memory limits')
        self._checked(self.kernel.QueryInformationJobObject(self.job,15,ctypes.byref(cpu),ctypes.sizeof(cpu),None),'Read worker CPU limits')
        return {'job_memory_limit_bytes':int(limits.JobMemoryLimit),'peak_job_memory_bytes':int(limits.PeakJobMemoryUsed),
                'cpu_rate':int(cpu.CpuRate),'cpu_hard_cap':bool(cpu.ControlFlags&4),
                'cpu_budget_scope':'aggregate_system_capacity',
                'active_process_limit':int(limits.BasicLimitInformation.ActiveProcessLimit),'kill_on_owner_close':bool(limits.BasicLimitInformation.LimitFlags&0x2000),
                'explicit_handle_allowlist':self.explicit_handle_allowlist}

    def active_processes(self):
        """Authoritative active count for the whole owned Job Object tree."""
        if self.closed and self.quiescence_confirmed:return 0
        if not self.job:raise ResourceControlUnavailable('Worker job handle unavailable; quiescence is unknown')
        info=self.Accounting()
        self._checked(self.kernel.QueryInformationJobObject(self.job,1,ctypes.byref(info),ctypes.sizeof(info),None),
                      'Read worker-tree active-process count')
        return int(info.ActiveProcesses)

    def _retain_tree_handles(self):
        """Stop new associations, then retain only verified current job members.

        TerminateJobObject can set ActiveProcesses to zero before every process
        object becomes signaled. The retained member handles close that gap.
        The zero active-process limit prevents a new executing descendant from
        appearing between membership enumeration and the termination request.
        """
        from ctypes import wintypes as w
        limits=self.Extended()
        self._checked(self.kernel.QueryInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits),None),
                      'Read worker limits before stopping tree')
        limits.BasicLimitInformation.LimitFlags|=0x8
        limits.BasicLimitInformation.ActiveProcessLimit=0
        self._checked(self.kernel.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)),
                      'Prevent new worker descendants during shutdown')
        capacity=64
        while True:
            class Members(ctypes.Structure):
                _fields_=[('NumberOfAssignedProcesses',w.DWORD),('NumberOfProcessIdsInList',w.DWORD),
                          ('ProcessIdList',ctypes.c_size_t*capacity)]
            members=Members()
            ok=self.kernel.QueryInformationJobObject(self.job,3,ctypes.byref(members),ctypes.sizeof(members),None)
            if ok:break
            if ctypes.get_last_error()!=234 or capacity>=1024:
                self._checked(False,'Enumerate owned worker descendants')
            capacity=min(1024,max(capacity*2,int(members.NumberOfAssignedProcesses)))
        if members.NumberOfProcessIdsInList>capacity:
            raise ResourceControlUnavailable('Incomplete owned-worker membership snapshot')
        for value in members.ProcessIdList[:members.NumberOfProcessIdsInList]:
            pid=int(value)
            if pid in self._member_handles or pid==self.pid:continue
            handle=self.kernel.OpenProcess(0x1000|0x100000,False,pid)
            if not handle:
                if ctypes.get_last_error()==87:continue  # already gone, no handle/PID reuse action
                self._checked(False,'Retain owned descendant for exit confirmation')
            retained=False
            try:
                in_job=w.BOOL()
                self._checked(self.kernel.IsProcessInJob(handle,self.job,ctypes.byref(in_job)),
                              'Verify descendant belongs to owned job')
                if in_job.value:
                    self._member_handles[pid]=handle;retained=True
                # A PID that has already left this job is never waited on or
                # terminated individually. No unrelated process is selected.
            finally:
                if not retained:self.kernel.CloseHandle(handle)

    def _wait_for_quiescence(self,timeout_s=3):
        """No capture-safe success until both root and all descendants exited."""
        if not isinstance(timeout_s,(int,float)) or not math.isfinite(timeout_s) or not 0<=timeout_s<=30:
            raise ValueError('Worker-tree wait must be 0..30 seconds')
        deadline=time.monotonic()+timeout_s
        while True:
            active=self.active_processes();exit_code=self.poll()
            members_exited=True
            for handle in self._member_handles.values():
                wait=self.kernel.WaitForSingleObject(handle,0)
                self._checked(wait in (0,258),'Confirm retained descendant exit')
                members_exited&=wait==0
            if active==0 and members_exited and self._members_proven and (not self.process or exit_code is not None):
                self.quiescence_confirmed=True
                return exit_code
            remaining=deadline-time.monotonic()
            if remaining<=0:
                self.quiescence_confirmed=False
                raise ResourceControlUnavailable('Worker-tree stop was not confirmed before timeout; capture must not assume optional work is quiescent')
            time.sleep(min(.01,remaining))

    def stop(self,exit_code=75,timeout_s=3):
        """Terminate only this job's tree, then confirm zero active processes."""
        if self.closed:
            if not self.quiescence_confirmed:raise ResourceControlUnavailable('Closed worker without confirmed quiescence')
            return self._exit_code
        self.quiescence_confirmed=False
        if self.job:
            try:self._retain_tree_handles()
            except Exception:
                self._members_proven=False
                self.kernel.TerminateJobObject(self.job,exit_code)
                raise
            self._checked(self.kernel.TerminateJobObject(self.job,exit_code),'Stop owned worker tree')
        return self._wait_for_quiescence(timeout_s)

    def close(self,timeout_s=3):
        """Confirm tree termination before releasing handles; retain on failure."""
        if self.closed:return
        if self.job:
            self.stop(timeout_s=timeout_s)
        elif self.process:
            # Constructor failure before job creation/assignment never resumes
            # a child. Its exact retained process handle must still be signaled.
            if self.wait(timeout_s) is None:raise ResourceControlUnavailable('Unassigned suspended worker cleanup is not confirmed')
        else:self.quiescence_confirmed=True
        if self.job:self.kernel.CloseHandle(self.job);self.job=None
        if self.process:
            self.kernel.CloseHandle(self.process);self.process=None
        for handle in self._member_handles.values():self.kernel.CloseHandle(handle)
        self._member_handles.clear()
        if self._output:self._output.close()
        if self._input:self._input.close()
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

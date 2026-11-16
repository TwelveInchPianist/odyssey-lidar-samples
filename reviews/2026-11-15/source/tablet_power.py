"""Explicit local user sleep request; never invoked by timers or driving logic."""
import ctypes,os

def prepare_sleep():
    if os.name!='nt':raise OSError('Sleep tablet is supported on Windows only')
    from ctypes import wintypes as w
    class LUID(ctypes.Structure):_fields_=[('low',w.DWORD),('high',w.LONG)]
    class Entry(ctypes.Structure):_fields_=[('luid',LUID),('attributes',w.DWORD)]
    class Privileges(ctypes.Structure):_fields_=[('count',w.DWORD),('entry',Entry)]
    k=ctypes.WinDLL('kernel32',use_last_error=True);a=ctypes.WinDLL('advapi32',use_last_error=True)
    k.GetCurrentProcess.restype=w.HANDLE;k.CloseHandle.argtypes=[w.HANDLE]
    a.OpenProcessToken.argtypes=[w.HANDLE,w.DWORD,ctypes.POINTER(w.HANDLE)];a.OpenProcessToken.restype=w.BOOL
    a.LookupPrivilegeValueW.argtypes=[w.LPCWSTR,w.LPCWSTR,ctypes.POINTER(LUID)];a.LookupPrivilegeValueW.restype=w.BOOL
    a.AdjustTokenPrivileges.argtypes=[w.HANDLE,w.BOOL,ctypes.POINTER(Privileges),w.DWORD,ctypes.c_void_p,ctypes.c_void_p];a.AdjustTokenPrivileges.restype=w.BOOL
    token=w.HANDLE()
    if not a.OpenProcessToken(k.GetCurrentProcess(),0x20|0x8,ctypes.byref(token)):raise ctypes.WinError(ctypes.get_last_error())
    try:
        p=Privileges();p.count=1;p.entry.attributes=2
        if not a.LookupPrivilegeValueW(None,'SeShutdownPrivilege',ctypes.byref(p.entry.luid)):raise ctypes.WinError(ctypes.get_last_error())
        ctypes.set_last_error(0)
        ok=a.AdjustTokenPrivileges(token,False,ctypes.byref(p),0,None,None);error=ctypes.get_last_error()
        if not ok or error:raise ctypes.WinError(error)
    finally:k.CloseHandle(token)

def sleep_now():
    # False = sleep rather than hibernate; retain the user's configured wake events.
    p=ctypes.WinDLL('powrprof',use_last_error=True)
    p.SetSuspendState.argtypes=[ctypes.c_ubyte,ctypes.c_ubyte,ctypes.c_ubyte]
    p.SetSuspendState.restype=ctypes.c_ubyte
    if not p.SetSuspendState(0,0,0):raise ctypes.WinError(ctypes.get_last_error())

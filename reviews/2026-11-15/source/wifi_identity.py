"""Read only the currently connected Wi-Fi SSID through the Windows WLAN API.

No scan, credential access, profile guessing or privacy-permission bypass.
Access denial returns unknown and prevents automatic copy.
"""
import ctypes,os
from ctypes import wintypes

def connected_ssids():
    if os.name!='nt':return {'ssids':[],'error':'Windows Wi-Fi identity is unavailable on this platform'}
    class GUID(ctypes.Structure):
        _fields_=[('Data1',wintypes.DWORD),('Data2',wintypes.WORD),('Data3',wintypes.WORD),('Data4',ctypes.c_ubyte*8)]
    class Interface(ctypes.Structure):
        _fields_=[('guid',GUID),('description',wintypes.WCHAR*256),('state',wintypes.DWORD)]
    class ConnectionPrefix(ctypes.Structure):
        _fields_=[('state',wintypes.DWORD),('mode',wintypes.DWORD),('profile',wintypes.WCHAR*256),
                  ('ssid_length',wintypes.DWORD),('ssid',ctypes.c_ubyte*32)]
    wlan=ctypes.WinDLL(os.path.join(os.environ.get('SystemRoot',r'C:\Windows'),'System32','wlanapi.dll'));handle=wintypes.HANDLE();version=wintypes.DWORD();interfaces=ctypes.c_void_p()
    wlan.WlanOpenHandle.argtypes=[wintypes.DWORD,ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.POINTER(wintypes.HANDLE)]
    wlan.WlanOpenHandle.restype=wintypes.DWORD
    wlan.WlanEnumInterfaces.argtypes=[wintypes.HANDLE,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)];wlan.WlanEnumInterfaces.restype=wintypes.DWORD
    wlan.WlanQueryInterface.argtypes=[wintypes.HANDLE,ctypes.POINTER(GUID),ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int)]
    wlan.WlanQueryInterface.restype=wintypes.DWORD
    wlan.WlanFreeMemory.argtypes=[ctypes.c_void_p];wlan.WlanFreeMemory.restype=None
    wlan.WlanCloseHandle.argtypes=[wintypes.HANDLE,ctypes.c_void_p];wlan.WlanCloseHandle.restype=wintypes.DWORD
    error=wlan.WlanOpenHandle(2,None,ctypes.byref(version),ctypes.byref(handle))
    if error:return {'ssids':[],'error':f'Windows Wi-Fi service unavailable ({error})'}
    names=[];errors=[]
    try:
        error=wlan.WlanEnumInterfaces(handle,None,ctypes.byref(interfaces))
        if error:return {'ssids':[],'error':f'Cannot enumerate Wi-Fi interfaces ({error})'}
        count=ctypes.cast(interfaces,ctypes.POINTER(wintypes.DWORD))[0]
        if count>64:return {'ssids':[],'error':'Invalid Wi-Fi interface count'}
        for i in range(count):
            interface=Interface.from_address(interfaces.value+8+i*ctypes.sizeof(Interface))
            if interface.state!=1:continue
            size=wintypes.DWORD();data=ctypes.c_void_p();kind=ctypes.c_int()
            error=wlan.WlanQueryInterface(handle,ctypes.byref(interface.guid),7,None,ctypes.byref(size),ctypes.byref(data),ctypes.byref(kind))
            try:
                if error:
                    errors.append(f'Wi-Fi identity unavailable ({error}); Windows may require location/Wi-Fi permission');continue
                if size.value<ctypes.sizeof(ConnectionPrefix):errors.append('Invalid Wi-Fi response');continue
                connection=ConnectionPrefix.from_address(data.value)
                if connection.state!=1 or not 0<connection.ssid_length<=32:continue
                try:names.append(bytes(connection.ssid[:connection.ssid_length]).decode('utf-8'))
                except UnicodeDecodeError:errors.append('Connected SSID is not valid UTF-8; automatic matching disabled')
            finally:
                if data.value:wlan.WlanFreeMemory(data)
    finally:
        if interfaces.value:wlan.WlanFreeMemory(interfaces)
        wlan.WlanCloseHandle(handle,None)
    return {'ssids':sorted(set(names)),'error':'; '.join(errors) if errors else None}

"""Keep FCW warm across screen-off / Modern Standby; no hibernate request."""
import ctypes,os,threading,time,uuid

class DisplayPower:
    def __init__(self,on_change):
        if os.name!='nt':raise OSError('Tablet standby requires Windows')
        from ctypes import wintypes as w
        class Guid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
        class Setting(ctypes.Structure):_fields_=[('guid',Guid),('length',w.DWORD),('data',w.DWORD)]
        Callback=ctypes.WINFUNCTYPE(w.ULONG,ctypes.c_void_p,w.ULONG,ctypes.c_void_p)
        class Recipient(ctypes.Structure):_fields_=[('callback',Callback),('context',ctypes.c_void_p)]
        self.guid=Guid();raw=uuid.UUID('6fe69556-704a-47a0-8f24-c28d936fda47').bytes_le;self.guid.bytes[:]=raw
        def changed(context,kind,setting):
            if kind==0x8013 and setting:
                value=ctypes.cast(setting,ctypes.POINTER(Setting)).contents
                if bytes(value.guid.bytes)==raw and value.length==4:on_change(value.data)
            return 0
        self.callback=Callback(changed);self.recipient=Recipient(self.callback,None)
        self.handle=w.HANDLE();self.dll=ctypes.WinDLL('powrprof',use_last_error=True)
        self.dll.PowerSettingRegisterNotification.argtypes=[ctypes.POINTER(Guid),w.DWORD,ctypes.c_void_p,ctypes.POINTER(w.HANDLE)]
        self.dll.PowerSettingRegisterNotification.restype=w.DWORD
        self.dll.PowerSettingUnregisterNotification.argtypes=[w.HANDLE];self.dll.PowerSettingUnregisterNotification.restype=w.DWORD
        error=self.dll.PowerSettingRegisterNotification(ctypes.byref(self.guid),2,ctypes.byref(self.recipient),ctypes.byref(self.handle))
        if error:raise ctypes.WinError(error)
    def off(self):
        from ctypes import wintypes as w
        user=ctypes.WinDLL('user32',use_last_error=True)
        user.SendMessageTimeoutW.argtypes=[w.HWND,w.UINT,ctypes.c_size_t,ctypes.c_ssize_t,w.UINT,w.UINT,ctypes.POINTER(ctypes.c_size_t)]
        user.SendMessageTimeoutW.restype=ctypes.c_ssize_t
        result=ctypes.c_size_t()
        if not user.SendMessageTimeoutW(0xffff,0x112,0xf170,2,2,2000,ctypes.byref(result)):
            raise OSError('Windows did not acknowledge display-off request')
    def close(self):
        if self.handle:self.dll.PowerSettingUnregisterNotification(self.handle);self.handle=None

class Standby:
    def __init__(self,app,native_factory=DisplayPower):
        self.app=app;self.events=threading.Event();self.lock=threading.Lock()
        self.armed=False;self.saw_off=False;self.resume=False;self.status='awake';self.deadline=0.
        self.native=native_factory(self.display_changed)
        self.thread=threading.Thread(target=self.run,daemon=True,name='tablet-standby');self.thread.start()
    def display_changed(self,value):
        with self.lock:
            if self.armed:
                if value==0:self.saw_off=True;self.status='standby'
                elif value==1 and self.saw_off:self.resume=True
        self.events.set()
    def request(self):
        with self.app.start_gate,self.app.lock,self.lock:
            if self.app.closing or self.app.shutdown.is_set():raise ValueError('FCW is closing')
            if self.app.starting or self.app.busy or (self.app.session and not self.app.session.stopped):raise ValueError('Wait for recording closeout before standby')
            if self.armed:raise ValueError('Standby already requested')
            self.armed=True;self.saw_off=False;self.resume=False;self.status='preparing';self.deadline=time.monotonic()+15
            self.app.standby_pending=True
        threading.Thread(target=self.enter,daemon=True,name='enter-tablet-standby').start()
    def enter(self):
        try:
            if self.app.learning:self.app.learning.quiesce()
            if self.app.closing:return
            time.sleep(.3)
            if self.app.closing or self.app.shutdown.is_set():return
            self.native.off()
        except Exception as e:self.finish('Standby failed: '+str(e))
    def finish(self,message):
        with self.app.lock,self.lock:
            self.armed=False;self.resume=False;self.saw_off=False
            self.app.standby_pending=False;self.status=message
            if self.app.autodrive:
                self.app.autodrive.manual_hold=False;self.app.autodrive.restart_hold_until=0.
        self.app.review_status=message
    def run(self):
        while not self.app.shutdown.is_set():
            self.events.wait(.5);self.events.clear()
            with self.lock:
                resume=self.resume;failed=self.armed and not self.saw_off and time.monotonic()>self.deadline
            if resume:self.finish('Awake — FCW ready; automatic recording re-armed.')
            elif failed:self.finish('Display did not turn off; FCW remains ready.')
    def snapshot(self):
        with self.lock:return {'pending':self.armed,'state':self.status}
    def close(self):
        self.events.set();self.native.close()

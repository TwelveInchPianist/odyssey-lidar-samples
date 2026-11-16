"""Explicit saved-profile Wi-Fi connection, outside capture/inference threads."""
import os,re,subprocess,threading,time
COMMA_WIFI_PROFILE='weedle-3bf3'  # Owner confirmed the existing saved Windows profile.

class WifiConnection:
 def __init__(self,runner=None,*,supported=None,sleep=time.sleep):
  self._run=runner or self._command;self._sleep=sleep;self._supported=os.name=='nt' if supported is None else supported;self._lock=threading.Lock();self._worker=None;self._stop=threading.Event()
  self._state={'state':'idle','busy':False,'profile':COMMA_WIFI_PROFILE,'message':'Connect the tablet to the comma hotspot.'}
 @staticmethod
 def _command(args):
  return subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,text=True,errors='replace',timeout=5,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 def snapshot(self):
  with self._lock:return dict(self._state)
 def _set(self,state,message,busy=False):
  with self._lock:self._state.update(state=state,message=message,busy=busy)
 @staticmethod
 def connected(text):
  names=re.findall(r'^\s*SSID\s*:\s*(.*?)\s*$',text,re.M)
  return COMMA_WIFI_PROFILE in names and bool(re.search(r'^\s*State\s*:\s*connected\s*$',text,re.M|re.I))
 def start(self):
  with self._lock:
   if self._state['busy']:return dict(self._state)
   self._state.update(state='connecting',busy=True,message='Checking saved comma Wi-Fi profile…')
  self._stop.clear();self._worker=threading.Thread(target=self._connect,daemon=True,name='comma-wifi-connect');self._worker.start();return self.snapshot()
 def _connect(self):
  if not self._supported:self._set('error','Wi-Fi connection is available on the Windows tablet.');return
  try:
   check=self._run(['netsh','wlan','show','profile','name='+COMMA_WIFI_PROFILE])
   if check.returncode:self._set('error',"Saved Wi-Fi profile '"+COMMA_WIFI_PROFILE+"' was not found. Connect once using Windows Wi-Fi settings.");return
   if self._stop.is_set():return
   current=self._run(['netsh','wlan','show','interfaces'])
   if current.returncode==0 and self.connected(current.stdout):self._set('connected','Connection verified: '+COMMA_WIFI_PROFILE+' at '+time.strftime('%H:%M')+'.');return
   if self._stop.is_set():return
   self._set('connecting','Connecting to '+COMMA_WIFI_PROFILE+'…',True)
   result=self._run(['netsh','wlan','connect','name='+COMMA_WIFI_PROFILE])
   if result.returncode:self._set('error','Windows could not connect to '+COMMA_WIFI_PROFILE+'. Check that Wi-Fi and the comma hotspot are on.');return
   deadline=time.monotonic()+25
   for _ in range(12):
    if time.monotonic()>=deadline:break
    if self._stop.is_set():return
    self._sleep(2);state=self._run(['netsh','wlan','show','interfaces'])
    if state.returncode==0 and self.connected(state.stdout):self._set('connected','Connection verified: '+COMMA_WIFI_PROFILE+' at '+time.strftime('%H:%M')+'.');return
   self._set('error','Not connected yet. Keep the comma powered with its hotspot on, then retry.')
  except (OSError,subprocess.SubprocessError):self._set('error','Windows Wi-Fi request did not finish. Check Wi-Fi, then retry.')
  finally:
   with self._lock:self._state['busy']=False
 def close(self):self._stop.set()


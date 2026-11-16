"""User intent only; no capture/perception policy. The watchdog honors this marker."""
import ctypes,json,os,time
from pathlib import Path
MARKER='manual-quit.json'
def _session():
 if os.name!='nt':return None
 value=ctypes.c_ulong()
 if not ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(),ctypes.byref(value)):raise OSError('Could not identify Windows session')
 return value.value
def mark_manual_quit(base):
 if os.name!='nt':return
 p=Path(base)/MARKER;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'intent':'user_quit','session_id':_session(),'at':time.time()}),encoding='utf-8');tmp.replace(p)
def clear_manual_quit(base):
 p=Path(base)/MARKER
 try:p.unlink()
 except FileNotFoundError:pass

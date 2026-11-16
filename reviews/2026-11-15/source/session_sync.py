"""Phase 4: home-network-gated, verified copies. Originals are never deleted."""
from contextlib import contextmanager
import hashlib, json, os, shutil, threading
import runtime_clock as time
from pathlib import Path
from hindsight import write_json
from wifi_identity import connected_ssids

DEFAULT_SYNC={'enabled':False,'home_ssid':'','destination':''}
class Paused(Exception):pass
class Conflict(Exception):pass

def validate_sync(value,record_root):
    if not isinstance(value,dict) or set(value)!=set(DEFAULT_SYNC) or type(value['enabled']) is not bool:raise ValueError('Invalid sync settings')
    ssid=value['home_ssid'];destination=value['destination']
    if not isinstance(ssid,str) or not isinstance(destination,str):raise ValueError('SSID and destination must be text')
    if any(c in ssid for c in '\r\n\0') or len(ssid.encode('utf-8'))>32:raise ValueError('Invalid home Wi-Fi name')
    destination=os.path.expandvars(os.path.expanduser(destination.strip().strip('"')))
    if value['enabled'] and (not ssid or not destination):raise ValueError('Choose both the home Wi-Fi name and destination folder')
    if destination:
        p=Path(destination)
        if not p.is_absolute():raise ValueError('Choose an absolute destination folder')
        # Lexical check here avoids contacting an unavailable UNC server during
        # configuration. Resolved ancestry is checked again before any copy.
        a=os.path.normcase(os.path.abspath(record_root));b=os.path.normcase(os.path.abspath(p))
        try:common=os.path.commonpath([a,b])
        except ValueError:common=None
        if common in (a,b):raise ValueError('Destination and recording folders must be separate, not nested')
    return {'enabled':value['enabled'],'home_ssid':ssid,'destination':destination}

def reparse(path):
    st=Path(path).lstat()
    return Path(path).is_symlink() or bool(getattr(st,'st_file_attributes',0)&0x400)

def unsafe_destination_link(path):
    p=Path(path)
    return p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction())

def files_in_session(folder):
    folder=Path(folder);result=[];skipped=[]
    for base,dirs,files in os.walk(folder,followlinks=False):
        parent=Path(base)
        for name in list(dirs):
            if reparse(parent/name):skipped.append((parent/name).relative_to(folder).as_posix());dirs.remove(name)
        for name in files:
            p=parent/name
            if reparse(p):skipped.append(p.relative_to(folder).as_posix());continue
            if name.endswith(('.tmp','.partial')) or name=='.odyssey-sync.json':continue
            if p.resolve().is_relative_to(folder.resolve()):result.append(p)
            else:skipped.append(p.relative_to(folder).as_posix())
    return sorted(result),skipped

def stat_signature(path):
    s=Path(path).stat();return [s.st_size,s.st_mtime_ns]

class SessionSync:
    def __init__(self,record_root,data_root,config_provider,activity,activity_lock=None,ssid_provider=connected_ssids,chunk_size=4*1024*1024):
        self.record_root=Path(record_root);self.data_root=Path(data_root);self.config_provider=config_provider;self.activity=activity
        self.activity_lock=activity_lock or threading.RLock();self.ssid_provider=ssid_provider;self.chunk_size=chunk_size
        self.stop_event=threading.Event();self.thread=None;self.state={'status':'disabled','message':'Choose home Wi-Fi and a destination to enable copying.'}
        self.run_lock=threading.Lock()
    def profile(self):return validate_sync(self.config_provider(),self.record_root)
    @contextmanager
    def allowed(self,profile):
        if self.stop_event.is_set():raise Paused('sync stopping')
        if self.activity():raise Paused('recording or offline review active')
        wifi=self.ssid_provider()
        if wifi.get('error') or profile['home_ssid'] not in wifi.get('ssids',[]):raise Paused(wifi.get('error') or 'waiting for configured home Wi-Fi')
        with self.activity_lock:
            if self.activity() or self.profile()!=profile or self.stop_event.is_set():raise Paused('recording, review or settings changed')
            yield
    def hash_file(self,path,profile):
        digest=hashlib.sha256()
        with Path(path).open('rb') as f:
            while True:
                with self.allowed(profile):data=f.read(self.chunk_size)
                if not data:break
                digest.update(data)
        return digest.hexdigest()
    def _copy_file(self,source,target,profile,expected,signature):
        temporary=target.with_name(target.name+'.odyssey-'+expected[:12]+'.partial')
        with self.allowed(profile):target.parent.mkdir(parents=True,exist_ok=True)
        if not target.resolve().is_relative_to(Path(profile['destination']).resolve()):raise ValueError('Destination path escaped selected folder')
        if temporary.exists() and unsafe_destination_link(temporary):raise Conflict('Unexpected link at staging path')
        offset=temporary.stat().st_size if temporary.exists() else 0
        if offset>signature[0]:raise Conflict('Invalid staging file size')
        if offset:
            # Validate the resumed prefix rather than trusting a matching size.
            with source.open('rb') as a,temporary.open('rb') as b:
                remaining=offset
                while remaining:
                    with self.allowed(profile):left=a.read(min(self.chunk_size,remaining));right=b.read(len(left))
                    if not left or left!=right:raise Conflict('Staging prefix differs from original; original retained')
                    remaining-=len(left)
        needed=signature[0]-offset
        if shutil.disk_usage(target.parent).free<needed+4*1024**3:raise OSError('Not enough destination space while reserving 4 GB; originals retained')
        with self.allowed(profile):
            inp=source.open('rb');out=temporary.open('ab' if offset else 'wb')
        with inp,out:
            inp.seek(offset)
            while True:
                with self.allowed(profile):
                    block=inp.read(self.chunk_size)
                    if block:out.write(block);out.flush()
                if not block:break
        if stat_signature(source)!=signature:raise Paused('source changed during copy; waiting for a stable completed session')
        if self.hash_file(temporary,profile)!=expected:raise Conflict('Copy checksum mismatch; original retained')
        with self.allowed(profile):temporary.replace(target)
    def copy_session(self,folder,profile):
        folder=Path(folder)
        if reparse(folder) or folder.resolve().parent!=self.record_root.resolve():raise ValueError('Source must be a direct, regular session folder')
        if not (folder/'final-state.json').exists():return {'status':'incomplete'}
        offline=folder/'offline-status.json'
        if offline.exists() and json.loads(offline.read_text()).get('state') not in ('complete','complete_with_errors'):return {'status':'review_pending'}
        source_id=hashlib.sha256((folder/'session.json').read_bytes()+folder.name.encode()).hexdigest()
        destination=Path(profile['destination']).resolve();source_root=self.record_root.resolve()
        if destination==source_root or destination.is_relative_to(source_root) or source_root.is_relative_to(destination):raise ValueError('Resolved source and destination overlap')
        with self.allowed(profile):destination.mkdir(parents=True,exist_ok=True)
        target=destination/folder.name;marker=target/'.odyssey-sync.json'
        if target.exists():
            if unsafe_destination_link(target):raise Conflict('Destination session is a link; refusing copy')
            try:old=json.loads(marker.read_text())
            except (OSError,ValueError):old={}
            if old.get('source_id')!=source_id:
                target=destination/(folder.name+'-fcw-'+source_id[:8]);marker=target/'.odyssey-sync.json'
                if target.exists():
                    try:old=json.loads(marker.read_text())
                    except (OSError,ValueError):old={}
                    if old.get('source_id')!=source_id:raise Conflict('Destination name collision; existing folder retained')
                else:old={}
        else:old={}
        if target.exists() and unsafe_destination_link(target):raise Conflict('Destination session is a link; refusing copy')
        if not target.resolve().is_relative_to(destination):raise ValueError('Destination escaped selected folder')
        with self.allowed(profile):
            target.mkdir(exist_ok=True)
            write_json(marker,{'schema_version':1,'source_id':source_id,'status':'copying','files':old.get('files',{})})
        paths,skipped=files_in_session(folder);manifest=dict(old.get('files',{}));copied=0;unchanged=0
        for source in paths:
            relative=source.relative_to(folder).as_posix();dest=target/relative;sig=stat_signature(source);prior=manifest.get(relative)
            # Any existing intermediate junction must remain inside the chosen
            # destination. The source walker never follows links.
            if not dest.resolve().is_relative_to(destination):raise Conflict('Destination contains an escaping link')
            if dest.exists() and unsafe_destination_link(dest):raise Conflict('Destination file is a link')
            for parent in dest.parents:
                if parent==destination:break
                if parent.exists() and unsafe_destination_link(parent):raise Conflict('Destination contains a directory link')
            if prior and prior.get('source_signature')==sig and dest.exists() and prior.get('destination_signature')==stat_signature(dest):
                unchanged+=1;continue
            digest=self.hash_file(source,profile)
            if dest.exists():
                current=self.hash_file(dest,profile)
                if current!=digest and (prior is None or current!=prior.get('sha256')):raise Conflict('Destination was edited: '+relative+'; existing file and original retained')
                if current==digest:
                    manifest[relative]={'sha256':digest,'source_signature':sig,'destination_signature':stat_signature(dest)};unchanged+=1;continue
            self.state={'status':'copying','message':'Copying '+folder.name+' / '+relative,'session':folder.name,'files_copied':copied}
            self._copy_file(source,dest,profile,digest,sig)
            manifest[relative]={'sha256':digest,'source_signature':sig,'destination_signature':stat_signature(dest)};copied+=1
            with self.allowed(profile):write_json(marker,{'schema_version':1,'source_id':source_id,'status':'copying','files':manifest})
        with self.allowed(profile):write_json(marker,{'schema_version':1,'source_id':source_id,'status':'complete_with_skipped_links' if skipped else 'complete',
                                                      'files':manifest,'skipped_links':skipped,'completed_at':time.time(),'originals_deleted':False})
        return {'status':'complete','session':folder.name,'copied_files':copied,'unchanged_files':unchanged,'skipped_links':skipped,'destination':str(target),'originals_deleted':False}
    def run_once(self):
        if not self.run_lock.acquire(blocking=False):return self.state
        try:
            profile=self.profile()
            if not profile['enabled']:self.state={'status':'disabled','message':'Sync is disabled until home Wi-Fi and a destination are configured.'};return self.state
            with self.allowed(profile):pass
            results=[]
            if self.record_root.exists():
                for folder in sorted(self.record_root.iterdir()):
                    if folder.is_dir() and not reparse(folder) and (folder/'final-state.json').exists():
                        try:results.append(self.copy_session(folder,profile))
                        except (OSError,ValueError,Conflict) as e:results.append({'status':'attention','session':folder.name,'message':str(e)})
            complete=sum(r['status']=='complete' for r in results);pending=sum(r['status']=='review_pending' for r in results)
            attention=any(r['status']=='attention' for r in results)
            self.state={'status':'attention' if attention else 'idle','message':f'{complete} completed-session copies checked; {pending} awaiting review. Originals retained.','sessions':results,'originals_deleted':False}
        except Paused as e:self.state={'status':'paused','message':str(e),'originals_deleted':False}
        except (OSError,ValueError,Conflict) as e:self.state={'status':'attention','message':str(e),'originals_deleted':False}
        finally:self.run_lock.release()
        self.data_root.mkdir(parents=True,exist_ok=True);write_json(self.data_root/'sync-status.json',self.state)
        return self.state
    def start(self):
        if self.thread:return
        def worker():
            while not self.stop_event.is_set():
                self.run_once();self.stop_event.wait(30)
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=3)

"""Authenticated, leased stages inside one kernel-bounded optional worker.

The app signs a short-lived launch spec using a per-launch environment secret.
This detects accidental/tampered launch files; it is not a security boundary
against the same OS user controlling process environments. Actual job limits
and live parent creation identity are checked before opening a work store.
"""
import errno, hashlib, hmac, json, os, re, secrets, shutil, threading
import runtime_clock as time
from pathlib import Path
from offline_jobs import WorkPaused
from .jobs import JobStore,LeaseLost,current_owner,probe_process
from .datasets import DatasetPaused
from .resources import current_job_limits

VERSION='learning-pipeline-v1'
IDLE_RESERVE_BYTES=10_000_000_000+512*1024*1024
AUTH_SECRET='ODYSSEY_LEARNING_LAUNCH_SECRET'
AUTH_TAG='ODYSSEY_LEARNING_SPEC_HMAC'
NO_CLAIM=76
class StageBlocked(Exception):pass

def _regular(path):
    path=Path(path)
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('Worker paths may not be links/junctions')

def validate_spec_paths(path,spec):
    required={'pipeline_version','run_id','parent_owner','job_id','data_root','record_root','assets','engine_version','cancel_file','budget'}
    if not isinstance(spec,dict) or set(spec)!=required:raise ValueError('Unsupported worker specification fields')
    if spec['pipeline_version']!=VERSION:raise ValueError('Worker pipeline version mismatch')
    if not re.fullmatch('[0-9a-f]{32}',str(spec['run_id'])) or not re.fullmatch('[0-9a-f]{64}',str(spec['job_id'])):raise ValueError('Invalid worker/job identity')
    path=Path(path);root=Path(spec['data_root']);records=Path(spec['record_root']);worker=root/'learning/worker'
    for item in (root,root/'learning',worker,records,path,Path(spec['cancel_file']),Path(spec['assets'])):_regular(item)
    if not root.is_absolute() or not records.is_absolute() or not Path(spec['assets']).is_absolute():raise ValueError('Worker paths must be absolute')
    if records.resolve().parent!=root.resolve() or not records.is_dir():raise ValueError('Recording root must be a direct data-root child')
    if path.resolve()!=worker.resolve()/(spec['run_id']+'.json'):raise ValueError('Worker specification outside app launch directory')
    if Path(spec['cancel_file']).resolve()!=worker.resolve()/(spec['run_id']+'.cancel'):raise ValueError('Cancellation path outside launch directory')
    if not Path(spec['assets']).is_dir():raise ValueError('Worker asset directory unavailable')
    owner=spec['parent_owner']
    if not isinstance(owner,dict) or set(owner)!={'pid','start_token'} or type(owner['pid']) is not int or owner['pid']<=0 or not isinstance(owner['start_token'],str) or not 1<=len(owner['start_token'])<=200:raise ValueError('Invalid parent creation identity')
    budget=spec['budget']
    if not isinstance(budget,dict) or set(budget)!={'memory_mb','cpu_rate'} or type(budget['memory_mb']) is not int or not 128<=budget['memory_mb']<=1024 or type(budget['cpu_rate']) is not int or not 100<=budget['cpu_rate']<=3000:raise ValueError('Worker budget outside the bounded learning contract')
    return spec

def write_launch_spec(path,spec):
    """App-only creator: auth key is passed to the child, never saved on disk."""
    validate_spec_paths(path,spec)
    raw=json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>64_000:raise ValueError('Worker launch specification too large')
    secret=secrets.token_hex(32);tag=hmac.new(bytes.fromhex(secret),raw,hashlib.sha256).hexdigest()
    path=Path(path);temporary=path.with_suffix('.json.tmp')
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return {AUTH_SECRET:secret,AUTH_TAG:tag}

def load_launch_spec(path):
    path=Path(path);_regular(path)
    if path.stat().st_size>64_000:raise ValueError('Worker launch specification too large')
    raw=path.read_bytes();secret=os.environ.pop(AUTH_SECRET,'');tag=os.environ.pop(AUTH_TAG,'')
    if not re.fullmatch('[0-9a-f]{64}',secret) or not re.fullmatch('[0-9a-f]{64}',tag):raise ValueError('Worker was not authenticated by the application')
    if not hmac.compare_digest(hmac.new(bytes.fromhex(secret),raw,hashlib.sha256).hexdigest(),tag):raise ValueError('Worker specification authentication failed')
    spec=validate_spec_paths(path,json.loads(raw))
    from runtime import APP_VERSION
    if spec['engine_version']!=APP_VERSION:raise ValueError('Worker engine version differs from application')
    proof=probe_process(spec['parent_owner']['pid'])
    if not (proof.get('authoritative') and proof.get('state')=='alive' and proof.get('start_token')==spec['parent_owner']['start_token']):raise ValueError('Worker parent is not authoritatively alive')
    limits=current_job_limits('Local\\OdysseyFCW-learning-'+spec['run_id']);budget=spec['budget']
    if not (limits.get('job_memory_limit_enforced') and 0<limits.get('job_memory_limit_bytes',0)<=budget['memory_mb']*1024**2 and limits.get('cpu_enabled') and limits.get('cpu_hard_cap') and 0<limits.get('cpu_rate',0)<=budget['cpu_rate'] and limits.get('kill_on_owner_close') and 0<limits.get('active_process_limit',0)<=4):raise ValueError('Worker kernel budget does not satisfy launch contract')
    return spec

def session_path(record_root,name):
    root=Path(record_root).resolve()
    if not isinstance(name,str) or not name or Path(name).name!=name or name in ('.','..'):raise ValueError('Invalid session identity')
    folder=root/name;_regular(folder)
    if folder.resolve().parent!=root or not (folder/'final-state.json').is_file():raise ValueError('Only direct, closed recording children can be processed')
    return folder

def execute(job,spec,cancel,progress):
    root=Path(spec['data_root']);stage=job['stage']
    if stage=='radar_training':
        from .training import train_available,input_signature
        if input_signature(root,Path(spec['assets']),spec['engine_version'])!=job['input_signature']:raise ValueError('Training inputs changed before dispatch')
        return train_available(root,Path(spec['assets']),spec['engine_version'],cancel=cancel,progress=progress)
    folder=session_path(spec['record_root'],job['payload']['session'])
    from .catalog import inventory_cache_key
    if job['input_signature'].get('engine_version')!=spec['engine_version'] or inventory_cache_key(folder,cancel)!=job['input_signature'].get('inventory'):raise ValueError('Recording inventory/engine changed before stage dispatch')
    if stage=='audible_images':
        from audible_review import build_images
        result=build_images(folder,cancel=cancel)
    elif stage=='self_calibration':
        from selfcal import run_session
        result=run_session(folder,root,cancel=cancel,progress=progress)
    elif stage=='catalog':
        from .catalog import index_session
        result=index_session(folder,root,cancel)
    elif stage=='radar_dataset':
        from .datasets import extract_session_radar
        result=extract_session_radar(folder,root,cancel)
    elif stage=='route_trajectory':
        from mapping.trajectory import build_trajectory
        result=build_trajectory(folder,cancel)
    elif stage=='raw_replay':
        from runtime import replay_and_review
        result=replay_and_review(folder,cancel,progress=progress)
    elif stage=='review':
        from evaluation import summarize
        value=summarize(folder,recompute_hindsight=False,cancel=cancel)
        result={'frames':value['frames'],'hindsight':value.get('hindsight'),'folder':str(folder)}
    elif stage=='atlas':
        from mapping.pipeline import update_atlas
        result=update_atlas(folder,root/'maps',cancel)
        if result.get('status') in ('unavailable','rejected'):raise StageBlocked('atlas_prerequisite: '+str(result.get('reason','unavailable'))[:1000])
    else:raise ValueError('Unsupported learning stage')
    if inventory_cache_key(folder,cancel)!=job['input_signature']['inventory']:raise ValueError('Recording inventory changed before stage completion')
    return result

def _reserve_error(error):
    return getattr(error,'errno',None)==errno.ENOSPC or any(text in str(error).lower() for text in ('disk reserve','above the disk reserve','free-space reserve'))

def run_worker(spec_path):
    spec=load_launch_spec(spec_path);root=Path(spec['data_root'])
    store=JobStore(root/'learning/jobs.sqlite');stop=threading.Event();lease_fault=threading.Event()
    parent=spec['parent_owner'];last_check=[0.,False];cancel_reason=['']
    def cancelled():
        if stop.is_set() or lease_fault.is_set():cancel_reason[0]='lease_or_worker_stop';return True
        if Path(spec['cancel_file']).exists():cancel_reason[0]='capture_preempted';return True
        now=time.monotonic()
        if now-last_check[0]>=.5:
            proof=probe_process(parent['pid'])
            last_check[:]=[now,not (proof.get('authoritative') and proof.get('state')=='alive' and proof.get('start_token')==parent['start_token'])]
            if last_check[1]:cancel_reason[0]='parent_identity_unavailable'
            elif shutil.disk_usage(root).free<=IDLE_RESERVE_BYTES:cancel_reason[0]='disk_reserve';last_check[1]=True
        return last_check[1]
    try:job=store.claim(current_owner(),job_id=spec['job_id'],cancel=cancelled,exclusive=True)
    except WorkPaused:return 75
    if not job:return NO_CLAIM
    token=job['lease_token']
    def heartbeat():
        while not stop.wait(5):
            try:store.heartbeat(job['id'],token)
            except Exception:lease_fault.set();return
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();last_progress=[0.]
    def progress(value):
        if cancelled():raise WorkPaused('Worker cancellation requested')
        if time.monotonic()-last_progress[0]>=1:
            store.checkpoint(job['id'],token,value if isinstance(value,dict) else {'message':str(value)[:2000]})
            last_progress[0]=time.monotonic()
    try:
        result=execute(job,spec,cancelled,progress)
        if cancelled():raise WorkPaused('Cancellation before stage commit')
        store.finish(job['id'],token,result);return 0
    except StageBlocked as error:
        try:store.block(job['id'],token,str(error)[:2000])
        except LeaseLost:return NO_CLAIM
        return 77
    except (WorkPaused,DatasetPaused) as error:
        try:
            if cancel_reason[0]=='disk_reserve' or _reserve_error(error):store.block(job['id'],token,'disk_reserve')
            elif cancel_reason[0]=='capture_preempted':store.pause(job['id'],token,'capture_preempted')
            else:store.block(job['id'],token,cancel_reason[0] or 'worker_prerequisite_unavailable')
        except LeaseLost:return NO_CLAIM
        return 75
    except Exception as error:
        try:
            if _reserve_error(error):store.block(job['id'],token,'disk_reserve')
            else:store.fail(job['id'],token,type(error).__name__+': '+str(error)[:2000])
        except LeaseLost:return NO_CLAIM
        return 1
    finally:stop.set();thread.join(timeout=1)

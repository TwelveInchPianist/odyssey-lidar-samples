"""Audible-event bookmarks and automatically matched road-camera stills."""
import json,os,re,shutil,subprocess,threading,uuid,html
from pathlib import Path
import runtime_clock as time
from offline_jobs import check_work
import numpy as np

def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

ENCODERS={'roadEncodeIdx':'fcamera.hevc','qRoadEncodeIdx':'qcamera.ts','wideRoadEncodeIdx':'ecamera.hevc'}
ENCODE_DATA={'roadEncodeData':'roadEncodeIdx','qRoadEncodeData':'qRoadEncodeIdx','wideRoadEncodeData':'wideRoadEncodeIdx'}
ROUTE=re.compile(r'(?:[0-9a-f]{8}--[0-9a-f]{10}|[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2})')
EVENT=re.compile(r'[0-9a-f]{32}')

def rows(path):
    if not Path(path).is_file():return []
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            try:
                value=json.loads(line)
                if isinstance(value,dict):result.append(value)
            except ValueError:continue
    return result

def save(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

class Journal:
    def __init__(self,folder,session_id,enabled):
        self.folder=Path(folder);self.session_id=session_id;self.enabled=enabled;self.route=None;self.lock=threading.RLock()
        self.events=(self.folder/'audible-warning-events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.index=(self.folder/'camera-frame-index.jsonl').open('a',encoding='utf-8',buffering=1)
    def ingest(self,item,receipt):
        if not self.enabled:return
        with self.lock:
            if item.get('kind')=='hello':self.route=None
            if item.get('kind')=='route_reference':
                value=item.get('route');self.route=value if isinstance(value,str) and ROUTE.fullmatch(value) else None
            service=ENCODE_DATA.get(item.get('service'),item.get('service'))
            if item.get('kind')!='sample' or service not in ENCODERS or item.get('valid') is not True:return
            d=item.get('data',{})
            if item.get('service') in ENCODE_DATA:d=d.get('idx',{})
            if not all(type(d.get(k)) is int and d[k]>=0 for k in ('frameId','segmentNum','segmentId','timestampEof')) or d['timestampEof']<=0:return
            row={'route':self.route,'service':service,'source_service':item['service'],'frame_id':d['frameId'],'segment':d['segmentNum'],
                 'frame_index':d['segmentId'],'timestamp_eof_ns':d['timestampEof'],'receipt_ns':receipt}
            self.index.write(json.dumps(row,separators=(',',':'))+'\n')
    def warning(self,result,context,played,request_ns,sync):
        event_id=uuid.uuid4().hex;alert=dict(result['alerts'][0]);alert['audible_event_id']=event_id
        event={'schema_version':1,'event_id':event_id,'session_id':self.session_id,'alert':alert,
               'frame_at_ns':context['at_ns'],'speaker_request_ns':request_ns,'recorded_wall_ns':time.time_ns(),
               'delivery':'submitted_to_windows' if played else 'failed_or_expired','audibility_physically_verified':False,
               'comma_request_ns':int(request_ns-sync[0]) if sync is not None else None,
               'clock_bound_ns':int(sync[1]) if sync is not None else None,
               'forced_experiment':result.get('forced_experiment'),'quality_status':result.get('status'),
               'speed_mps':context.get('speed_mps'),'yaw_rate_rps':context.get('yaw_rate_rps'),
               'images_status':'pending_automatic_road_video_match','review_label':None}
        with self.lock:self.events.write(json.dumps(event,allow_nan=False,default=json_value,separators=(',',':'))+'\n')
        result['alerts'][0]=alert;result['audible_event_id']=event_id
        return event
    def close(self):
        with self.lock:self.events.close();self.index.close()

def _fetch(folder,cfg,row,cancel):
    route=row.get('route');segment=row.get('segment');service=row.get('service')
    if not isinstance(route,str) or not ROUTE.fullmatch(route) or type(segment) is not int or not 0<=segment<100000 or service not in ENCODERS:raise ValueError('No verified native recording reference')
    cache=folder/'road-video-cache'/f'{route}--{segment}';cache.mkdir(parents=True,exist_ok=True)
    target=cache/ENCODERS[service]
    if target.is_file() and target.stat().st_size:return target
    if not cfg.get('audible_review',{}).get('auto_fetch'):raise ValueError('Automatic road-video retrieval is disabled')
    host=cfg.get('comma_host','')
    if not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*',host):raise ValueError('Comma host unavailable')
    if shutil.disk_usage(folder).free<max(2e9,float(cfg.get('minimum_free_gb',0))*1e9+1e9):raise ValueError('Insufficient free space above recording reserve')
    scp=shutil.which('scp')
    if not scp:raise ValueError('Windows OpenSSH scp unavailable')
    part=target.with_suffix(target.suffix+'.part')
    args=[scp,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2']
    if cfg.get('ssh_key'):args+=['-i',os.path.expandvars(os.path.expanduser(cfg['ssh_key']))]
    args += [f'comma@{host}:/data/media/0/realdata/{route}--{segment}/{ENCODERS[service]}',str(part)]
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (folder/'road-video-fetch-errors.txt').open('ab') as err:
        p=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=err,creationflags=flags);started=time.monotonic()
        try:
            while p.poll() is None:
                check_work(cancel)
                if time.monotonic()-started>120 or (part.exists() and part.stat().st_size>512*1024**2):raise ValueError('Road-video transfer exceeded bounded size/time')
                time.sleep(.1)
            if p.returncode or not part.is_file() or not part.stat().st_size:raise ValueError('Road video unavailable; reconnect comma and retry image review')
            os.replace(part,target)
        finally:
            if p.poll() is None:p.terminate();p.wait(timeout=10)
    return target

def build_images(folder,cfg=None,cancel=None):
    folder=Path(folder)
    if not (folder/'final-state.json').is_file():raise ValueError('Stop recording before retrieving review images')
    cfg=cfg or json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    events=[e for e in rows(folder/'audible-warning-events.jsonl') if e.get('delivery')=='submitted_to_windows'];indexes=rows(folder/'camera-frame-index.jsonl');manifest={'schema_version':1,'events':{},'images':0,'missing':0}
    images=folder/'audible-images';images.mkdir(exist_ok=True);failed_files=set();jobs={}
    import bisect
    groups={s:sorted([r for r in indexes if r.get('service')==s and r.get('route') and type(r.get('timestamp_eof_ns')) is int],key=lambda r:r['timestamp_eof_ns']) for s in ENCODERS}
    for event in events:
        check_work(cancel);event_id=event.get('event_id')
        if not isinstance(event_id,str) or not EVENT.fullmatch(event_id):continue
        pictures=[];native=event.get('comma_request_ns')
        for label,offset in [('before',-2.),('at',0.),('after',2.)]:
            picture={'label':label,'requested_offset_s':offset,'status':'unavailable'};pictures.append(picture)
            if type(native) is not int:picture['reason']='Warning has no bounded comma clock match';continue
            target=native+int(offset*1e9)
            for service in ('roadEncodeIdx','qRoadEncodeIdx'):
                candidates=groups[service]
                if not candidates:continue
                times=[r['timestamp_eof_ns'] for r in candidates];j=bisect.bisect_left(times,target)
                choices=candidates[max(0,j-1):min(len(candidates),j+1)]
                if not choices:continue
                row=min(choices,key=lambda r:abs(r['timestamp_eof_ns']-target));delta=(row['timestamp_eof_ns']-target)/1e9
                if abs(delta)>.25:continue
                key=(row['route'],row['segment'],service)
                if key in failed_files:continue
                try:video=_fetch(folder,cfg,row,cancel)
                except ValueError as e:picture['reason']=str(e);failed_files.add(key);continue
                picture.update(status='pending_decode',native_reference=row,match_delta_s=delta,clock_bound_ns=event.get('clock_bound_ns'),
                               alignment_basis='Native segment presentation index; timestamp match, not independent clock verification')
                dest=images/f'{event_id}-{label}.jpg';jobs.setdefault(str(video),[]).append((row['frame_index'],dest,picture));break
            if picture['status']=='unavailable' and 'reason' not in picture:picture['reason']='No road-camera index within 250 ms; image not guessed'
        manifest['events'][event_id]=pictures
    import av
    for file,requests in jobs.items():
        needed={i for i,_,_ in requests};decoded=set()
        try:
            with av.open(file) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    check_work(cancel)
                    if frame.is_corrupt:raise ValueError('Corrupt video frame; frame-index correspondence uncertain')
                    if i in needed:
                        picture_frame=frame.to_image()
                        for index,dest,picture in requests:
                            if index==i:
                                picture_frame.save(dest,format='JPEG',quality=88)
                                picture.update(status='ready',file='audible-images/'+dest.name)
                        decoded.add(i)
                    if needed<=decoded:break
            for i,dest,picture in requests:
                if i not in decoded:picture.update(status='unavailable',reason='Requested native frame is absent from video');continue
        except (OSError,ValueError,av.error.FFmpegError) as e:
            for _,_,picture in requests:picture.update(status='unavailable',reason='Road video decode failed: '+str(e))
    for pictures in manifest['events'].values():
        for p in pictures:
            manifest['images']+=p['status']=='ready';manifest['missing']+=p['status']!='ready'
    save(folder/'audible-review.json',manifest)
    return {'warnings':len(events),'images':manifest['images'],'missing_images':manifest['missing']}

def label_event(folder,session_id,event_id,label):
    folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    if not (folder/'final-state.json').is_file():raise ValueError('Stop recording before reviewing warnings')
    if session_id!=cfg.get('session_id') or not isinstance(event_id,str) or not EVENT.fullmatch(event_id):raise ValueError('Review session or warning identity changed')
    if label not in ('good','bad','unsure'):raise ValueError('Unknown warning label')
    event=next((r for r in rows(folder/'audible-warning-events.jsonl') if r.get('event_id')==event_id),None)
    if event is None:raise ValueError('Warning event not found')
    if event.get('delivery')!='submitted_to_windows':raise ValueError('This audio attempt was not submitted to the speakers')
    record={'event_id':event_id,'session_id':session_id,'label':label,'reviewed_wall_ns':time.time_ns(),'source':'manual_post_drive_review'}
    with (folder/'audible-warning-labels.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    # Append revisions here; the latest label for an event wins. Do not append
    # contradictory duplicates to the legacy button-label file or auto-labels.
    return {'ok':True,'event_id':event_id,'label':label}

def page(folder,token):
    folder=Path(folder);events=rows(folder/'audible-warning-events.jsonl');labels={r['event_id']:r['label'] for r in rows(folder/'audible-warning-labels.jsonl') if 'event_id' in r}
    try:pictures=json.loads((folder/'audible-review.json').read_text())['events']
    except (OSError,ValueError,KeyError):pictures={}
    session=json.loads((folder/'session.json').read_text(encoding='utf-8-sig')).get('session_id');cards=[]
    for e in events:
        if not isinstance(e.get('event_id'),str) or not EVENT.fullmatch(e['event_id']):continue
        eid=e['event_id'];a=e['alert'];parts=[]
        for label in ('before','at','after'):
            p=next((p for p in pictures.get(eid,[]) if p['label']==label),{})
            url=p.get('file','')
            if p.get('status')=='ready' and re.fullmatch(r'audible-images/[0-9a-f]{32}-(before|at|after)\.jpg',url):
                parts.append(f'<figure><img loading="lazy" src="/{url}"><figcaption>{label}: match offset {p.get("match_delta_s",0)*1000:.0f} ms</figcaption></figure>')
            else:parts.append('<figure><p>'+html.escape(p.get('reason','Road image pending automatic retrieval'))+'</p><figcaption>'+label+'</figcaption></figure>')
        bypass=(e.get('forced_experiment') or {}).get('bypassed_checks',[])
        cards.append('<article><h2>'+html.escape(a.get('phrase','Warning'))+'</h2><time data-ms="'+str(int(e.get('recorded_wall_ns',0))//1_000_000)+'"></time><p>'+html.escape(f"Range {a.get('range_m')} m · TTC {a.get('ttc_s')} s · {e['delivery']}")+'</p><div class="pictures">'+''.join(parts)+'</div><p class="small">Bypassed checks: '+html.escape(', '.join(bypass) or 'none')+'</p><p id="label-'+eid+'">'+html.escape(labels.get(eid,'Not reviewed'))+'</p>'+''.join(f'<button onclick="mark(\'{eid}\',\'{label}\')">{text}</button>' for label,text in [('good','Good warning'),('bad','Bad warning'),('unsure','Unsure')])+'</article>')
    data=json.dumps({'token':token,'session_id':session}).replace('<','\\u003c')
    return '<!doctype html><meta charset="utf-8"><title>Audible warning review</title><style>body{font:17px system-ui;background:#101d29;color:#eaf2f7;margin:28px}article{border:1px solid #526674;border-radius:10px;padding:20px;margin:20px 0}.pictures{display:flex;gap:12px}figure{flex:1;margin:0;min-width:0;background:#203440;padding:8px}img{width:100%}button{font:inherit;padding:10px;margin-right:10px}.small,figcaption{font-size:13px;color:#b8cbd5}</style><h1>Audible warnings — image review</h1><p>Speaker submission is logged; physical audibility is not measured. Pictures are matched automatically from native comma road-video indexes. No video scrubbing required.</p>'+(''.join(cards) or '<p>No audible-warning attempts were recorded.</p>')+'<script>const cfg='+data+';document.querySelectorAll("time[data-ms]").forEach(t=>t.textContent=new Date(Number(t.dataset.ms)).toLocaleString());async function mark(id,label){try{const r=await fetch("/audible-label",{method:"POST",headers:{"Content-Type":"application/json","X-FCW-Token":cfg.token},body:JSON.stringify({session_id:cfg.session_id,event_id:id,label})});const d=await r.json();if(!r.ok)throw Error(d.error);document.getElementById("label-"+id).textContent=d.label;}catch(e){alert(e.message);}}</script>'

"""Bounded frozen-worker qualification with original HEVC, never live Comma/audio.

Fixture timestamps are explicitly synthetic and retimed. This verifies the
native decode/model/identity/pause/stop path, not Wi-Fi transport or detection recall.
"""
import collections,hashlib,json,threading,time,zipfile
from pathlib import Path
import runtime_clock as clock

MODEL_SHA='451353a9cb5d793a2db78dd584d8ebef873960adcfc047cb53eb8cfd4f4fb48d'

def load_fixture(path,expected_sha):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>256*1024*1024:raise ValueError('Missing/oversize native fixture')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:raise ValueError('Native fixture digest mismatch')
    with zipfile.ZipFile(path) as z:
        if set(z.namelist())!={'manifest.json','packets.bin'}:raise ValueError('Unexpected fixture members')
        if z.getinfo('packets.bin').file_size>256*1024*1024 or z.getinfo('manifest.json').file_size>2*1024*1024:raise ValueError('Oversize fixture contents')
        manifest=json.loads(z.read('manifest.json'));payload=z.read('packets.bin')
    if manifest.get('fixture_only') is not True or manifest.get('fps')!=20 or set(manifest.get('streams',{}))!={'narrow','wide'}:raise ValueError('Wrong fixture contract')
    if hashlib.sha256(payload).hexdigest()!=manifest['payload_sha256']:raise ValueError('Packet payload digest mismatch')
    for cam,s in manifest['streams'].items():
        if (s['width'],s['height'])!=(1928,1208) or not 200<=len(s['packets'])<=400 or not s['packets'][0]['keyframe']:raise ValueError('Unsupported fixture stream')
        for i,p in enumerate(s['packets']):
            if p['ordinal']!=i or not 0<p['length']<=8*1024*1024 or not 0<=p['offset']<=len(payload)-p['length']:raise ValueError('Invalid fixture packet index')
    return manifest,payload

def run(config_path,assets):
    from video_processing import VideoProcessing
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    manifest,payload=load_fixture(cfg['fixture'],cfg['fixture_sha256'])
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    log=(out/'events.jsonl').open('w',encoding='utf-8',buffering=1);rows=[];guard=threading.Lock();expected={};errors=[]
    def receive(row):
        item={**row,'selftest_callback_ns':clock.monotonic_ns()}
        with guard:rows.append(item)
        log.write(json.dumps(item,separators=(',',':'),allow_nan=False)+'\n')
    worker=VideoProcessing({'weights':str(Path(assets)/'models/yolo11m.onnx'),'sha256':MODEL_SHA,
                            'provider':'DmlExecutionProvider','aggregate_inference_hz':4.0},receive)
    offered=collections.Counter();accepted=collections.Counter();pause_ns=resume_ns=None;started=clock.monotonic_ns()
    try:
        worker.start();deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            with guard:ready=[r for r in rows if r.get('kind')=='inference_ready'];bad=[r for r in rows if r.get('kind')=='worker_error']
            if bad:raise RuntimeError('Worker initialization error: '+str(bad))
            if ready:break
            time.sleep(.05)
        else:raise RuntimeError('Camera inference readiness timed out')
        if ready[-1].get('provider')!='DmlExecutionProvider' or ready[-1].get('model_sha256')!=MODEL_SHA:raise RuntimeError('Wrong model/provider')
        start=clock.monotonic_ns()+200_000_000
        count=min(len(v['packets']) for v in manifest['streams'].values())
        for ordinal in range(count):
            due=start+ordinal*50_000_000
            wait=(due-clock.monotonic_ns())/1e9
            if wait>0:time.sleep(wait)
            if ordinal==100:
                worker.set_camera_paused(True,'frozen_selftest_fcw_pressure');pause_ns=clock.monotonic_ns()
            if ordinal==140:
                worker.set_camera_paused(False);resume_ns=clock.monotonic_ns()
            for index,cam in enumerate(('narrow','wide')):
                s=manifest['streams'][cam];p=s['packets'][ordinal];native=1_000_000_000_000+ordinal*50_000_000;fid=100000*(index+1)+ordinal
                meta={'kind':'video','service':'roadEncodeData' if cam=='narrow' else 'wideRoadEncodeData',
                      'codec':'fullHEVC','writer':4242+index,'segment_num':6,'encode_id':ordinal,'frame_id':fid,
                      'width':1928,'height':1208,'header_bytes':0,'keyframe':p['keyframe'],
                      'source_ns':due,'native_source_ns':native,'sent_ns':clock.monotonic_ns(),'receipt_ns':clock.monotonic_ns(),
                      'source_interval_ns':[due,due],'clock_bound_ns':0,'native_clock_domain':'explicit-retimed-selftest',
                      'replay_test':True}
                expected[(cam,fid)]=meta;offered[cam]+=1
                accepted[cam]+=bool(worker.on_packet(meta,payload[p['offset']:p['offset']+p['length']]))
        time.sleep(.5)
    except Exception as e:errors.append(type(e).__name__+': '+str(e))
    finally:
        before_stop=worker.snapshot();stop_started=time.monotonic();worker.stop();stop_s=time.monotonic()-stop_started
        log.close()
    observations=[r for r in rows if r.get('kind')=='observations'];fresh=[r for r in observations if r.get('fresh') is True]
    for r in observations:
        m=expected.get((r.get('camera'),r.get('frame_id')))
        if not m or r.get('source_ns')!=m['source_ns'] or r.get('native_source_ns')!=m['native_source_ns']:
            errors.append('Source/frame identity mismatch');break
        if r.get('fresh') and (r.get('consumer_age_ms',float('inf'))>300 or r.get('control_epoch') not in (0,2)):
            errors.append('Fresh result violated age/control-epoch bound');break
    pre={cam:sum(r['camera']==cam and pause_ns is not None and r['selftest_callback_ns']<pause_ns for r in fresh) for cam in ('narrow','wide')}
    post={cam:sum(r['camera']==cam and resume_ns is not None and r['selftest_callback_ns']>resume_ns and r.get('control_epoch')==2 for r in fresh) for cam in ('narrow','wide')}
    during=[r for r in fresh if pause_ns is not None and resume_ns is not None and pause_ns+100_000_000<r['selftest_callback_ns']<resume_ns]
    if during:errors.append('Fresh inference delivered while paused')
    if any(n<3 for n in pre.values()) or any(n<3 for n in post.values()):errors.append('Insufficient fresh per-camera results before/after pause')
    if any(accepted[c]<.98*offered[c] for c in offered):errors.append('Encoded input backpressure exceeds fixture gate')
    worker_errors=[r for r in rows if r.get('kind')=='worker_error']
    if worker_errors:errors.append('Worker emitted error')
    children=[{'name':p.name,'pid':p.pid,'exitcode':p.exitcode,'alive':p.is_alive()} for p in worker.processes]
    if any(c['alive'] or c['exitcode']!=0 for c in children):errors.append('Child did not shut down cleanly')
    if worker.thread is not None and worker.thread.is_alive():errors.append('Result-consumer thread did not shut down')
    def percentiles(values):
        values=sorted(values)
        return {str(p):values[min(len(values)-1,int((len(values)-1)*p/100))] for p in (50,95,99,100)} if values else {}
    result={'status':'passed' if not errors else 'failed','errors':errors,'fixture_sha256':cfg['fixture_sha256'],
            'model_sha256':MODEL_SHA,'offered':dict(offered),'accepted':dict(accepted),'observation_results':len(observations),
            'fresh_results':len(fresh),'fresh_before_pause':pre,'fresh_after_resume':post,'fresh_while_paused':len(during),
            'consumer_age_ms':percentiles([r['consumer_age_ms'] for r in observations]),
            'processing_ms':percentiles([r['processing_ms'] for r in observations]),'worker_errors':worker_errors,
            'pause_ns':pause_ns,'resume_ns':resume_ns,'children':children,'stop_s':stop_s,'before_stop':before_stop,
            'elapsed_s':(clock.monotonic_ns()-started)/1e9,'physical_audio_played':False,'live_comma_access':False,
            'scope':'Frozen original-HEVC decode/DML/motion/identity/consumer/pause/stop fixture; retimed timestamps, no Wi-Fi transport or recall claim'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));return result

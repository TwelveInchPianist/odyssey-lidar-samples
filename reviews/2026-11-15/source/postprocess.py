"""Resumable offline stages; live capture may preempt them at safe boundaries."""
import json
from pathlib import Path
from hindsight import write_json,input_signature
from offline_jobs import WorkPaused,check_work

def session_signature(folder):
    folder=Path(folder);result=input_signature(folder);manual=folder/'labels.jsonl'
    if manual.exists():result['labels.jsonl']=[manual.stat().st_size,manual.stat().st_mtime_ns]
    for path in sorted(folder.glob('route-journal-*.jsonl')):result[path.name]=[path.stat().st_size,path.stat().st_mtime_ns]
    return result

def run_offline(folder,data_root,assets=None,status=None,cancel=None):
    folder=Path(folder);data_root=Path(data_root)
    if not (folder/'final-state.json').exists():raise ValueError('Completed session required')
    cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    from runtime import replay_and_review,APP_VERSION
    from mapping.pipeline import update_atlas as update_map
    from evaluation import summarize
    signature=session_signature(folder)
    try:previous=json.loads((folder/'offline-status.json').read_text())
    except (OSError,ValueError):previous={}
    reusable=previous.get('engine_version')==APP_VERSION and previous.get('source_signature')==signature
    progress={'schema_version':1,'engine_version':APP_VERSION,'source_signature':signature,'state':'running','stages':{}}
    def stage(name,fn,reuse=True):
        if status:status('Offline review: '+name+'…')
        try:
            check_work(cancel)
            old=previous.get('stages',{}).get(name,{})
            if reusable and reuse and old.get('status')=='complete':progress['stages'][name]=old
            else:progress['stages'][name]={'status':'complete','result':fn()}
        except WorkPaused:
            progress['state']='paused';progress['stages'][name]={'status':'paused'}
            write_json(folder/'offline-status.json',progress);return False
        except Exception as e:
            progress['stages'][name]={'status':'failed','error':str(e)}
            if name=='static map':write_json(folder/'map-review.json',{'status':'failed','reason':str(e),'warning_override':False})
            if name=='gated parameter proposal':write_json(folder/'proposal-review.json',{'status':'failed','eligible':False,'paragraph':'Proposal review failed; settings unchanged. '+str(e)})
        write_json(folder/'offline-status.json',progress);return True
    def hindsight():
        report=summarize(folder,recompute_hindsight=False,cancel=cancel)
        if report['hindsight']['status']!='complete':raise RuntimeError(report['hindsight'].get('error','Hindsight unavailable'))
        return {'frames':report['frames']}
    if (folder/'audible-warning-events.jsonl').is_file() and (folder/'audible-warning-events.jsonl').stat().st_size:
        from audible_review import build_images
        if not stage('audible warning images',lambda:build_images(folder,cfg,cancel),reuse=False):return progress
    if not stage('hindsight',hindsight,reuse=False):return progress
    from mapping.trajectory import build_trajectory
    if not stage('route trajectory',lambda:build_trajectory(folder,cancel)):return progress
    if cfg.get('capture_mode')=='live':
        if any(folder.glob('lidar-*.pcap')):
            if cfg.get('self_calibration_enabled',True) and cfg.get('mount_epoch'):
                from selfcal import run_session
                def calibration_progress(value):
                    if status:
                        status(value.get('message') or ('Automatic calibration: '+str(value.get('analyzed_frames',0))+' frames, '+str(value.get('radar_pairs',0))+' radar matches') if isinstance(value,dict) else str(value))
                if not stage('automatic sensor calibration',lambda:run_session(folder,data_root,cancel=cancel,progress=calibration_progress)):return progress
            if not stage('raw replay',lambda:replay_and_review(folder,cancel)):return progress
        if not stage('static map',lambda:update_map(folder,data_root/'maps',cancel)):return progress
        from proposals import propose_after_session
        if not stage('gated parameter proposal',lambda:propose_after_session(folder,data_root,assets,APP_VERSION,status,cancel)):return progress
    if not stage('combined report',lambda:{'frames':summarize(folder,recompute_hindsight=False,cancel=cancel)['frames']},reuse=False):return progress
    progress['state']='complete_with_errors' if any(v['status']=='failed' for v in progress['stages'].values()) else 'complete'
    write_json(folder/'offline-status.json',progress)
    return progress

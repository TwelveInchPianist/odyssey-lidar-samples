"""Read-only installed-cereal observer with compact path data and clock pings.

Run with python -c via SSH, keeping stdin open for clock requests. No publishers,
device writes, package installation, driving-control changes or persistent daemon.
"""
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

try:
    ReadOnlyMessageReader  # Embedded before this bridge by the packaged runtime.
except NameError:
    from comma_reader import ReadOnlyMessageReader, verify_native_layout

KEEP={
 'gpsLocation':('latitude','longitude','altitude','horizontalAccuracy','verticalAccuracy','speed','bearingDeg','bearingAccuracyDeg','speedAccuracy','vNED','hasFix','flags','unixTimestampMillis','satelliteCount','source'),
 'gpsLocationExternal':('latitude','longitude','altitude','horizontalAccuracy','verticalAccuracy','speed','bearingDeg','bearingAccuracyDeg','speedAccuracy','vNED','hasFix','flags','unixTimestampMillis','satelliteCount','source'),
 'gnssMeasurements':('positionECEF','kalmanPositionECEF','velocityECEF','gpsWeek','gpsTimeOfWeek','measTime','timeToFirstFix'),
 'deviceState':('started','thermalStatus'),
 'modelV2':('frameId','frameAge','frameDropPerc','timestampEof','modelExecutionTime','leadsV3','position','orientation',
            'velocity','orientationRate','laneLines','laneLineProbs','laneLineStds','roadEdges','roadEdgeStds','confidence','meta'),
 'carState':('vEgo','vEgoRaw','wheelSpeeds','aEgo','yawRate','steeringAngleDeg','steeringRateDeg','steeringPressed',
             'brakePressed','gasPressed','standstill','gearShifter','canValid','canTimeout','leftBlinker','rightBlinker'),
 'livePose':('orientationNED','velocityDevice','accelerationDevice','angularVelocityDevice','inputsOK','posenetOK','sensorsOK','timestamp'),
 'liveLocationKalman':('angularVelocityCalibrated','velocityCalibrated','calibratedOrientationNED','status','inputsOK','sensorsOK','positionECEF','positionGeodetic','gpsOK','unixTimestampMillis'),
 'liveCalibration':('calStatus','calPerc','validBlocks','rpyCalib','rpyCalibSpread','height'),
 'cameraOdometry':('frameId','timestampEof','trans','rot','transStd','rotStd','wideFromDeviceEuler','roadTransformTrans'),
 # Keep pre-vision OEM targets separate from selected/fused leads. Optional
 # Honda aRel/yvRel NaNs are preserved by sanitize as unknown JSON nulls.
 'liveTracks':('errors','points'),
 'radarState':('mdMonoTime','carStateMonoTime','radarErrors','leadOne','leadTwo'),
 'selfdriveState':('enabled','active','state'),
 'controlsState':('enabled','active','state','curvature','desiredCurvature')
}
KEEP['carParams']=('carFingerprint','brand','radarUnavailable','radarDelay','wheelbase','centerToFront','wheelSpeedFactor')
for camera_service in ('roadCameraState','narrowRoadCameraState','wideRoadCameraState'):
    KEEP[camera_service]=('frameId','timestampSof','timestampEof','frameType')
for encoder_service in ('roadEncodeIdx','qRoadEncodeIdx','wideRoadEncodeIdx'):
    KEEP[encoder_service]=('frameId','segmentNum','segmentId','segmentIdEncode','timestampSof','timestampEof','type')
ENCODE_DATA={'roadEncodeData','qRoadEncodeData','wideRoadEncodeData'}
for encoder_service in ENCODE_DATA:KEEP[encoder_service]=('idx',)

def compact_value(name,value):
    # Current fork publishes EncodeData, while EncodeIdx is written to rlogs.
    # Read only its scalar index: never materialize or transmit encoded video.
    if name in ENCODE_DATA:return {'idx':value.idx.to_dict()}
    return [v.to_dict() for v in value] if name=='sensorEvents' else value.to_dict()

def sanitize(value,path='',missing=None):
    """JSON null means unknown/nonfinite, never a fabricated numeric zero."""
    if missing is None:missing=[]
    if isinstance(value,dict):return {k:sanitize(v,path+'.'+k,missing) for k,v in value.items()}
    if isinstance(value,list):return [sanitize(v,path+'['+str(i)+']',missing) for i,v in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):missing.append(path);return None
    return value

def main(reader_factory=None):
    os.environ.pop('ZMQ',None)
    sys.path.insert(0,'/data/openpilot');sys.path.insert(0,'/data')
    errors=[]
    for prefix in ('cereal','openpilot.cereal'):
        try:
            messaging=importlib.import_module(prefix+'.messaging')
            registry_module=importlib.import_module(prefix+'.services')
            registry=getattr(registry_module,'SERVICE_LIST',getattr(registry_module,'service_list',{}))
            break
        except ImportError as e:errors.append(str(e))
    else:raise RuntimeError('Installed cereal unavailable: '+'; '.join(errors))
    wanted=list(KEEP)+['accelerometer','gyroscope','sensorEvents','deviceMotion','extrinsicsCalibration']
    services=[s for s in wanted if s in registry]
    if reader_factory is None:
        import msgq.ipc_pyx
        native_digest=verify_native_layout(msgq.ipc_pyx)
        sm=ReadOnlyMessageReader(services,messaging,registry)
        transport='msgq_read_only_snapshot_v1'
    else:
        sm=reader_factory(services,messaging,registry)
        native_digest=None;transport='injected_test_reader'
    lock=threading.Lock();stop=threading.Event()
    def emit(item):
        with lock:print(json.dumps(item,separators=(',',':'),allow_nan=False),flush=True)
    def ping_reader():
        try:
            for line in sys.stdin:
                t1=time.monotonic_ns()
                if len(line)>1000:continue
                try:
                    p=json.loads(line)
                    if p.get('kind')=='ping' and isinstance(p.get('id'),int):
                        emit({'kind':'pong','id':p['id'],'remote_receive_ns':t1,'remote_send_ns':time.monotonic_ns()})
                except (ValueError,TypeError):pass
        finally:stop.set()
    try:
        commit=subprocess.check_output(['git','-C','/data/openpilot','rev-parse','HEAD'],stderr=subprocess.DEVNULL,timeout=3).decode().strip()
    except Exception:commit='unknown'
    try:
        git_dirty=bool(subprocess.check_output(['git','--no-optional-locks','-C','/data/openpilot','status','--porcelain','--untracked-files=normal'],stderr=subprocess.DEVNULL,timeout=3).strip())
    except Exception:git_dirty=None
    emit({'kind':'hello','schema_version':4,'radar_contract_version':1,'services':services,'commit':commit,'bridge_hz':50,
          'git_dirty':git_dirty,
          'transport':transport,'msgq_native_sha256':native_digest,
          'subscriber_slots_used':0 if transport=='msgq_read_only_snapshot_v1' else None,
          'location_services_registered':[s for s in services if s in ('gpsLocation','gpsLocationExternal','gnssMeasurements','liveLocationKalman')],
          'radar_services_registered':[s for s in services if s in ('liveTracks','radarState')],
          'radar_note':'liveTracks is OEM radar targets before vision association; radarState may contain vision-only leads. Publication time is not verified sensor measurement time.',
          'location_note':'Registry presence does not establish a running publisher or a valid GPS fix.',
          'python_executable':sys.executable,
          'sampling':'latest-value <=50 Hz; path retains source camera timestamp',
          'comma_wall_ns':time.time_ns(),'comma_mono_ns':time.monotonic_ns()})
    threading.Thread(target=ping_reader,daemon=True).start()
    last_heartbeat=0;last_route=None
    while not stop.is_set():
        start=time.monotonic();sm.update(0)
        for name in services:
            if not sm.updated[name]:continue
            value=sm[name]
            data=compact_value(name,value)
            if name in KEEP:data={k:data[k] for k in KEEP[name] if k in data}
            try:
                missing=[];data=sanitize(data,missing=missing)
                emit({'kind':'sample','service':name,'logMonoTime':int(sm.logMonoTime[name]),
                      'valid':bool(sm.valid[name]),'comma_mono_ns':time.monotonic_ns(),
                      'comma_wall_ns':time.time_ns(),'data':data,
                      'nonfinite_count':len(missing),'nonfinite_paths':missing[:16]})
            except ValueError:emit({'kind':'invalid_sample','service':name,'reason':'nonfinite data'})
        if start-last_heartbeat>1:
            emit({'kind':'heartbeat','comma_mono_ns':time.monotonic_ns(),
                  'reader_stats':{k:dict(v.stats) for k,v in getattr(sm,'rings',{}).items()}});last_heartbeat=start
            try:route=Path('/data/params/d/CurrentRoute').read_text().strip()
            except OSError:route=None
            if route and route!=last_route:
                emit({'kind':'route_reference','route':route,'comma_mono_ns':time.monotonic_ns(),
                      'note':'Existing comma recording reference only; no video transmitted by this bridge'})
                last_route=route
        stop.wait(max(0,.02-(time.monotonic()-start)))
    if hasattr(sm,'close'):sm.close()

if __name__=='__main__':
    try:main()
    except (BrokenPipeError,KeyboardInterrupt):pass

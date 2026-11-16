"""Independent read-only SSH video channel with measured source-age bounds."""
import base64,collections,json,os,re,shutil,subprocess,threading,zlib
from pathlib import Path
import runtime_clock as clock
from telemetry import ClockSync
from video_transport import read_message


def server_program(assets):
    assets=Path(assets)
    modules={name:(assets/(name+'.py')).read_text(encoding='utf-8')
             for name in ['comma_reader','video_transport','comma_video_bridge']}
    packed=base64.b64encode(zlib.compress(json.dumps(modules).encode('utf-8'))).decode('ascii')
    source=("import base64,json,sys,types,zlib\nsys.dont_write_bytecode=True\n"
            "modules=json.loads(zlib.decompress(base64.b64decode('"+packed+"')))\n"
            "for name,source in modules.items():\n"
            " module=types.ModuleType(name);sys.modules[name]=module;exec(compile(source,name,'exec'),module.__dict__)\n"
            "sys.modules['comma_video_bridge'].main()\n")
    return base64.b64encode(zlib.compress(source.encode('utf-8'))).decode('ascii')


def command(config,assets):
    host=config['comma_host'];python=config.get('comma_python','auto')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',host):raise ValueError('Invalid comma host')
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',python) or python.startswith('-'):raise ValueError('Invalid comma interpreter')
    selection=''
    if python in ('auto','python3'):
        selection='FCW_VIDEO_PYTHON=python3; if [ -x /usr/local/venv/bin/python ]; then FCW_VIDEO_PYTHON=/usr/local/venv/bin/python; fi; '
        python='"$FCW_VIDEO_PYTHON"'
    payload=server_program(assets)
    remote=(f'cd /data/openpilot && ( {selection}exec nice -n 15 env PYTHONDONTWRITEBYTECODE=1 '
            f'PYTHONPATH=/data/openpilot:/data {python} -u -c '
            "'import base64,zlib;exec(zlib.decompress(base64.b64decode(\""+payload+"\")))' )")
    ssh=shutil.which('ssh')
    if not ssh:raise RuntimeError('OpenSSH client unavailable')
    args=[ssh,'-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',
          '-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2']
    key=config.get('ssh_key')
    if key:args+=['-i',os.path.expanduser(os.path.expandvars(key))]
    return [*args,'comma@'+host,remote]


class VideoClient:
    def __init__(self,config,assets,on_packet,on_status=None):
        self.args=command(config,assets);self.on_packet=on_packet;self.on_status=on_status or (lambda _:None)
        self.stop_event=threading.Event();self.process=None;self.thread=None
        self.metrics=collections.Counter();self.clock=ClockSync();self.pending={};self.lock=threading.Lock()
        self.last_error=None;self.hello=None

    def start(self):
        if self.thread and self.thread.is_alive():raise RuntimeError('Video client already running')
        self.thread=threading.Thread(target=self._run,daemon=True,name='awareness-video-receiver');self.thread.start()

    def stop(self):
        self.stop_event.set()
        p=self.process
        if p and p.poll() is None:p.terminate()
        if self.thread:self.thread.join(timeout=4)

    def _ping(self,p):
        index=0
        while not self.stop_event.is_set() and p.poll() is None:
            index+=1
            with self.lock:
                self.pending[index]=clock.monotonic_ns()
                if len(self.pending)>120:self.pending.pop(min(self.pending))
            try:p.stdin.write((json.dumps({'kind':'ping','id':index})+'\n').encode());p.stdin.flush()
            except (BrokenPipeError,OSError,ValueError):return
            if self.stop_event.wait(.1 if index<8 else 1.):return

    def _run(self):
        delay=2
        while not self.stop_event.is_set():
            self.clock=ClockSync();self.pending={};self.hello=None
            try:
                p=subprocess.Popen(self.args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.process=p
                errors=[]
                def drain():
                    for line in p.stderr:
                        errors.append(line.decode('utf-8',errors='replace').strip())
                        del errors[:-10]
                threading.Thread(target=drain,daemon=True).start()
                ping=threading.Thread(target=self._ping,args=(p,),daemon=True);ping.start()
                while not self.stop_event.is_set():
                    message=read_message(p.stdout);receipt=clock.monotonic_ns()
                    if message is None:break
                    meta,payload=message;kind=meta.get('kind')
                    if kind=='hello':
                        if (meta.get('schema_version')!=1 or meta.get('subscriber_slots_used')!=0
                                or meta.get('source_clock')!='linux_boottime'):
                            raise ValueError('Unexpected camera bridge contract')
                        self.hello=meta;self.on_status({'state':'connected_waiting_for_camera','hello':meta});delay=2
                    elif kind=='pong':
                        with self.lock:
                            sent=self.pending.pop(meta.get('id'),None)
                            if sent is not None:self.clock.add(sent,meta['remote_receive_ns'],meta['remote_send_ns'],receipt)
                    elif kind=='video':
                        if not self.hello or meta.get('replay_test'):raise ValueError('Unbound or replay video on live channel')
                        self.metrics['encoded_frames']+=1;self.metrics['encoded_bytes']+=len(payload)
                        with self.lock:estimate=self.clock.estimate(receipt)
                        if estimate is None or estimate[1]>40_000_000:
                            self.metrics['clock_unqualified']+=1;continue
                        offset,bound=estimate;earliest=int(meta['source_ns']+offset-bound)
                        latest=int(meta['source_ns']+offset+bound)
                        if earliest>receipt or receipt-earliest>250_000_000:
                            self.metrics['source_too_old_or_future']+=1;continue
                        local={**meta,'native_source_ns':meta['source_ns'],'source_ns':earliest,
                               'source_interval_ns':[earliest,latest],'receipt_ns':receipt,
                               'clock_bound_ns':int(bound),'native_clock_domain':'linux_boottime'}
                        # The consumer must enqueue without blocking the reader.
                        if self.on_packet(local,payload) is False:self.metrics['consumer_backpressure']+=1
                        else:self.metrics['accepted_frames']+=1
                    elif kind=='heartbeat':
                        with self.lock:estimate=self.clock.estimate(receipt)
                        self.on_status({'state':'receiving' if self.metrics['accepted_frames'] else 'waiting_for_camera',
                            'metrics':dict(self.metrics),'clock_bound_ms':estimate[1]/1e6 if estimate else None,'rings':meta.get('rings',{})})
                if p.poll() is None:p.terminate()
                p.wait(timeout=5);ping.join(timeout=2)
                if not self.stop_event.is_set():self.last_error='; '.join(errors)[-1000:] or 'camera connection ended'
            except Exception as error:
                self.last_error=type(error).__name__+': '+str(error)
                if self.process and self.process.poll() is None:self.process.terminate()
            finally:
                if self.process:
                    for stream in [self.process.stdin,self.process.stdout,self.process.stderr]:
                        if stream:
                            try:stream.close()
                            except OSError:pass
                self.process=None
            if not self.stop_event.is_set():
                self.on_status({'state':'reconnecting','reason':self.last_error})
                if self.stop_event.wait(delay):break
                delay=min(15,delay*2)

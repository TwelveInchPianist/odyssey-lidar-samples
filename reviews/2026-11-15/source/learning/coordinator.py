"""Idle-only optional work with capture-priority cancellation and bounded status.

quiesce()/close() MUST be called outside Application.lock. The start operation
reserves capture first. Worker tree waits occur only under coordinator.lock;
GUI snapshot reads an immutable bounded cache without taking that lock.
"""
import copy, json, math, os, shutil, sys, threading, uuid
import runtime_clock as time
from pathlib import Path
from offline_jobs import WorkPaused
from .jobs import JobStore,current_owner
from .resources import BoundedProcess
from .pipeline import VERSION,IDLE_RESERVE_BYTES,NO_CLAIM,write_launch_spec
from .catalog import Catalog,VERSION as CATALOG_VERSION,inventory_cache_key

def _bounded(value,budget=None,depth=0):
    budget=[24000] if budget is None else budget
    if budget[0]<=0 or depth>5:return '[details in durable job record]'
    if value is None or isinstance(value,(bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,str):
        text=value[:min(1200,max(0,budget[0]))];budget[0]-=len(text);return text
    if isinstance(value,dict):
        result={}
        for i,(k,v) in enumerate(value.items()):
            if i>=24 or budget[0]<=0:result['details_omitted']=True;break
            key=str(k)[:100];budget[0]-=len(key);result[key]=_bounded(v,budget,depth+1)
        return result
    if isinstance(value,(list,tuple)):
        result=[]
        for v in value[:20]:
            if budget[0]<=0:break
            result.append(_bounded(v,budget,depth+1))
        if len(value)>len(result):result.append({'items_omitted':len(value)-len(result)})
        return result
    return str(type(value).__name__)

class Coordinator:
    def __init__(self,app,assets,engine_version,*,process_factory=BoundedProcess,allowed_stages=None):
        if allowed_stages is not None and frozenset(allowed_stages)!=frozenset({'self_calibration'}):raise ValueError('Only bounded calibration may be allowlisted')
        self.allowed_stages=None if allowed_stages is None else frozenset(allowed_stages)
        self.app,self.assets,self.engine_version=app,Path(assets).resolve(),engine_version
        self.root=app.record_root.parent.resolve();self.work=self.root/'learning/worker';self.work.mkdir(parents=True,exist_ok=True)
        self.store=JobStore(self.root/'learning/jobs.sqlite');self.process_factory=process_factory
        self.lock=threading.RLock();self.capture_requested=threading.Event();self.closed=False
        self.process=None;self.job_id=None;self.cancel_file=None;self.last_scan=0.
        self.cached={'state':'idle','counts':{},'warning_override':False}
        self.last_result=None;self.last_promotion=None;self._seen_promotions=set()
    def snapshot(self):return copy.deepcopy(self.cached)
    def _idle(self):return not (self.closed or self.capture_requested.is_set() or self.app.activity() or self.app.closing)
    def resume_idle(self):
        with self.lock:
            if self.closed or self.app.activity() or self.app.closing:return
            self.capture_requested.clear();self.last_scan=0.
    def _settle_owned(self,intentional=False):
        """Caller owns coordinator.lock, never Application.lock."""
        if not self.process:return
        process=self.process;identifier=self.job_id;exit_code=process.poll()
        if intentional:
            process.wait(.15)
            process.stop(timeout_s=3)
        process.close(timeout_s=3)  # verifies entire tree, even after root exit
        self.process=None;self.job_id=None;self.cancel_file=None
        job=self.store.get(identifier) if identifier else None
        if job and not intentional:
            reason='worker_process_exit_'+str(exit_code)+'_without_committed_result'
            if job['state']=='running':self.store.fail_dead_owner(identifier,job['lease_token'],reason)
            elif job['state'] in ('pending','paused') or job['state']=='blocked' and job['reason']=='dependencies':
                if exit_code!=NO_CLAIM:self.store.fail_dispatch(identifier,reason)
        self.store.recover_abandoned()
        if identifier:self._completed(self.store.get(identifier))
        self.last_scan=0.
    def quiesce(self):
        """Successful return proves owned worker tree stopped; no app-lock wait."""
        self.capture_requested.set()
        # Set the current child's cancellation flag before waiting on a tick.
        path=self.cancel_file
        if path:path.touch(exist_ok=True)
        with self.lock:
            if self.cancel_file:self.cancel_file.touch(exist_ok=True)
            self._settle_owned(intentional=True)
            recovery=self.store.recover_abandoned()
            if recovery['live'] or recovery['unknown']:raise RuntimeError('An existing offline worker has not been proven stopped')
            self.cached=_bounded({**self.cached,'state':'paused_for_capture','worker':None})
    def close(self):self.closed=True;self.quiesce()
    def _refresh(self,state=None):
        catalog=({'state':'offloaded'} if self.allowed_stages is not None else Catalog(self.root).summary() if (self.root/'learning/catalog.sqlite').exists() else {'sessions':0,'known_comma_route_groups':0,'unknown_route_sessions':0})
        allowed_checkpoint={'message','phase','frames','packets','rows','examples','samples','stage','completed','total','context_bytes','pcap_index','pcap_offset','source_index','sources_total','analyzed_frames','radar_pairs','independent_windows'}
        jobs=[]
        for row in self.store.recent(20,include_details=False):
            item={k:row.get(k) for k in ('id','stage','state','reason','updated')}
            cp=row.get('checkpoint') or {}
            item['checkpoint']=_bounded({k:v for k,v in cp.items() if k in allowed_checkpoint},[800]) if isinstance(cp,dict) else {}
            jobs.append(item)
        value={'state':state or ('working' if self.process else 'idle'),'allowed_stages':sorted(self.allowed_stages) if self.allowed_stages is not None else None,'counts':self.store.counts(),
               'catalog':catalog,'worker':{'job_id':self.job_id,'limits':self.process.limits()} if self.process else None,
               'latest_result':self.last_result,'promotion':self.last_promotion,'warning_override':False,
               'note':('Only bounded self-calibration runs locally; heavy analysis and training are offloaded.' if self.allowed_stages is not None else 'Radar training predicts future radar measurements, not collision truth. Live safety gates are unchanged.'),'jobs':jobs}
        self.cached=_bounded(value)
    def _schedule(self):
        from .datasets import VERSION as DATASET_VERSION
        for folder in sorted(self.app.record_root.iterdir()) if self.app.record_root.exists() else []:
            if not self._idle():return
            if not folder.is_dir() or folder.is_symlink() or getattr(folder,'is_junction',lambda:False)() or not (folder/'final-state.json').is_file():continue
            try:
                path=folder/'session.json'
                if path.stat().st_size>1_000_000:continue
                cfg=json.loads(path.read_text(encoding='utf-8-sig'))
                if cfg.get('synthetic_fixture') or cfg.get('capture_mode') not in ('live','comma'):continue
                signature={'inventory':inventory_cache_key(folder,lambda:not self._idle()),'engine_version':self.engine_version};payload={'session':folder.name}
                # Current mounting data gets first useful processing; a large
                # historical replay backlog must not delay ordinary-drive setup.
                epoch=self.app.config.get('mount_epoch')
                if (cfg.get('capture_mode')=='live' and epoch and cfg.get('mount_epoch')==epoch and
                    self.app.config.get('self_calibration_enabled',True) and any(folder.glob('lidar-*.pcap'))):
                    from selfcal import VERSION as SELFCAL_VERSION
                    self.store.enqueue('self_calibration',SELFCAL_VERSION,signature,payload,priority=5)
                if self.allowed_stages is not None:continue  # No heavy jobs are enqueued in calibration-only mode.
                if (folder/'audible-warning-events.jsonl').is_file() and (folder/'audible-warning-events.jsonl').stat().st_size:
                    self.store.enqueue('audible_images',VERSION,signature,payload,priority=1)
                catalog=self.store.enqueue('catalog',CATALOG_VERSION,signature,payload,priority=10)
                self.store.enqueue('radar_dataset',DATASET_VERSION,signature,payload,(catalog,),priority=20)
                route=self.store.enqueue('route_trajectory',VERSION,signature,payload,(catalog,),priority=40)
                prerequisites=(catalog,)
                if cfg.get('capture_mode')=='live' and any(folder.glob('lidar-*.pcap')):
                    replay=self.store.enqueue('raw_replay',VERSION,signature,payload,(catalog,),priority=90);prerequisites=(replay,)
                self.store.enqueue('review',VERSION,signature,payload,prerequisites,priority=100)
                if cfg.get('capture_mode')=='live':self.store.enqueue('atlas',VERSION,signature,payload,(route,),priority=50)
            except WorkPaused:return
            except (OSError,ValueError) as error:self.last_result={'stage':'inventory','session':folder.name,'error':str(error)[:1000]}
        if not self._idle():return
        if self.allowed_stages is not None:return
        from .training import input_signature
        self.store.enqueue('radar_training',VERSION,input_signature(self.root,self.assets,self.engine_version),priority=30)
    def _completed(self,job):
        if not job:return
        if self.allowed_stages is not None and job['stage'] not in self.allowed_stages:return
        self.last_result={'stage':job['stage'],'state':job['state'],'reason':job['reason']}
        if job['state']=='complete':
            completed=getattr(self.app,'offline_completed',None)
            if callable(completed):
                try:completed(job)
                except Exception as error:self.last_result['display_refresh_error']=str(error)[:1000]
        if job['stage']!='radar_training' or job['state']!='complete':return
        result=job.get('result') or {}
        public_fields={'state','qualified','model_id','report_id','reasons','metrics','counts','objective','trip_count','training_examples','evaluation_examples','example_counts','independent_trips'}
        self.last_result.update(_bounded({k:v for k,v in result.items() if k in public_fields},[5000]))
        if job['id'] in self._seen_promotions or not result.get('qualified') or not self.app.config.get('learning',{}).get('automatic_promotion',True):return
        from .training import check_promotion_identity
        from .model_registry import ModelRegistry
        with self.app.lock:
            if not self._idle():return  # retry completed candidate after capture
            try:
                registry=ModelRegistry(self.root/'learning/radar-models')
                if registry.active_id==result.get('model_id'):
                    self.last_promotion={'activated':False,'already_active':True,'model_id':result['model_id']}
                else:
                    proof=check_promotion_identity(self.root,self.assets,self.engine_version,result)
                    self.last_promotion=registry.activate(result['model_id'],result['report_id'],can_activate=self._idle,engine_version=self.engine_version,**proof)
                self._seen_promotions.add(job['id'])
            except (OSError,ValueError) as error:
                self.last_promotion={'activated':False,'reason':str(error)[:1000],'job_id':job['id']}
                self._seen_promotions.add(job['id'])
        if len(self._seen_promotions)>1000:self._seen_promotions={job['id']}
    def _promote_completed(self):
        if self.allowed_stages is not None:return
        for job in self.store.completed('radar_training',20):
            if not self._idle():return
            if job['id'] not in self._seen_promotions and (job.get('result') or {}).get('qualified'):
                self._completed(job)
                break
    def tick(self):
        with self.lock:
            if not self._idle():return
            if self.process:
                if self.process.poll() is None:self._refresh();return
                self._settle_owned()
            recovery=self.store.recover_abandoned()
            if recovery['live'] or recovery['unknown']:self._refresh('waiting_for_existing_worker');return
            reserve=max(IDLE_RESERVE_BYTES,self.app.config.get('minimum_free_gb',10)*1e9+512*1024*1024)
            if shutil.disk_usage(self.root).free<=reserve:self._refresh('waiting_for_disk_reserve');return
            for job in self.store.blocked('disk_reserve',100):self.store.retry(job['id'],'disk_reserve_restored')
            if time.monotonic()-self.last_scan>30:
                self._schedule();self.last_scan=time.monotonic()
            if not self._idle():return
            self._promote_completed()
            if not self._idle():return
            ready=(self.store.runnable(1,stages=self.allowed_stages) if self.allowed_stages is not None else self.store.runnable(1))
            if not ready:self._refresh();return
            job=ready[0]
            if self.allowed_stages is not None and job['stage'] not in self.allowed_stages:raise RuntimeError('Disallowed background stage')
            run_id=uuid.uuid4().hex
            self.cancel_file=self.work/(run_id+'.cancel');spec_path=self.work/(run_id+'.json')
            memory=self.app.config.get('learning',{}).get('memory_limit_mb',1024)
            cpu=min(2000,max(100,10000//max(1,os.cpu_count() or 1)))
            if type(memory) is not int or not 128<=memory<=1024:
                self.last_result={'stage':'configuration','error':'Worker memory budget must be128-1024MiB'};self._refresh('invalid_worker_configuration');return
            spec={'pipeline_version':VERSION,'run_id':run_id,'parent_owner':current_owner(),'job_id':job['id'],
                  'data_root':str(self.root),'record_root':str(self.app.record_root.resolve()),'assets':str(self.assets),
                  'engine_version':self.engine_version,'cancel_file':str(self.cancel_file),'budget':{'memory_mb':memory,'cpu_rate':cpu}}
            command=[sys.executable]
            if not getattr(sys,'frozen',False):command.append(str(Path(__file__).resolve().parents[1]/'app.py'))
            command+=['--learning-worker',str(spec_path)]
            try:
                auth=write_launch_spec(spec_path,spec)
                if not self._idle():self.cancel_file.touch(exist_ok=True);self._refresh('paused_for_capture');return
                self.job_id=job['id']
                self.process=self.process_factory(command,self.work/(run_id+'.log'),memory_mb=memory,cpu_rate=cpu,
                                                   environment=auth,job_name='Local\\OdysseyFCW-learning-'+run_id)
            except Exception as error:
                self.store.fail_dispatch(job['id'],'worker_launch_failed: '+str(error)[:1000])
                self.process=None;self.job_id=None;self.cancel_file=None
                self.last_result={'stage':job['stage'],'state':'failed','reason':str(error)[:1000]};self._refresh('worker_launch_failed');return
            if not self._idle():
                self.cancel_file.touch(exist_ok=True);self._settle_owned(intentional=True);self._refresh('paused_for_capture');return
            self._refresh()

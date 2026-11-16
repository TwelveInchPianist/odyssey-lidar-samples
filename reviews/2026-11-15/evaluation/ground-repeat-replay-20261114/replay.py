import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,copy,gzip,hashlib,time,collections
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--pack',required=True);ap.add_argument('--out',required=True);ap.add_argument('--jobs');a=ap.parse_args();src=Path(a.source).resolve();pack=Path(a.pack).resolve();out=Path(a.out);out.mkdir(exist_ok=False);sys.path.insert(0,str(src))
import numpy as np,zstandard
from fcw_engine import Engine,Mount,Ground
from forced_experiment import ForcedExperiment
from frames import decode_with_time
def enc(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
fit=Ground.fit
def traced(self,p):
 prior={'stable':self.stable,'plane':None if self.plane is None else self.plane.copy(),'miss':dict(self.miss_state)};r=fit(self,p);self.debug={'before':prior,'after':{'stable':self.stable,'plane':self.plane,'miss':dict(self.miss_state)},'fit':r,'reason':self.last_reason};return r
Ground.fit=traced
summary=[];D=zstandard.ZstdDecompressor()
for j in json.loads((pack/'MANIFEST.json').read_text())['jobs']:
 if a.jobs and j['name'] not in a.jobs.split(','):continue
 data=json.loads((pack/j['metadata']).read_text());cfg=data['config'];engine=Engine(Mount(**cfg['mount']),policy=cfg.get('perception_policy'),forced_experiment=ForcedExperiment(cfg['forced_experiment'],cfg['mount_epoch']));alerts=[];reasons=collections.Counter();timing=[];status=collections.Counter()
 with gzip.open(out/(j['name']+'.jsonl.gz'),'wt',encoding='utf-8') as f:
  for row in data['frames']:
   raw=D.decompress((pack/row['file']).read_bytes());assert hashlib.sha256(raw).hexdigest()==row['raw_sha256'];p,t=decode_with_time([raw[i:i+row['packet_size']] for i in range(0,len(raw),row['packet_size'])]);ctx=copy.deepcopy(row['recorded']['context']);ctx['path']=np.asarray(ctx['path']).reshape(-1,2)
   for k in ['radar_points','radar_publication_ns','radar_publication_age_s','radar_clock_bound_s','radar_frame_native']:
    if k in row.get('raw_context',{}):ctx[k]=copy.deepcopy(row['raw_context'][k])
   q=copy.deepcopy(row['recorded']['quality']);start=time.perf_counter();r=engine.update(p,ctx,q,(t-t.max())/1e9);timing.append((time.perf_counter()-start)*1000);r['ground_debug']=copy.deepcopy(engine.ground.debug);r['input_line']=row['recorded']['original_line'];r['speed_mps']=ctx.get('speed_mps');r['yaw_rate_rps']=ctx.get('yaw_rate_rps');r['ground_held']=copy.deepcopy(engine.held);f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');reasons.update(r['reasons']);status[r['status']]+=1
   if r['alerts']:alerts.append({'at_ns':r['at_ns'],'alerts':r['alerts']})
 result={'job':j['name'],'frames':len(data['frames']),'alerts':alerts,'reasons':dict(reasons),'statuses':dict(status),'engine_ms_p50':float(np.median(timing)),'engine_ms_p95':float(np.percentile(timing,95))};summary.append(result);print(json.dumps(result,default=enc),flush=True);(out/'PROGRESS.json').write_text(json.dumps(summary,default=enc))
(out/'RESULT.json').write_text(json.dumps(summary,indent=2,default=enc))

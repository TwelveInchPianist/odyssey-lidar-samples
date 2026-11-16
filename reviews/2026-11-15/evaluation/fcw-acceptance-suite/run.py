"""Reusable offline gate. No installation, network access or audio playback."""
import argparse,json,hashlib,subprocess,sys,time,gzip,collections
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);ap.add_argument('--reuse-frozen-baseline',action='store_true');args=ap.parse_args()
source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(exist_ok=False);manifest=json.loads((P/'MANIFEST.json').read_text())
def fingerprint():return {str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()for f in source.rglob('*.py')if '__pycache__'not in f.parts}
before=fingerprint();(out/'SOURCE.json').write_text(json.dumps(before,indent=2))
probe=subprocess.run([sys.executable,str(P/'synthetic_checks.py'),str(source)],capture_output=True,text=True,timeout=60)
(out/'synthetic.stdout').write_text(probe.stdout);(out/'synthetic.stderr').write_text(probe.stderr)
try:synthetic=json.loads(probe.stdout)
except ValueError:synthetic={'passed':False,'error':'probe failed; see stdout/stderr'}
results={};readiness={}
for name,item in manifest['packs'].items():
 pack=W/item['relative_path'];assert hashlib.sha256((pack/'MANIFEST.json').read_bytes()).hexdigest()==item['manifest_sha256']
 for file,digest in item['metadata_sha256'].items():assert hashlib.sha256((pack/file).read_bytes()).hexdigest()==digest
 if args.reuse_frozen_baseline:
  for file,digest in manifest['baseline_core_sha256'].items():assert hashlib.sha256((source/file).read_bytes()).hexdigest()==digest,'Cannot reuse a changed baseline'
  result=W/item['baseline_result'];assert hashlib.sha256(result.read_bytes()).hexdigest()==item['baseline_result_sha256'];rows=json.loads(result.read_text());trace_root=result.parent
  for filename,digest in item['baseline_trace_sha256'].items():assert hashlib.sha256((trace_root/filename).read_bytes()).hexdigest()==digest
 else:
  destination=out/name
  with (out/(name+'.log')).open('w')as log:
   process=subprocess.Popen([sys.executable,str(W/'perception-iteration-20261109/replay.py'),'--source',str(source),'--pack',str(pack),'--out',str(destination)],stdout=log,stderr=subprocess.STDOUT)
   (out/'PROGRESS.json').write_text(json.dumps({'pack':name,'pid':process.pid,'started_ns':time.time_ns(),'deadline_s':900}))
   try:code=process.wait(timeout=900)
   except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5);raise
   assert code==0,'Replay failed; see log'
  rows=json.loads((destination/'RESULT.json').read_text());trace_root=destination
 unique={}
 for row in rows:
  with gzip.open(trace_root/(row['job']+'.jsonl.gz'),'rt')as stream:frames=[json.loads(line)for line in stream]
  warm=frames[0]['at_ns']+2_000_000_000
  for first,second in zip(frames,frames[1:]):
   if first['at_ns']<warm:continue
   unique.setdefault(first['at_ns'],{'dt':min(1.,max(0.,(second['at_ns']-first['at_ns'])/1e9)),
       'unavailable':first['status']=='not_ready','settling':'ground estimate settling'in first['reasons'],'reasons':first['reasons']})
 readiness[name]={'unique_frames_after_2s_warmup':len(unique),'evaluated_seconds':sum(r['dt']for r in unique.values()),
   'unavailable_seconds':sum(r['dt']for r in unique.values()if r['unavailable']),
   'settling_seconds':sum(r['dt']for r in unique.values()if r['settling']),
   'overlapping_reason_frames':dict(collections.Counter(reason for r in unique.values()for reason in r['reasons']))}
 results[name]=rows
assert before==fingerprint(),'Source changed while testing'
checks=[]
for case in manifest['positive_controls']:
 row=next(r for r in results[case['pack']]if r['job']==case['job'])
 alerts=[{'at_ns':a['at_ns'],'phrase':v['phrase'],'range_m':v['range_m'],'ttc_s':v['ttc_s']}for a in row['alerts']for v in a['alerts']]
 checks.append({'case':case['name'],'pass':alerts==case['expected'],'observed':alerts})
summaries={name:[{'job':r['job'],'frames':r['frames'],'alert_frames':len(r['alerts'])}for r in rows]for name,rows in results.items()}
result={'hard_controls_pass':all(x['pass']for x in checks)and synthetic.get('passed',False),'positive_controls':checks,'synthetic':synthetic,'metrics':summaries,'readiness':readiness,
 'reused_baseline':args.reuse_frozen_baseline,'coverage':'Recorded FCW positives/readiness/nuisance windows plus synthetic failure gates. Pedestrian/cyclist/animal artifacts are retained as review controls, not automated recall assertions.',
 'automatic_deployment_authorized':False,'physical_audio_tested':False,'onroad_validation':False}
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(0 if result['hard_controls_pass']else 1)

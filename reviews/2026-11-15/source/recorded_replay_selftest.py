"""Explicit offline qualification only; no capture sockets or speaker output."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,zstandard
from fcw_engine import Engine,Mount
from forced_experiment import ForcedExperiment
from inference_worker import EngineProcess
from frames import decode_with_time

def run(config_path):
    config=json.loads(Path(config_path).read_text());out=Path(config['output']);out.mkdir(exist_ok=False)
    results=[];D=zstandard.ZstdDecompressor()
    for spec in config['packs']:
        pack=Path(spec['path']);manifest=pack/'MANIFEST.json'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==spec['manifest_sha256']
        for job in json.loads(manifest.read_text())['jobs']:
            if spec.get('jobs') and job['name'] not in spec['jobs']:continue
            data=json.loads((pack/job['metadata']).read_text());cfg=data['config'];engine=Engine(Mount(**cfg['mount']),policy=cfg.get('perception_policy'),forced_experiment=ForcedExperiment(cfg['forced_experiment'],cfg['mount_epoch']))
            worker=EngineProcess(engine);rows=[];worker.wait_ready()
            try:
                for row in data['frames']:
                    raw=D.decompress((pack/row['file']).read_bytes());assert hashlib.sha256(raw).hexdigest()==row['raw_sha256']
                    points,stamps=decode_with_time([raw[i:i+row['packet_size']] for i in range(0,len(raw),row['packet_size'])]);ctx=copy.deepcopy(row['recorded']['context']);ctx['path']=np.asarray(ctx['path']).reshape(-1,2)
                    for k in ('radar_points','radar_publication_ns','radar_publication_age_s','radar_clock_bound_s','radar_frame_native'):
                        if k in row.get('raw_context',{}):ctx[k]=copy.deepcopy(row['raw_context'][k])
                    quality=copy.deepcopy(row['recorded']['quality']);deadline=None
                    if config.get('deadline'):quality['processing_overrun']=False;deadline=time.perf_counter_ns()+150_000_000
                    started=time.perf_counter_ns();r=worker.update(points,ctx,quality,(stamps-stamps.max())/1e9,processing_deadline_ns=deadline)
                    rows.append({'at_ns':r['at_ns'],'alerts':r['alerts'],'reasons':r['reasons'],'status':r['status'],'roundtrip_ms':(time.perf_counter_ns()-started)/1e6,'worker':dict(worker.last_metrics)})
                    if len(rows)%50==0:(out/'PROGRESS.json').write_text(json.dumps({'job':job['name'],'frames':len(rows),'total':len(data['frames'])}))
            finally:worker.close()
            alerts=[{'at_ns':r['at_ns'],'phrase':v['phrase'],'range_m':v['range_m'],'ttc_s':v['ttc_s']}for r in rows for v in r['alerts']]
            result={'job':job['name'],'frames':len(rows),'alerts':alerts,'roundtrip_ms_p50':float(np.median([r['roundtrip_ms']for r in rows])),'roundtrip_ms_p95':float(np.percentile([r['roundtrip_ms']for r in rows],95)),'deadline_failures':sum('inference exceeded 150 ms'in r['reasons']for r in rows),'startup_ms':worker.startup_ms}
            results.append(result);(out/(job['name']+'.json')).write_text(json.dumps(rows));(out/'RESULT.json').write_text(json.dumps(results,indent=2))
    return {'status':'completed','results':results,'deadline':bool(config.get('deadline')),'physical_audio':False}

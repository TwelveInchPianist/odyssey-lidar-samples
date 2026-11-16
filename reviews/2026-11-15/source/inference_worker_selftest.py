"""Frozen-package inference-worker smoke test. Synthetic data, no audio/control."""
import json,os,time
from pathlib import Path
import numpy as np
from fcw_engine import Engine,Mount
from inference_worker import EngineProcess,WorkerFailure

class RecoveryFixture:
    mount=None
    def __init__(self):self.rejected=[]
    def update(self,points,context,quality,relative,**kwargs):
        time.sleep(context.get('sleep',0))
        return {'at_ns':1,'alerts':[{'track_id':7}], 'status':'eligible_experimental',
                'reasons':[], 'rejected':list(self.rejected)}
    def reject_delivery(self,track_id,at_ns):self.rejected.append([track_id,at_ns])

def recovery_probe():
    """Actual spawned child, deliberately late output and a hung child, no sensor/audio."""
    worker=EngineProcess(RecoveryFixture())
    try:
        worker.wait_ready();pid=worker.process.pid
        result=worker.update(np.zeros((1,4)),{'sleep':.9},processing_deadline_ns=time.perf_counter_ns()+150_000_000)
        assert not result['alerts'] and result['status']=='not_ready'
        assert worker.last_metrics['late_response_recovered']
        late_metrics=dict(worker.last_metrics)
        result=worker.update(np.zeros((1,4)),{})
        assert result['rejected']==[[7,1]] and worker.process.pid==pid
        t=time.perf_counter()
        try:worker.update(np.zeros((1,4)),{'sleep':10})
        except WorkerFailure as e:
            assert 'command=update' in str(e);timeout_error=str(e)
        else:raise AssertionError('Hung child did not time out')
        assert not worker.process.is_alive()
        return {'late_alert_discarded':True,'same_child_recovered':True,'cooldown_undone':True,
                'hung_child_reaped':True,'hung_elapsed_s':time.perf_counter()-t,
                'late_metrics':late_metrics,'timeout_error':timeout_error}
    finally:worker.close()


def run(owner_exit_path=None):
    worker=EngineProcess(Engine(Mount(measured=True,alignment_verified=True)))
    worker.wait_ready()
    if owner_exit_path:
        if os.name!='nt':raise RuntimeError('Owner-exit test requires Windows job ownership')
        with Path(owner_exit_path).open('x',encoding='utf-8') as f:json.dump({'child_pid':worker.process.pid},f)
        os._exit(0)
    try:
        x,y=np.meshgrid(np.linspace(5,39,80),np.linspace(-8,8,45));ground=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        yy,zz=np.meshgrid(np.linspace(-.8,.8,15),np.linspace(.25,1.7,12));alerts=[];base=time.perf_counter_ns()
        for i in range(14):
            obstacle=np.column_stack([np.full(yy.size,15-.5*i),yy.ravel(),zz.ravel()]);body=np.vstack([ground,obstacle]);points=np.column_stack([body-np.asarray(worker.mount.xyz),np.ones(len(body))*40])
            ctx={'at_ns':base+i*100000000,'speed_mps':5.,'yaw_rate_rps':0.,'acceleration_mps2':0.,'ready':True,'motion_ready':True,'reasons':[],'path':np.column_stack([np.linspace(0,120,61),np.zeros(61)])}
            result=worker.update(points,ctx,{'completeness':1.,'timestamp_valid':True,'arrival_age_s':0.,'timing_bound_s':.001})
            alerts.extend(result['alerts'])
        if not alerts:raise RuntimeError('Synthetic worker smoke did not reach an alert')
        return {'status':'passed','synthetic_frames':14,'alerts':len(alerts),'startup_ms':worker.startup_ms,'last_metrics':worker.last_metrics,'recovery':recovery_probe(),'physical_audio_played':False,'scope':'packaged IPC/engine functionality only, not real-sensor detection validation'}
    finally:worker.close()

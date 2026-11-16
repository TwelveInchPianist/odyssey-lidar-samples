"""Session-pinned radar measurement forecasts; never a warning-gate override.

Each future XY is in the future ego/bumper frame, not a world-space object path.
Cross-sensor association must separately establish timing and extrinsics before
these observations can affect lidar tracks. This consumer does not assert that.
"""
import copy
from pathlib import Path
import threading
import runtime_clock as time
from .model_registry import ModelRegistry
from .radar_forecast import FeatureBuilder,ForecastModel,OBJECTIVE
from .runtime_contract import load_contract


class RadarLiveForecaster:
    def __init__(self,data_root,assets,engine_version,session_id):
        data_root=Path(data_root)
        self.lock=threading.RLock();self.inference_disabled=False;self.max_inference_ms=0.;self.last_inference_ms=0.
        self.builder=FeatureBuilder();self.session_id=session_id
        self.model=None;self.predictor=ForecastModel(None);self.model_error=None
        self.commit=None;self.vehicle=None;self.last_source=None;self.result=None
        try:
            contract=load_contract(assets)
            self.model=ModelRegistry(data_root/'learning/radar-models').load_active(
                engine_version,expected_runtime_signature=contract['runtime_signature'])
            if self.model is not None:
                profile=self.model.get('provenance',{}).get('source_profile')
                if not isinstance(profile,dict) or not profile.get('openpilot_commit') or not profile.get('car_fingerprint'):
                    raise ValueError('Active model lacks a verified vehicle/software source profile')
                self.predictor=ForecastModel(self.model)
        except Exception as error:
            self.model=None;self.predictor=ForecastModel(None);self.model_error=str(error)

    @property
    def model_id(self):return self.model.get('model_id') if self.model else None

    def identity_record(self,item):
        with self.lock:
            if item.get('kind')=='hello':
                self.commit=item.get('commit') if item.get('git_dirty') is False else None
                self.vehicle=None;self.clear()
            elif item.get('kind') in ('stream_restart','route_reference'):
                self.clear()
            elif item.get('kind')=='sample' and item.get('service')=='carParams':
                data=item.get('data') or {}
                vehicle=data.get('carFingerprint') if item.get('valid') is True and data.get('radarUnavailable') is False else None
                if vehicle!=self.vehicle:self.clear()
                self.vehicle=vehicle

    def clear(self):
        with self.lock:self.builder.clear();self.last_source=None;self.result=None

    def accept(self,record):
        with self.lock:
            if self.inference_disabled:return
            started=time.perf_counter_ns()
            try:self._accept(record)
            finally:
                self.last_inference_ms=(time.perf_counter_ns()-started)/1e6
                self.max_inference_ms=max(self.max_inference_ms,self.last_inference_ms)
                # Optional display inference must never accumulate work behind
                # raw capture. A budget overrun retires it for this session.
                if self.last_inference_ms>10:
                    self.inference_disabled=True;self.clear()

    def _accept(self,record):
        if not record.get('available') or record.get('kind')!='raw_oem_tracks':
            self.clear();return
        stamp=record['source_ns']
        if stamp==self.last_source:return  # M1/display polls never advance a radar history.
        self.last_source=stamp
        route=record.get('route_id')
        trip='comma:'+route if isinstance(route,str) and route else 'unattributed:'+self.session_id
        route=route if isinstance(route,str) and route else 'unattributed'
        rows=self.builder.update(record,trip,route)
        profile=self.model.get('provenance',{}).get('source_profile') if self.model else None
        compatible=bool(profile and profile['openpilot_commit']==self.commit and profile['car_fingerprint']==self.vehicle)
        predictor=self.predictor if compatible else ForecastModel(None)
        forecasts=[]
        for row in rows:
            value=predictor.predict(row)
            forecasts.append({'track_id':row['track_id'],'source_ns':stamp,**value})
        self.result={'source_ns':stamp,'epoch':record.get('epoch'),'model_id':self.model_id if compatible else None,
                     'model_pinned_for_session':self.model_id,'source_profile_compatible':compatible,
                     'objective':OBJECTIVE,'frame_origin':'front_bumper_flu',
                     'frame_time_basis':'future_ego_frame_at_each_horizon','tracks':forecasts,
                     'warning_override':False,'fusion_eligible':False}

    def snapshot(self,record):
        # The caller may hold the lidar context lock. Never wait for optional
        # model inference while that lock protects acquisition/normalization.
        if not self.lock.acquire(blocking=False):
            return {'status':'inference_busy','tracks':[],'model_id':self.model_id,
                    'warning_override':False,'fusion_eligible':False,'inference_budget_ms':10,
                    'note':'Optional radar forecast is busy; raw recording and FCW gates are independent.'}
        try:return self._snapshot(record)
        finally:self.lock.release()

    def _snapshot(self,record):
        status='qualified_model_pinned' if self.model else 'baseline_only_no_qualified_model'
        if self.model_error:status='model_unavailable'
        if self.inference_disabled:status='disabled_after_inference_budget_overrun'
        result={'status':status,'model_id':self.model_id,'error':self.model_error,
                'warning_override':False,'fusion_eligible':False,
                'last_inference_ms':self.last_inference_ms,'max_inference_ms':self.max_inference_ms,'inference_budget_ms':10,
                'note':'Future radar measurements in future vehicle coordinates; not collision truth or a world-space object trajectory.'}
        if self.result and record.get('available') and self.result['source_ns']==record.get('source_ns') and self.result['epoch']==record.get('epoch'):
            result.update(copy.deepcopy(self.result))
            if self.model and not self.result['source_profile_compatible']:
                result['status']='baseline_source_profile_mismatch'
        else:result['tracks']=[]
        return result

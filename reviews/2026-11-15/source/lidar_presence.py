"""Read-only presence adapter over the existing engine's pre-collision tracks."""
import math
class LidarPresence:
    def __init__(self):self.first={};self.last=None
    def observations(self,result,context,quality,receipt_ns):
        stamp=result['at_ns']
        if self.last is not None and not 0<stamp-self.last<=500_000_000:self.first.clear()
        self.last=stamp;ground=result.get('ground') or {}
        valid=bool(ground and ground.get('source','fit')=='fit' and not ground.get('degraded') and quality.get('timestamp_valid') is True and quality.get('completeness',0)>=.98 and quality.get('arrival_age_s',1)<=.2)
        current={t['id']:t for t in result.get('tracks',[]) if not t['misses']}
        self.first={k:v for k,v in self.first.items() if k in current};out=[]
        for key,t in current.items():
            self.first.setdefault(key,stamp);p=t['position'];out.append({'key':'lidar:'+str(key),'sensor':'lidar','source_ns':stamp,'receipt_ns':receipt_ns,'valid':valid and not t.get('association_ambiguous',True),'position_m':list(p),'range_m':math.hypot(*p),'bearing_deg':math.degrees(math.atan2(p[1],p[0])),'label':'unknown','class_confirmed':False,'support':t['hits'],'history_s':(stamp-self.first[key])/1e9,'bounds_m':t['bounds_m'],'points':t['points'],'motion_confirmed':False,'map_suppression':False,'scan_first_device_ns':quality.get('first_device_ns'),'scan_last_receipt_ns':quality.get('last_receipt_ns'),'physical_calibration':'provisional','ground_source':ground.get('source','fit')})
        return out

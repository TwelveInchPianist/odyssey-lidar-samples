"""Independent camera recovery, with epoch-safe IDs and no global speech reset."""
import copy,hashlib,threading
class CameraRecovery:
    def __init__(self):self.lock=threading.Lock();self.states={}
    def invalidate(self):
        with self.lock:
            for s in self.states.values():s['good']=0;s['live']=False
    def status(self):
        with self.lock:return {cam:bool(s['live']) for cam,s in self.states.items()}
    def accept(self,row,now,*,paused=False,control_epoch=None):
        cam=row.get('camera');stamp=row.get('source_ns');native=row.get('native_source_ns');frame=row.get('frame_id');worker=row.get('worker_epoch');writer=(row.get('stream_identity') or {}).get('writer');domain=row.get('native_clock_domain')
        if cam not in ('narrow','wide') or any(type(x) is not int for x in (stamp,native,frame)) or not isinstance(worker,str) or not worker:
            return {**row,'fresh':False,'observations':[],'recovery_reason':'missing_capture_identity'}
        epoch=(worker,str(writer),str(domain));prefix=hashlib.sha256(repr(epoch).encode()).hexdigest()[:16]
        with self.lock:
            old=self.states.get(cam)
            if old is None or old['epoch']!=epoch or stamp-old['source_ns']>1_250_000_000:
                old={'epoch':epoch,'prefix':prefix,'start_ns':stamp,'source_ns':0,'native_ns':0,'frame_id':-1,'good':0,'live':False}
                self.states[cam]=old
            distinct=stamp>old['source_ns'] and native>old['native_ns'] and frame!=old['frame_id']
            fresh=bool(distinct and row.get('fresh') is True and 0<=now-stamp<=300_000_000 and not paused and (control_epoch is None or row.get('control_epoch')==control_epoch))
            if distinct:
                old.update(source_ns=stamp,native_ns=native,frame_id=frame)
                old['good']=old['good']+1 if fresh else 0
                if not fresh:old['live']=False
                elif old['good']>=3:old['live']=True
            if not fresh or not old['live']:return {**row,'fresh':False,'observations':[],'recovery_reason':'timely_distinct_capture_sequence_required'}
            def key(value,camera):
                head='camera:'+camera+':'
                if not isinstance(value,str) or not value.startswith(head):return value
                return head+self.states[camera]['prefix']+':'+value[len(head):]
            observations=copy.deepcopy(row.get('observations',[]))
            for ob in observations:
                ob['key']=key(ob.get('key'),cam)
                for part in ob.get('components',[]):part['key']=key(part.get('key'),cam)
                links=[]
                for link in ob.get('visual_identity_links',[]):
                    peer=link.get('peer_key','');peer_cam=next((c for c in self.states if peer.startswith('camera:'+c+':')),None)
                    peer_ns=link.get('peer_source_ns')
                    if peer_cam is None or type(peer_ns) is not int or not self.states[peer_cam]['start_ns']<=peer_ns<=self.states[peer_cam]['source_ns']:continue
                    link['source_key']=key(link.get('source_key'),cam);link['peer_key']=key(peer,peer_cam);links.append(link)
                ob['visual_identity_links']=links
            return {**row,'observations':observations,'recovery_reason':'qualified_capture_stream'}

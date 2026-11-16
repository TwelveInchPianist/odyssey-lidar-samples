"""Experimental longitudinal corroboration; no crossing/classification authority.

Candidate only. Repeated measured radar identity + independently observed near-surface speed must
agree before using ground speed in reach-time. Missing evidence preserves the
baseline stationary possibility. No camera recognition or new subscriptions.
"""
import math
from radar_fusion import ASSOC_X_M, ASSOC_Y_M, ASSOC_V_MPS, STATIONARY_MPS
from radar import MAX_AGE_NS

class LongitudinalEvidence:
 def __init__(self): self.history={}
 def update(self, tracks, resolutions, context):
  now=context.get('at_ns');publication=context.get('radar_publication_ns');age=context.get('radar_publication_age_s');answer={};live={t.id for t in tracks}
  self.history={k:v for k,v in self.history.items() if k in live}
  for t in tracks:
   from radar_fusion import associate
   from motion_trust import evidence
   assoc=(resolutions[t.id]['radar'] if t.id in resolutions else associate(t,context.get('radar_points') or [],float(context.get('speed_mps') or 0.),evidence(t)['trusted']));good=False
   try:
    p=assoc['point'];v=float(assoc['ground_speed_mps']);lv=float(t.velocity[0]);nx=float((getattr(t,'near_edge_state',None) or {}).get('v_near_x'));width=float(t.hi[1]-t.lo[1]);length=float(t.hi[0]-t.lo[0])
    rivals=[q for q in context.get('radar_points',[]) if q.get('measured') is True and abs(q['x']-float(t.lo[0]))<=ASSOC_X_M and abs(q['y']-float(t.position[1]))<=ASSOC_Y_M]
    good=(isinstance(now,int) and isinstance(publication,int) and publication>0 and isinstance(age,(int,float)) and 0<=age<=MAX_AGE_NS/1e9 and not t.misses and not t.association_ambiguous and t.hits>=10 and t.age_s>=.9 and p.get('measured') is True and not assoc['stationary'] and len(rivals)==1 and 0<=width<=8 and 0<=length<=8 and all(math.isfinite(x) for x in (v,lv,float(p['x']),float(p['vrel']))) and min(v,lv,nx)>STATIONARY_MPS and abs(v-lv)<=ASSOC_V_MPS and abs(v-nx)<=ASSOC_V_MPS)
   except (KeyError,TypeError,ValueError,AttributeError):good=False
   if not good:self.history.pop(t.id,None);continue
   old=self.history.get(t.id);same=old and old['radar_id']==p['id'] and 0<(now-old['at_ns'])/1e9<=.25 and publication>=old['publication']
   if same and publication>old['publication']:
    dt=(now-old['at_ns'])/1e9
    # A jump inconsistent with measured closing speed means identity cannot coast.
    same=abs((p['x']-old['x'])/dt-(p['vrel']+old['vrel'])/2)<=ASSOC_V_MPS
   if not same:old={'first_ns':now,'count':0,'radar_id':p['id'],'publication':None}
   count=old['count']+int(publication!=old['publication']);span=(now-old['first_ns'])/1e9
   self.history[t.id]={'first_ns':old['first_ns'],'at_ns':now,'count':count,'radar_id':p['id'],'publication':publication,'x':p['x'],'vrel':p['vrel']}
   if count>=3 and span>=.5:answer[t.id]={'basis':'persistent_radar_near_surface_longitudinal_agreement','ground_speed_mps':min(v,lv,nx),'near_surface_speed_mps':nx,'radar_id':p['id'],'distinct_publications':count,'span_s':span,'crossing_authority':False}
  return answer



def usable_for_timing(engine, track_id, evidence, now_s):
 """Only ease stationary timing for a non-braking, corroborated lead.

 Missing/negative acceleration evidence retains the original conservative path.
 This does not grant lateral/crossing authority or add a lead-suppression rule.
 """
 from lead_gate import lead_decel, LEAD_BRAKING_MPS2
 values=[lead_decel(engine,track_id,now_s),lead_decel(engine,('r',evidence['radar_id']),now_s)]
 return all(v is not None and math.isfinite(v) and v>LEAD_BRAKING_MPS2 for v in values)

"""Radar association for the forced/experimental mode (Claude fork v10).

The Honda Nidec radar (comma liveTracks) reports, per target, longitudinal range dRel from the FRONT BUMPER, lateral
offset yRel (left +) and the RELATIVE longitudinal speed vRel (negative = closing). The engine origin is under the
comma camera; the radar origin sits RADAR_X_OFFSET_M ahead of it (1.52 m, the lever arm estimated from the drives in
the baseline analysis; re-estimated here from associated pairs and logged, never silently applied).

What this module does, per frame, for the forced candidate block:
  * associate(track, radar_points): the radar point whose position matches the lidar track's near face (along-gate
    ASSOC_X_M, lateral gate ASSOC_Y_M) and whose range-rate is consistent with the lidar track's own velocity estimate
    when that estimate is trusted; returns the point and the residuals, or None.
  * An associated point gives the CLOSING SPEED directly: closing = -vRel (ego speed minus the object's ground speed),
    exact and instantaneous, instead of the alpha-beta estimate that needs ~0.3 s of consistent motion.
  * contradiction(): a lidar track believed to be moving (>= 1 m/s) whose associated radar point says stationary
    (|vRel + ego| < STATIONARY_MPS) is velocity noise on static clutter (0.9.7's whole nuisance family).
  * shadow_candidates(): radar-only moving targets on a collision course beyond the lidar's reach or not matched to any
    lidar track. Logged only; they become warnings only once the 32-route clutter measurement says which bands are
    clean enough (report\RADAR-CLUTTER-32-ROUTES.md).

Inputs are plain lists of dicts {'x': x_engine_m, 'y': y_m, 'vrel': m/s, 'id': int, 'measured': bool}; the caller
(runtime for the live app, replay_forced for the harness) converts the radar record. Nothing here claims calibration.
"""
import math
from radar import MAX_AGE_NS

RADAR_X_OFFSET_M = 1.52      # radar origin ahead of the engine (camera) origin; baseline estimate, re-estimated from data
ASSOC_X_M = 3.0              # along-track gate between radar range and the lidar track's near face
ASSOC_Y_M = 1.6              # lateral gate (radar yRel is coarse: 1/16 m quantised, cross-range resolution ~1-2 m)
ASSOC_V_MPS = 3.0            # range-rate vs trusted lidar velocity disagreement that rejects an association
STATIONARY_MPS = 1.5         # |vRel + ego| below this: the radar says the target is not moving over the ground
SHADOW_MAX_TTC_S = 3.5
SHADOW_MAX_RANGE_M = 150.0
SHADOW_LATERAL_M = 1.5


def radar_points_from_record(record, x_offset_m=RADAR_X_OFFSET_M):
    """Convert a normalised liveTracks record (radar.py) or a list of raw dicts into engine-frame points."""
    pts = []
    if isinstance(record, dict): values = record.get('points') or []
    else: values = record or []
    for p in values:
        if not isinstance(p, dict): continue
        d = p.get('dRel', p.get('d_rel', p.get('x'))); y = p.get('yRel', p.get('y_rel', p.get('y'))); v = p.get('vRel', p.get('v_rel', p.get('vrel')))
        try: d = float(d); y = float(y); v = float(v)
        except (TypeError, ValueError): continue
        if not (math.isfinite(d) and math.isfinite(y) and math.isfinite(v)): continue
        try:tid=int(p.get('trackId', p.get('id', -1)))
        except (TypeError,ValueError,OverflowError):continue
        if tid<0:continue
        pts.append({'x': d + (0.0 if 'x' in p and 'dRel' not in p else x_offset_m), 'y': y, 'vrel': v,
                    'id': tid, 'measured': bool(p.get('measured', True))})
    return pts


def associate(track, radar_points, ego_speed_mps, lidar_velocity_trusted):
    """Best radar point for a lidar track: nearest in the (x, y) gate to the track's near face; None if none/ambiguous."""
    if not radar_points: return None
    near_x = float(track.lo[0]); cy = float(track.position[1]); vx = float(track.velocity[0])
    best = None
    for p in radar_points:
        dx = p['x'] - near_x; dy = p['y'] - cy
        if abs(dx) > ASSOC_X_M or abs(dy) > ASSOC_Y_M: continue
        # Radar vRel is longitudinal relative velocity; lidar vx is ground-frame velocity.
        if lidar_velocity_trusted and abs((p['vrel'] + ego_speed_mps) - vx) > ASSOC_V_MPS: continue
        score = abs(dx) / ASSOC_X_M + abs(dy) / ASSOC_Y_M
        if best is None or score < best['score']:
            best = {'score': score, 'point': p, 'dx_m': round(dx, 2), 'dy_m': round(dy, 2), 'vrel_mps': round(p['vrel'], 2),
                    'ground_speed_mps': round(p['vrel'] + ego_speed_mps, 2), 'stationary': abs(p['vrel'] + ego_speed_mps) < STATIONARY_MPS}
    return best


def contradiction(assoc, lidar_velocity_trusted, lidar_speed_mps):
    """True when the lidar believes the object moves but the radar sees it stationary (velocity noise on clutter)."""
    return bool(assoc and assoc['stationary'] and lidar_velocity_trusted and lidar_speed_mps >= 1.0)


RADAR_ONLY_LATERAL_M = 1.0        # 32-route measurement: |yRel| < 1.0 m same-direction -> 0.25/h unconfirmed
RADAR_ONLY_STREAK = 5             # >= 5 consecutive liveTracks messages (0.25 s): 0/h in that sample
RADAR_ONLY_MAX_TTC_S = 4.0


def radar_only_candidates(radar_points, ego_speed_mps, matched_ids, streaks, *,
                          publication_ns=None, now_ns=None, publication_times=None, publication_age_s=None,
                          publication_clock_bound_s=0.):
    """Same-direction moving radar targets on a collision course, persistent, unmatched by lidar: warning candidates.
    Both state dictionaries belong to the engine and are checkpointed. Native publication
    identity must be supplied by the capture/replay caller; repeated lidar frames and
    duplicate IDs within one radar publication cannot increase the streak. Missing,
    nonmonotonic, stale and interrupted evidence never manufactures persistence.
    """
    if publication_times is None:
        streaks.clear()
        return []
    valid_stamp=isinstance(publication_ns,int) and not isinstance(publication_ns,bool) and publication_ns>0
    valid_now=isinstance(now_ns,int) and not isinstance(now_ns,bool) and now_ns>0
    valid_bound=isinstance(publication_clock_bound_s,(int,float)) and math.isfinite(publication_clock_bound_s) and 0<=publication_clock_bound_s<=.04
    valid_age=(valid_bound and isinstance(publication_age_s,(int,float)) and math.isfinite(publication_age_s)
               and publication_age_s>=-publication_clock_bound_s and publication_age_s+publication_clock_bound_s<=MAX_AGE_NS/1e9)
    if not (valid_stamp and valid_now and valid_age):
        streaks.clear();publication_times.clear()
        return []
    out = []; seen = set()
    # Ignore duplicate IDs in this publication. Conflicting duplicate values are
    # disqualified rather than counted as multiple independent measurements.
    unique={};conflicts=set()
    for p in radar_points:
        try:
            rid=int(p['id'])
            if rid<0 or not all(math.isfinite(float(p[x])) for x in ('x','y','vrel')):continue
        except (KeyError,TypeError,ValueError):continue
        if rid in unique:
            if any(unique[rid].get(k)!=p.get(k) for k in ('x','y','vrel','measured')):conflicts.add(rid)
        else:unique[rid]=p
    for rid,p in unique.items():
        if rid in conflicts:continue
        closing = -p['vrel']; ground = p['vrel'] + ego_speed_mps
        ok = (p['measured'] and ground > STATIONARY_MPS and closing > 1.0 and abs(p['y']) < RADAR_ONLY_LATERAL_M
              and 0 < p['x'] <= SHADOW_MAX_RANGE_M and p['x'] / closing <= RADAR_ONLY_MAX_TTC_S)
        if ok:
            seen.add(rid);previous=publication_times.get(rid)
            if previous is None or publication_ns<previous or publication_ns-previous>250_000_000:
                streaks[rid]=1
            elif publication_ns>previous:
                streaks[rid]=streaks.get(rid,0)+1
            publication_times[rid]=publication_ns
            if streaks.get(rid,0) >= RADAR_ONLY_STREAK and rid not in matched_ids:
                out.append({'radar_id': rid, 'x_m': round(p['x'], 2), 'y_m': round(p['y'], 2), 'closing_mps': round(closing, 2),
                            'ground_speed_mps': round(ground, 2), 'ttc_s': round(p['x'] / closing, 3), 'streak': streaks[rid],
                            'publication_ns':publication_ns})
    for k in list(streaks):
        if k not in seen:del streaks[k];publication_times.pop(k,None)
    return out


def shadow_candidates(radar_points, ego_speed_mps, matched_ids):
    """Radar-only moving targets on a collision course; logged, not warned (gating awaits the 32-route clutter rates)."""
    out = []
    for p in radar_points:
        if p['id'] in matched_ids or not p['measured']: continue
        closing = -p['vrel']; ground = p['vrel'] + ego_speed_mps
        if abs(ground) < STATIONARY_MPS or closing < 1.0 or abs(p['y']) > SHADOW_LATERAL_M or p['x'] > SHADOW_MAX_RANGE_M: continue
        ttc = p['x'] / closing
        if ttc <= SHADOW_MAX_TTC_S:
            out.append({'radar_id': p['id'], 'x_m': round(p['x'], 2), 'y_m': round(p['y'], 2), 'closing_mps': round(closing, 2),
                        'ground_speed_mps': round(ground, 2), 'ttc_s': round(ttc, 3), 'label': 'radar_only_shadow_not_a_warning'})
    return out

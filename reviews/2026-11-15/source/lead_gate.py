"""v16 moving-lead gate (Claude fork, 2026-10-20). MEASURED basis: report/LEAD-FCW-CLOUD.md (862 h of comma cloud qlogs).

A same-direction moving lead (ground speed >= 1.5 m/s along the path) only warns when
  (a) stopping 2 m short at the current closing speed needs >= LEAD_GATE_AREQ_MPS2, or
  (b) TTC <= LEAD_GATE_URGENT_TTC_S, or
  (c) the lead itself has been decelerating at <= LEAD_BRAKING_MPS2 over the last ~1 s (per-object ground-speed history).
Static objects and crossing objects are never gated. Temporal history belongs to the Engine and is explicitly checkpointed.
"""
from collections import deque

LEAD_GATE_AREQ_MPS2 = 3.5
LEAD_GATE_MIN_GROUND_MPS = 1.5
LEAD_GATE_MIN_SPEED_FRAC = 0.4   # gate only leads moving at >= 40 % of ego speed: a much slower vehicle ahead keeps the v15 warning (Astra test_forced_replay_preserves_policy..., 3.8 m/s lead at 12 m/s ego); cloud cost: Odyssey 0.36 -> 0.42/h
LEAD_GATE_URGENT_TTC_S = 2.4   # v18: was 1.5; NHTSA FCW slower-lead test needs a warning by TTC 2.0 s (report LITERATURE-REVIEW-FCW.md); cloud cost Odyssey 0.415 -> 0.423/h
LEAD_BRAKING_MPS2 = -2.0       # v18: was -3.0; NHTSA decelerating-lead test brakes the lead at 0.3 g (-2.94 m/s2), which -3.0 missed
HIST_S = 1.6


def _hist(engine):
    h = getattr(engine, '_lead_history', None)
    if h is None: h = engine._lead_history = {}
    return h


def update(engine, tracks, radar_points, ego_speed, now):
    """Record ground speed along x per lidar track id and per radar id ('r', id); drop stale entries."""
    h = _hist(engine)
    for t in tracks:
        try: vx = float(t.velocity[0])
        except Exception: continue
        h.setdefault(t.id, deque(maxlen=40)).append((now, vx))
    for p in radar_points or []:
        try: h.setdefault(('r', int(p['id'])), deque(maxlen=60)).append((now, float(p['vrel']) + float(ego_speed or 0.)))
        except Exception: continue
    for k in [k for k, d in h.items() if not d or now - d[-1][0] > 2.0]: del h[k]


def lead_decel(engine, key, now):
    """Least-squares slope of ground speed over the last HIST_S seconds (needs >= 5 samples spanning >= 0.8 s)."""
    d = _hist(engine).get(key)
    if not d: return None
    pts = [(t, v) for t, v in d if now - t <= HIST_S]
    if len(pts) < 5 or pts[-1][0] - pts[0][0] < 0.8: return None
    n = len(pts); mt = sum(t for t, _ in pts) / n; mv = sum(v for _, v in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    return None if den <= 0 else sum((t - mt) * (v - mv) for t, v in pts) / den


def blocks(engine, candidate, mount, now):
    # Same-direction nuisance policy must not delay a collision predicted from
    # crossing motion. In-path objects with substantial transverse velocity are
    # still crossing objects; they do not become leads at the corridor boundary.
    me = candidate.get('motion_evidence') or {}
    if me.get('crossing_ok'): return False
    gv = candidate.get('ground_velocity_mps') or [0., 0.]
    try: vx, vy = float(gv[0]), float(gv[1])
    except Exception: return False
    closing0 = float(candidate.get('closing_mps') or 0.); ego = vx + closing0   # same-direction lead: ego speed = lead speed + closing
    if not (vx >= max(LEAD_GATE_MIN_GROUND_MPS, LEAD_GATE_MIN_SPEED_FRAC * ego) and abs(vy) < .8): return False
    if not me.get('trusted'): return False      # gate only leads with trusted motion (radar range rate / sustained consistency). 2026-10-20 drive G: an untrusted fence track at 5.5 m/s was gated and its warning delayed 0.5 s
    ttc = float(candidate.get('ttc_s') or 99.); closing = float(candidate.get('closing_mps') or 0.)
    if ttc <= LEAD_GATE_URGENT_TTC_S: return False
    gap = float(candidate.get('range_m') or 0.) - float(getattr(mount, 'bumper_x_m', 0.) or 0.) - 2.
    if closing * closing / (2 * max(gap, .5)) >= LEAD_GATE_AREQ_MPS2: return False
    tid = candidate.get('track_id')
    key = ('r', -int(tid) - 1) if isinstance(tid, int) and tid < 0 else tid
    a = lead_decel(engine, key, now)
    if a is not None and a <= LEAD_BRAKING_MPS2:
        candidate['lead_decel_mps2'] = round(a, 2); return False
    return True


# ---- v19: time to collision for a BRAKING lead (constant-velocity TTC runs long when the lead decelerates) ----
LEAD_TTC_DECEL_MPS2 = -1.5   # use the lead's measured deceleration once it is at least this strong (trusted moving lead only)


def braking_lead_ttc(engine, track_id, now, distance_m, ego_mps, lead_mps, ego_accel=0.0):
    """Time to contact with the lead decelerating at its measured rate and the van at its current acceleration (both stop at
    zero speed); None when the lead is not measured braking, or when the van stops first. Step simulation, 0.05 s, <= 8 s."""
    a = lead_decel(engine, track_id, now)
    if a is None or a > LEAD_TTC_DECEL_MPS2 or lead_mps <= 0.5 or ego_mps <= 0.5: return None
    ae = min(0.0, float(ego_accel)); ve, vl, gap, t, dt = float(ego_mps), float(lead_mps), float(distance_m), 0.0, .05
    while t < 8.0:
        if gap <= 0: return t
        if ve <= 0: return None
        gap -= (ve - vl) * dt; ve = max(0.0, ve + ae * dt); vl = max(0.0, vl + a * dt); t += dt
    return None


def required_decel(engine, track_id, now, distance_m, ego_mps, lead_mps, closing_mps):
    """v19: ego deceleration needed to stop 2 m behind where a BRAKING lead will stop (lead decel measured); falls back to
    closing^2 / (2 d) when the lead is not measured braking."""
    a = lead_decel(engine, track_id, now)
    if a is None or a > LEAD_TTC_DECEL_MPS2 or lead_mps <= 0.5:
        return closing_mps * closing_mps / (2 * max(distance_m, .5))
    d_lead = lead_mps * lead_mps / (2 * -a)
    return ego_mps * ego_mps / (2 * max(distance_m + d_lead - 2.0, .5))

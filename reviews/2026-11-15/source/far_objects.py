"""Far-range detection for the forced mode (Claude fork): objects judged by their OWN vertical extent, not by height above an
extrapolated road plane, beyond the plane's trust distance.

Why: the flat plane fitted at 4-40 m is off by more than 0.2 m at 100 m in two thirds of frames (overnight t4), so the
height-above-plane mask erases low and medium objects out there. A car's returns span ~1.4 m top to bottom whatever the plane
error; a deer ~1.1 m; a post is tall but thin. Beyond TRUST_M we therefore cluster everything in a wide height band and keep
clusters whose own z-span is at least MIN_SPAN_M and whose top is below the overhead cut.

A quadratic (curvature) term is fitted from the lowest returns 20-50 m ahead and used only here, so the near-field pipeline and
the default engine are untouched.
"""
import numpy as np

TRUST_M = 40.0          # plane trusted (90 % of frames within 0.2 m) out to ~40 m (t4)
MIN_SPAN_M = 0.5        # minimum own vertical extent for a far object
# v17 (2026-10-20): low compact objects (debris). SYNTHETIC t5 high-speed: a 0.5 m box returns 22-34 points at 50-70 m but its
# measured span is only 0.2-0.47 m (beam spacing), so MIN_SPAN rejected it frame after frame and the track churned until ~35 m.
LOW_SPAN_M = 0.2          # accept a smaller span when ...
LOW_TOP_M = 0.3           # ... the cluster top is clearly above the road plane (road paint / surface stays below this)
LOW_MAX_ALONG_M = 2.0     # ... it is compact
LOW_MAX_CROSS_M = 2.5
LOW_MAX_ABS_Y_M = 2.5     # ... it is in or next to the lane
LOW_MIN_FRAMES = 3        # ... and it is seen in >= 3 of the last 4 frames (fcw_engine far block)
# v18: low vertical faces (see far_clusters)
VFACE_MIN_SPAN_M = 0.25
HOLD_SPAN_M = 0.10; HOLD_TOP_M = 0.15; HOLD_RADIUS_M = 1.5; HOLD_MIN_HITS = 3; HOLD_MIN_X_M = 35.0   # v30 far-track hysteresis
VFACE_MIN_POINTS = 8
VFACE_MAX_ALONG_M = 0.4
VFACE_MIN_STEEP = 1.5      # height span >= 1.5 x along-range depth
MIN_POINTS = 4
BAND_LOW_M, BAND_HIGH_M = -0.8, 3.2    # wide acceptance band around the (uncertain) plane
OVERHEAD_TOP_M = 2.6    # cluster top above this = sign / branch / bridge, not an obstacle
LATERAL_M = 10.0        # only the road ahead matters at range
K_MAX = 0.0002          # v14: curvature clip (see curvature())
RANGE_TOL_PER_M = 0.012 # v14: plane uncertainty grows with range; the floating / overhead / band-top gates relax by this per metre beyond TRUST_M


def curvature(points, abc):
    """k in z ~ a x + b y + c + k x^2, from the lowest returns per 2 m cell between 20 and 50 m ahead. 0 if unsupported."""
    a, b, c = abc
    m = (points[:, 0] > 20) & (points[:, 0] < 50) & (np.abs(points[:, 1]) < 4)
    if m.sum() < 60: return 0.0
    q = points[m]; res = q[:, 2] - (a * q[:, 0] + b * q[:, 1] + c)
    cell = np.floor(q[:, 0] / 2).astype(int); lows = {}
    for i in range(len(q)):
        if res[i] < lows.get(cell[i], (9, 0))[0]: lows[cell[i]] = (res[i], q[i, 0])
    if len(lows) < 6: return 0.0
    r = np.array([v[0] for v in lows.values()]); x = np.array([v[1] for v in lows.values()])
    keep = np.abs(r) < 1.0
    if keep.sum() < 6: return 0.0
    x2 = x[keep] ** 2 - np.mean(x[keep] ** 2); rr = r[keep] - r[keep].mean()
    k = float((x2 @ rr) / max(x2 @ x2, 1e-9))
    return float(np.clip(k, -K_MAX, K_MAX))   # v14: |k| 0.0002 /m^2 -> 2 m at 100 m (sag/crest radius >= 2.5 km). The old +-0.004 clip
                                             # allowed 40 m at 100 m: a noisy fit from 20-50 m then put real objects 2-3 m 'above' the
                                             # road beyond 75 m and the floating/overhead rules rejected them (t5 trial A-05062, 10/20)


def far_clusters(points, ground, mount, policy, reach_m, point_components, existing, return_membership=False, held=None):
    """Detections beyond TRUST_M (up to reach_m) judged by own extent. `existing` = near-field detections to avoid duplicates."""
    if reach_m <= TRUST_M or ground is None: return []
    a, b, c = ground['abc']; k = curvature(points, (a, b, c))
    x, y = points[:, 0], points[:, 1]
    h = points[:, 2] - (a * x + b * y + c + k * x * x)
    tol = RANGE_TOL_PER_M * np.clip(x - TRUST_M, 0., None)   # v14: 0 at 40 m, ~1 m at 120 m
    m = (x > TRUST_M) & (x < reach_m) & (np.abs(y) < LATERAL_M) & (h > BAND_LOW_M - tol) & (h < BAND_HIGH_M + tol) & np.isfinite(points).all(axis=1)
    if m.sum() < MIN_POINTS: return []
    p = points[m]; hh = h[m]; src = np.flatnonzero(m)
    cell = policy['cluster_cell_m'] * 1.5          # returns are sparser out there
    out = []
    dbg = bool(__import__('os').environ.get('FCW_FAR_DEBUG'))
    for comp in point_components(p, cell):
        if len(comp) < MIN_POINTS: continue
        g = p[comp]; gh = hh[comp]; span = float(gh.max() - gh.min()); tolc = RANGE_TOL_PER_M * max(0., float(np.median(g[:, 0])) - TRUST_M)
        rej = span < MIN_SPAN_M or gh.max() > OVERHEAD_TOP_M + tolc or gh.min() > 1.2 + tolc   # too flat / overhead / floating (tolerance grows with range)
        low = False
        if rej and span < MIN_SPAN_M and span >= LOW_SPAN_M and gh.max() >= LOW_TOP_M + tolc and gh.min() <= 1.2 + tolc:
            ext = g[:, :2].max(axis=0) - g[:, :2].min(axis=0)
            if ext[0] <= LOW_MAX_ALONG_M and ext[1] <= LOW_MAX_CROSS_M and abs(float(np.median(g[:, 1]))) <= LOW_MAX_ABS_Y_M:
                rej = False; low = True   # v17: low compact object in the lane (debris); must persist (fcw_engine: >= LOW_MIN_FRAMES)
        vface = False
        if rej and span < MIN_SPAN_M and span >= VFACE_MIN_SPAN_M and len(comp) >= VFACE_MIN_POINTS and gh.max() <= OVERHEAD_TOP_M + tolc and gh.min() <= 1.2 + tolc:
            # v18: a low VERTICAL FACE: several rings at nearly the same range. Road surface at grazing angle spreads consecutive rings
            # metres apart in range, so it can never look like this, whatever the plane extrapolation says (it jitters by up to 1 m
            # frame to frame at 50-75 m: SYNTHETIC t5hs trial G-08695, a 0.5 m box read 0.5 m below the road one frame, above it the next;
            # no road returns come back beside it at 50-85 m, the beams graze the asphalt at ~1.4 deg).
            ext = g[:, :2].max(axis=0) - g[:, :2].min(axis=0)
            if (ext[0] <= VFACE_MAX_ALONG_M and span >= VFACE_MIN_STEEP * max(float(ext[0]), .05) and ext[1] <= LOW_MAX_CROSS_M
                    and abs(float(np.median(g[:, 1]))) <= LOW_MAX_ABS_Y_M):
                rej = False; low = True; vface = True
        held_far = False
        if rej and held and span >= HOLD_SPAN_M and gh.min() <= 1.2 + tolc and gh.max() <= OVERHEAD_TOP_M + tolc and gh.max() >= HOLD_TOP_M:
            # v30: hysteresis for an established low far track (SYNTHETIC t5hs G-10196: debris accepted at 70 m, then its measured span
            # flickered 0.10-0.17 m at 41-65 m, the far track died twice and the warning waited for the near stage at ~15 m)
            ext = g[:, :2].max(axis=0) - g[:, :2].min(axis=0); cen = np.median(g[:, :2], axis=0)
            if (ext[0] <= LOW_MAX_ALONG_M and ext[1] <= LOW_MAX_CROSS_M and abs(float(cen[1])) <= LOW_MAX_ABS_Y_M
                    and any(np.hypot(cen[0] - hx, cen[1] - hy) < HOLD_RADIUS_M for hx, hy in held)):
                rej = False; low = True; held_far = True
        if dbg and float(np.median(g[:, 0])) > 65.: print('FARDBG x=%.1f y=%.1f n=%d span=%.2f hmin=%.2f hmax=%.2f k=%.5f %s' % (np.median(g[:, 0]), np.median(g[:, 1]), len(comp), span, gh.min(), gh.max(), k, 'REJECT' if rej else 'ok'), flush=True)
        if rej: continue
        centre = np.median(g[:, :2], axis=0)
        if any(np.hypot(*(centre - d['center'])) < 1.5 for d in existing): continue
        lo = g.min(axis=0); hi = g.max(axis=0)
        d = {'center': centre, 'lo': lo, 'hi': hi, 'points': int(len(g)), 'far': True, 'own_span_m': round(span, 2), 'curvature_k': round(k, 5)}
        if low: d['low_far'] = True
        if vface: d['vface'] = True
        if held_far: d['held_far'] = True
        if return_membership: d['indices'] = src[comp]
        out.append(d)
    return out


# ---- Claude fork v14: detection reach decoupled from the warning horizon, and multi-frame accumulation of far returns ----
FAR_STAGE_REACH_M = 120.0   # the far stage looks this far at every speed (the warning envelope stays 3.5 s x v + 15 m: threat() only
                            # sweeps 5 s ahead and the corridor reach stays the forced reach). Night drive 10/19: first tracks formed at the
                            # forced reach (54 m at 28 mph) while the sensor had >= 8 returns on the same cars from 74-125 m.
ACCUM_FRAMES = 4            # frames of far-band returns kept, moved into the current ego frame
ACCUM_MIN_FRAMES = 2        # a far cluster that is sparse in the current frame must be seen in at least this many frames
ACCUM_MIN_H_M = 0.3         # only returns this far above the current road plane are carried over: road paint / surface returns at
                            # slightly different heights per frame otherwise stack up to a fake vertical span (night drive 10/19 24:28,
                            # a crosswalk stripe 0.25 m below the extrapolated plane became a 1 m tall 'object' at 46 m)


def _rotate2(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(points) @ np.array([[c, s], [-s, c]])


class FarAccumulator:
    """Ego-compensated union of the last ACCUM_FRAMES frames' far-band returns (x > TRUST_M - 5 m), so that a sparse far
    object (a deer gives 2-4 returns per frame at 100 m) reaches MIN_POINTS over several frames. The near field is untouched;
    membership indices handed back to the engine refer to the CURRENT frame's points only."""
    def __init__(self, frames=ACCUM_FRAMES):
        self.frames = frames; self.hist = []; self.frame = 0  # [(xyz in current ego frame, frame_no)]
        # Point ownership is explicit engine-owned state.
        # Positive ids identify associated tracks; 0=unassigned, -1=contested.
        # Checkpoint codec v4 preserves these fields and rejects older state.
        self.owner_ids = {}; self._pending = None; self.last_owner_stats = {}

    def reset(self):
        self.hist = []; self.owner_ids = {}; self._pending = None; self.last_owner_stats = {}

    def step(self, p, movement, angle, frame_no, abc=None, k=0.0, exclude=None, exclude_boxes=None):
        self.frame = int(frame_no)
        kept = []
        for pts, fno in self.hist:
            if frame_no - fno >= self.frames or not len(pts): continue
            xy = _rotate2(pts[:, :2] - movement, -angle)          # same convention as the tracker's prediction
            carried = np.column_stack((xy, pts[:, 2]))
            if len(carried): kept.append((carried, fno))
        self.hist = kept
        self.owner_ids = {f:self.owner_ids.get(f,np.zeros(len(q),np.int64)) for q,f in kept}
        fin = np.isfinite(p).all(axis=1)
        sel = fin & (p[:, 0] > TRUST_M - 5.) & (np.abs(p[:, 1]) < LATERAL_M + 2.)
        if abc is not None:   # v14: carry over only returns clearly above the road (own-span judged per frame, never across frames)
            a, b, c = abc; hh = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c + k * p[:, 0] * p[:, 0]); sel &= hh > ACCUM_MIN_H_M
        # Spatial overlap with a padded moving box is not ownership. Preserve
        # all fresh measurements; actual matched memberships are assigned below.
        layout=[];offset=len(p)
        for q,f in self.hist:
            layout.append((f,offset,offset+len(q)));offset+=len(q)
        if self.hist:
            aug = np.vstack([p] + [h for h, _ in self.hist])
            tags = np.concatenate([np.full(len(p), frame_no)] + [np.full(len(h), f) for h, f in self.hist])
        else:
            aug = p; tags = np.full(len(p), frame_no)
        self.hist.append((p[sel].copy(), frame_no))
        self.owner_ids[frame_no] = np.zeros(int(sel.sum()),np.int64)
        self._pending = {'frame':frame_no,'current_indices':np.flatnonzero(sel),
                         'layout':layout,'total':len(aug)}
        return aug, tags

    def assign_owners(self, detections, tracks, motion_authorities=None):
        """Retire only observed returns assigned to a moving track.

        Independent unclustered sparse returns remain available for accumulation.
        Ambiguous associations and overlapping memberships do not gain ownership.
        Historical coordinates are never tested against a broad current box.
        """
        pending=self._pending
        if pending is None:return
        assigned=np.zeros(pending['total'],np.int64)
        moving=set();memberships=[];contradictions={};resolutions={}
        for t in tracks:
            if float(np.hypot(*t.velocity)) >= MOVING_TRACK_MPS:
                from motion_trust import inherited_motion_contradiction
                authority=(inherited_motion_contradiction(t) if motion_authorities is None else
                           motion_authorities.get(int(t.id),{'contradicted':False}))
                if authority.get('lidar_contradiction'):resolutions[int(t.id)]=authority
                if authority['contradicted']:contradictions[int(t.id)]=authority
                else:moving.add(int(t.id))
            i=t.detection_index
            if t.misses or i is None or t.association_ambiguous:continue
            d=detections[i]
            idx=d.get('_ownership_indices',d.get('indices'))
            if idx is None:continue
            idx=np.asarray(idx,dtype=np.int64)
            if np.any(idx<0) or np.any(idx>=len(assigned)):
                raise ValueError('Far ownership index outside this frame')
            memberships.append((int(t.id),np.unique(idx)))
        # Near and far detections can overlap: a large far component may contain
        # an already measured compact near component. Its broader membership
        # cannot overwrite that more specific observation. Non-nested overlaps
        # remain contested rather than selecting an arbitrary owner.
        # Memberships are validated integer indices, already made unique above.
        # Cache their sets once: subset truth is identical to np.isin(...).all(),
        # without repeating NumPy sorting/allocation for every component pair.
        # Preserve the original list order for all actual ownership assignments.
        membership_sets=[set(idx.tolist()) for _,idx in memberships]
        nested=0
        for member_index,(owner,idx) in enumerate(memberships):
            specific=[]
            for other_index,(other,small) in enumerate(memberships):
                if other!=owner and len(small)<len(idx) and membership_sets[other_index].issubset(membership_sets[member_index]):
                    specific.append(small)
            if specific:
                idx=idx[~np.isin(idx,np.concatenate(specific))];nested+=1
            old=assigned[idx]
            conflict=(old!=0)&(old!=owner)
            assigned[idx]=np.where(conflict,-1,owner)
        def merge(old,new):
            result=old.copy();known=new!=0
            conflict=known&(old!=0)&(old!=new)
            result[known]=new[known];result[conflict]=-1
            return result
        for frame,start,end in pending['layout']:
            self.owner_ids[frame]=merge(self.owner_ids[frame],assigned[start:end])
        frame=pending['frame']
        self.owner_ids[frame]=merge(self.owner_ids[frame],assigned[pending['current_indices']])
        kept=[];purged=unknown=0
        for points,frame in self.hist:
            owners=self.owner_ids[frame]
            drop=np.isin(owners,list(moving)) if moving else np.zeros(len(owners),bool)
            purged+=int(drop.sum())
            points,owners=points[~drop],owners[~drop]
            unknown+=int((owners<=0).sum())
            if len(points):
                kept.append((points,frame));self.owner_ids[frame]=owners
            else:self.owner_ids.pop(frame,None)
        self.hist=kept;self._pending=None
        self.last_owner_stats={'moving_track_ids':sorted(moving),'purged_owned_points':purged,
                               'retained_unknown_points':unknown,'contested_augmented_points':int((assigned<0).sum()),
                               'nested_memberships_resolved':nested}
        if contradictions:self.last_owner_stats['inherited_motion_contradictions']=contradictions
        if resolutions:self.last_owner_stats['reconciled_motion_contradictions']=resolutions


MOVING_TRACK_MPS = 1.5      # Moving returns are not accumulated; current measurements remain eligible.


def moving_track_boxes(tracks, movement, angle, dt):
    """Predicted current-frame boxes [x_lo, x_hi, y_lo, y_hi] of tracks moving faster than MOVING_TRACK_MPS (tracker convention)."""
    out = []
    for t in tracks:
        v = np.asarray(t.velocity, float)
        if np.hypot(*v) < MOVING_TRACK_MPS: continue
        c = _rotate2(np.asarray(t.position, float) + v * dt - movement, -angle)
        hx = max(1.0, float(t.hi[0] - t.lo[0]) / 2) + 2.0; hy = max(0.8, float(t.hi[1] - t.lo[1]) / 2) + 1.5
        out.append((c[0] - hx, c[0] + hx, c[1] - hy, c[1] + hy))
    return out


def in_boxes(xy, boxes):
    m = np.zeros(len(xy), bool)
    for x0, x1, y0, y1 in boxes: m |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return m


def accumulator_for(engine):
    """Explicit engine-owned state survives deepcopy and the versioned replay codec."""
    acc = getattr(engine, '_far_accumulator', None)
    if acc is None:
        acc = FarAccumulator(); engine._far_accumulator = acc
    return acc

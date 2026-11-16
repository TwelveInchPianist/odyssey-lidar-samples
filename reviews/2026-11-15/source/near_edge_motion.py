"""Claude fork v22: near-edge velocity agreement before crossing motion is trusted (LITERATURE-REVIEW-FCW.md section 2,
recommendation 1; Petrovskaya & Thrun 2009 anchor point, Held et al. 2014, Zhang et al. 2017 L-shape).

Problem (drive F 10/18, night drive 10/20): an object revealed during a turn grows its visible extent frame by frame; its
centroid slides sideways while the object is parked, and the centroid-based velocity is trusted as crossing motion.

Rule: per track, the lateral (y) velocity of the box corner nearest the van is measured frame to frame in the same
ego-compensated frame the centroid velocity uses. Crossing motion is accepted only after the near-corner lateral velocity has
agreed with the centroid's lateral velocity (same sign, difference <= max(AGREE_ABS, AGREE_REL x |v_centroid|)) for
AGREE_S. A parked car being revealed has a fixed near corner (disagree); a crossing car or pedestrian moves its near corner
too (agree). State belongs to Track and is included in deepcopy and replay checkpoints.
"""
import numpy as np

AGREE_ABS_MPS = 0.6
AGREE_REL = 0.35
AGREE_S = 0.3


def _st(track):
    s = getattr(track, 'near_edge_state', None)
    if s is None:
        s = track.near_edge_state = {'near': None, 'agree_s': 0.0, 'v_near_y': None}
    return s


def _near_corner(lo, hi):
    c = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])], float)
    return c[int(np.argmin(np.hypot(c[:, 0], c[:, 1])))], int(np.argmin(np.hypot(c[:, 0], c[:, 1])))


def update(track, detection, movement, angle, dt, rotate2):
    """Call with the same ego motion as motion_trust.observe, BEFORE it overwrites the track's observed state."""
    s = _st(track)
    corner, which = _near_corner(detection['lo'][:2], detection['hi'][:2])
    prev = s['near']
    if prev is not None and .025 <= dt <= .25 and track.measured_velocity is not None:
        pc, pwhich = prev
        if pwhich == which:   # the same physical corner of the box stays nearest: compare it with itself
            moved = rotate2(np.asarray(pc) - movement, -angle)
            v_near = (corner - moved) / dt
            # centroid velocity of this frame = motion_trust's measured velocity (computed just after this call); use the
            # previous frame's measured velocity rotated into this frame as its stand-in (same quantity, one frame older)
            vc = rotate2(np.asarray(track.measured_velocity), -angle)
            ok = (np.sign(v_near[1]) == np.sign(vc[1])) and abs(v_near[1] - vc[1]) <= max(AGREE_ABS_MPS, AGREE_REL * abs(vc[1]))
            s['agree_s'] = s['agree_s'] + dt if ok else 0.0
            s['v_near_y'] = float(v_near[1]); s['v_near_x'] = float(v_near[0])
        else:
            s['agree_s'] = 0.0; s['v_near_y'] = None; s['v_near_x'] = None
    else:
        s['agree_s'] = 0.0; s['v_near_y'] = None; s['v_near_x'] = None
    s['near'] = (corner, which)


def reset(track):
    s = _st(track); s['near'] = None; s['agree_s'] = 0.0; s['v_near_y'] = None; s['v_near_x'] = None


def agrees(track):
    s = getattr(track, 'near_edge_state', None)
    return bool(s and s['agree_s'] >= AGREE_S)


def info(track):
    s = getattr(track, 'near_edge_state', None) or {}
    return {'near_edge_agree_s': round(float(s.get('agree_s') or 0.0), 2), 'near_edge_v_y': s.get('v_near_y')}

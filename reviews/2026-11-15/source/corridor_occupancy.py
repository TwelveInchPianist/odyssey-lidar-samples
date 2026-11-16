"""Point-based path occupancy for the forced/experimental mode (Claude fork of 0.9.8).

0.9.8 decided "object overlaps the path" from a track's axis-aligned bounding box. On city streets the clusterer merges
hedges, walls and rows of parked cars into single components 10-40 m wide, whose box overlaps any corridor even when the
object's centre is 10 m to the side. This module looks at the cluster's actual returns instead:

  * which returns lie inside the swept corridor (van half-width + margin, out to the forced range cap),
  * how far along the path the nearest of them is,
  * whether the in-corridor part is COMPACT along the path (a vehicle, animal or debris: <= COMPACT_ALONG_M) or a long
    PARALLEL STRUCTURE (kerb, wall, guardrail, hedge, parked row the van is driving beside).

Nothing here grants calibration authority; it only decides which clusters may become candidates in forced mode.
"""
import numpy as np

MIN_POINTS_IN_CORRIDOR = 6      # debris-sized objects at 40 m give few returns; the discriminator is extent, not count
COMPACT_ALONG_M = 7.0           # longest plausible single obstacle along the path (a bus is handled by its near face)
MARGIN_M = 0.25                 # same lateral padding as threat()
CENTRE_REACH_M = None           # superseded: the body half-width (mount.width_m / 2) is the reach; kept for the record
CENTRE_SPAN_M = 1.0
TURN_NO_EXTEND_RPS = 0.10       # do not extend the path straight beyond its horizon while turning


def corridor_path(path, max_range_m, yaw_rate_rps=0.0):
    """Extend the last tangent to the range cap (classification only), except while turning: a straight extension of a
    curved path sweeps across corners and kerbs."""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
        return None
    d = path[-1] - path[-2]; norm = float(np.linalg.norm(d))
    extended = False
    if norm > 1e-6 and abs(float(yaw_rate_rps or 0.0)) <= TURN_NO_EXTEND_RPS:
        extra = max(0.0, max_range_m - float(np.linalg.norm(path[-1])))
        if extra:
            path = np.vstack((path, path[-1] + d / norm * extra)); extended = True
    return path, extended


def path_coordinates(points_xy, path):
    """Arc-length 'along' and signed 'cross' (left +) of each point relative to its nearest path vertex."""
    seg = np.diff(path, axis=0); length = np.linalg.norm(seg, axis=1)
    good = length > 1e-6
    if not good.any():
        return None, None
    tangent = np.zeros_like(path); tangent[:-1][good] = seg[good] / length[good, None]; tangent[-1] = tangent[-2]
    for i in range(len(path) - 2, -1, -1):       # fill degenerate vertices with the next tangent
        if not good[i]:
            tangent[i] = tangent[i + 1]
    arc = np.r_[0.0, np.cumsum(length)]
    rel = points_xy[:, None, :] - path[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', rel, rel)
    j = np.argmin(dist2, axis=1)
    r = rel[np.arange(len(points_xy)), j]; t = tangent[j]
    along = arc[j] + np.einsum('ij,ij->i', r, t)
    cross = -r[:, 0] * t[:, 1] + r[:, 1] * t[:, 0]
    return along, cross


def assess(points_xy, path, mount, max_range_m, yaw_rate_rps=0.0):
    """Classify a cluster's returns against the corridor. Returns a dict (always JSON-serialisable)."""
    out = {'in_path': False, 'parallel_structure': False, 'points_inside': 0, 'nearest_along_m': None,
           'extent_along_m': None, 'extent_cross_m': None, 'mean_cross_m': None, 'min_abs_cross_m': None, 'edge_only': False}
    cp = corridor_path(path, max_range_m, yaw_rate_rps)
    if cp is None or points_xy is None or len(points_xy) == 0:
        return out
    cp, extended = cp
    along, cross = path_coordinates(np.asarray(points_xy, float), cp)
    if along is None:
        return out
    half = mount.width_m / 2 + MARGIN_M
    arc_end = float(np.sum(np.linalg.norm(np.diff(cp, axis=0), axis=1)))
    # Beyond the path's end (while turning, no straight extension) the corridor is undefined: those returns do not count.
    reach = (max_range_m + mount.bumper_x_m) if extended else min(max_range_m + mount.bumper_x_m, arc_end + 1.0)
    inside = (np.abs(cross) <= half) & (along >= -1.0) & (along <= reach)
    out['path_reach_m'] = round(reach, 1)
    # v13: nearest return of the whole cluster to the path within the reach, whatever side (near-edge measurement)
    ahead = (along >= -1.0) & (along <= reach)
    if ahead.any():
        k = int(np.argmin(np.abs(cross[ahead]))); out['nearest_cross_any_m'] = round(float(np.abs(cross[ahead])[k]), 2); out['nearest_along_any_m'] = round(float(along[ahead][k]), 2)
    n = int(inside.sum()); out['points_inside'] = n
    # Beyond the plane's trust distance the far-object pass already required a 0.5 m own height; a car at 60 m only
    # returns a handful of points, so 3 in the lane is the honest minimum there (6 up close).
    need = MIN_POINTS_IN_CORRIDOR if (n and float(np.min(along[inside])) < 40.0) else 3
    if n < need:
        return out
    a = along[inside]; c = cross[inside]
    body_gap=float(np.abs(c).min())-mount.width_m/2
    out['min_body_gap_m']=round(body_gap,4)
    out.update(nearest_along_m=round(float(a.min()), 2), extent_along_m=round(float(a.max() - a.min()), 2),
               extent_cross_m=round(float(c.max() - c.min()), 2), mean_cross_m=round(float(c.mean()), 2),
               min_abs_cross_m=round(float(np.abs(c).min()), 2))
    # Long along the path AND confined to one side of the corridor = wall/kerb/hedge/parked row the van drives beside.
    # Long along the path but spanning the corridor's width (a bus, a truck) is in the way: judged by its near face.
    if float(a.min()) > 40.0 and body_gap > 0:
        out['edge_only'] = True       # far object whose body sits outside the lane; only a few returns graze the corridor
        return out
    # v30e: length/narrowness does not establish clearance. A return inside the body
    # cannot be silenced by the roadside-structure label. Keep margin-only structure
    # handling for observations wholly outside one side of the physical envelope.
    if out['extent_along_m'] > COMPACT_ALONG_M and body_gap > 0 and (np.all(c>=0) or np.all(c<=0)):
        out['parallel_structure'] = True
        return out
    # Anything inside the van's own half-width would be struck (corner included); only returns confined to the 0.25 m
    # margin band outside the body count as grazing. (Validation 10/19: a 0.5 m band lost head-on deer and debris at +-1 m.)
    if body_gap > 0 and out['extent_cross_m'] < CENTRE_SPAN_M:
        out['edge_only'] = True
        return out
    out['in_path'] = True
    return out

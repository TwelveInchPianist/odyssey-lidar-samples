"""Live map prior for the forced/experimental mode (Claude fork v12). numpy only; loads in seconds; no pandas/scipy.

Map file (results\\mapprior-cloud\\live-map.npz, exported by scripts\\mapprior\\export_live_map.py): for every reference
cell with >= 2 passes: cell_xy (N,2) ENU m, cell_heading_deg (N) compass degrees clockwise from North, n_passes (N),
lat_mad (N), a_rel / b_rel (N) vehicle-relative road edges (left +, right -, NaN = unknown), fwd_path (N,30,2) the median
forward trajectory of the cell's passes at 4 m steps to 120 m in the cell's frame (x forward, y left), NaN beyond the
point where the passes fork; lat0 / lon0 origin.

Query: the van's lat/lon/heading -> nearest cell within QUERY_R with heading within HEADING_GATE -> that cell's forward
path rotated into the van's frame and anchored at the van (y(0)=0, as the model path and lane midline are). Trust is
pass count: usable only with >= MIN_PASSES passes at the start cell and lateral MAD <= MAX_LAT_MAD (one lane used).
The anchored route is a SHAPE prior, not registered lane geometry. Legacy
vehicle-relative edges have no known shared origin and are withheld. New
cell-normal edge geometry is translated separately, with explicit uncertainty.
Measured basis: report\\MAP-PRIOR.md (0.3-0.7 m lateral error at 30-120 m on >= 2-pass streets vs 3.6 m dead reckoning
in turns; edges repeat to 0.15 m vehicle-relative). Nothing here claims calibration authority.
"""
import math
import numpy as np

QUERY_R = 8.0
HEADING_GATE = 30.0
MIN_PASSES = 3            # live: one pass more than the offline evaluation's 2 (unseen roads must not borrow one odd pass)
MAX_LAT_MAD = 2.0
STEP_M = 4.0
GRID_M = 20.0
EDGE_GEOMETRY_BASIS='cell_normal_offsets_v1'
MAX_EDGE_POSE_BOUND_M=.5
MAX_EDGE_HEADING_BOUND_DEG=2.
_A = 6378137.0; _F = 1 / 298.257223563; _E2 = _F * (2 - _F)


def _ecef(lat, lon, alt):
    la = np.radians(lat); lo = np.radians(lon); n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return (n + alt) * np.cos(la) * np.cos(lo), (n + alt) * np.cos(la) * np.sin(lo), (n * (1 - _E2) + alt) * np.sin(la)


def enu_from_geodetic(lat, lon, lat0, lon0):
    x, y, z = _ecef(float(lat), float(lon), 0.0); x0, y0, z0 = _ecef(lat0, lon0, 0.0)
    la0 = math.radians(lat0); lo0 = math.radians(lon0); dx, dy, dz = x - x0, y - y0, z - z0
    e = -math.sin(lo0) * dx + math.cos(lo0) * dy
    n = -math.sin(la0) * math.cos(lo0) * dx - math.sin(la0) * math.sin(lo0) * dy + math.cos(la0) * dz
    return float(e), float(n)


def ang_diff(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180) % 360 - 180)

def _finite_number(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


class LiveMap:
    def __init__(self, path):
        z = np.load(path)
        self.xy = np.asarray(z['cell_xy'], np.float64); self.h = np.asarray(z['cell_heading_deg'], np.float64)
        self.n = np.asarray(z['n_passes'], np.int32); self.mad = np.asarray(z['lat_mad'], np.float64)
        self.a = np.asarray(z['a_rel'], np.float64); self.b = np.asarray(z['b_rel'], np.float64)
        self.fwd = np.asarray(z['fwd_path'])            # float16 as exported (153 MB as float32 for 639k cells); one cell is converted per query
        self.lat0 = float(z['lat0']); self.lon0 = float(z['lon0']); self.built = str(z['built']) if 'built' in z.files else ''
        self.edge_basis=str(z['edge_geometry_basis']) if 'edge_geometry_basis' in z.files else 'legacy_vehicle_relative_unregistered'
        self.edge_registered=self.edge_basis==EDGE_GEOMETRY_BASIS
        if self.edge_registered:
            self.edge_h=np.asarray(z['edge_heading_deg'],float)
            self.edge_a=np.asarray(z['edge_normal_a_m'],float);self.edge_b=np.asarray(z['edge_normal_b_m'],float)
            self.edge_verified=np.asarray(z['edge_registration_verified'],bool)
            self.edge_bound=np.asarray(z['edge_registration_error_bound_m'],float)
            self.edge_extent=np.asarray(z['edge_extent_m'],float)
            if any(v.shape!=(len(self.xy),) for v in (self.edge_h,self.edge_a,self.edge_b,self.edge_verified,self.edge_bound,self.edge_extent)):
                raise ValueError('registered edge metadata shape mismatch')
        z.close()
        self.steps = STEP_M * np.arange(self.fwd.shape[1])
        # grid hash for the nearest-cell query
        g = np.floor(self.xy / GRID_M).astype(np.int64); self.keys = g[:, 0] * 1_000_003 + g[:, 1]
        order = np.argsort(self.keys); self.order = order; self.sorted_keys = self.keys[order]

    def _near(self, x, y):
        gx, gy = int(math.floor(x / GRID_M)), int(math.floor(y / GRID_M)); idx = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                k = (gx + i) * 1_000_003 + (gy + j); lo = np.searchsorted(self.sorted_keys, k); hi = np.searchsorted(self.sorted_keys, k, side='right')
                if hi > lo: idx.append(self.order[lo:hi])
        return np.concatenate(idx) if idx else np.zeros(0, np.int64)

    def query(self, lat, lon, heading_deg, reach_m=120.0, *, localization_bound_m=None,heading_bound_deg=None):
        out = {'usable': False, 'stop_reason': None, 'reach_m': 0.0, 'path': None, 'a_rel': None, 'b_rel': None, 'n_passes': 0, 'cell': None,
               'path_basis':'route_shape_anchored_at_vehicle','path_registration_verified':False,
               'edges_usable':False,'edges_coordinate_frame':'vehicle_cross_section','edges_basis':self.edge_basis,
               'edges_reason':'legacy edge offsets lack a registered common origin','registered_edges':None}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lat, lon, heading_deg)): out['stop_reason'] = 'no position'; return out
        x, y = enu_from_geodetic(lat, lon, self.lat0, self.lon0); idx = self._near(x, y)
        if not len(idx): out['stop_reason'] = 'no map cell within %.0f m' % QUERY_R; return out
        d = np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y); ok = (d <= QUERY_R) & (ang_diff(heading_deg, self.h[idx]) < HEADING_GATE)
        if not ok.any(): out['stop_reason'] = 'no map cell within %.0f m with matching heading' % QUERY_R; return out
        c = int(idx[ok][np.argmin(d[ok])]); out['cell'] = c; out['n_passes'] = int(self.n[c])
        if self.n[c] < MIN_PASSES: out['stop_reason'] = 'start cell has %d pass(es) < %d' % (self.n[c], MIN_PASSES); return out
        if not (self.mad[c] <= MAX_LAT_MAD): out['stop_reason'] = 'start cell lateral MAD %.1f m (lanes differ)' % self.mad[c]; return out
        hq=math.radians(heading_deg);delta=self.xy[c]-np.array([x,y])
        offset=np.array([delta[0]*math.sin(hq)+delta[1]*math.cos(hq),-delta[0]*math.cos(hq)+delta[1]*math.sin(hq)])
        out['cell_offset_vehicle_m']=offset.tolist()
        self._edge_query(out,c,delta,heading_deg,localization_bound_m,heading_bound_deg)
        p = np.asarray(self.fwd[c], np.float64); good = np.isfinite(p).all(axis=1)
        k = int(np.argmin(good)) if not good.all() else len(good)
        if k < 4: out['stop_reason'] = 'map path forks within %.0f m' % (k * STEP_M); return out
        p = p[:k]
        # cell frame -> vehicle frame: rotate by (cell heading - vehicle heading); anchor at the van laterally
        dh = math.radians(self.h[c] - heading_deg); cs, sn = math.cos(dh), math.sin(dh)
        vx = p[:, 0] * cs + p[:, 1] * sn; vy = -p[:, 0] * sn + p[:, 1] * cs   # compass headings: a cell heading right of the van's turns the path right (y < 0)
        keep = vx <= reach_m + 1e-6
        if keep.sum() < 4: out['stop_reason'] = 'map path too short'; return out
        vx = vx[keep]; vy = vy[keep]
        out['cell_path_in_vehicle_frame']=np.column_stack((vx+offset[0],vy+offset[1]))
        if vx[0] <= 0.5 or np.any(np.diff(vx) <= 0): out['stop_reason'] = 'map path not monotone in x (sharp turn)'; return out
        # anchor laterally where the path crosses x = 0 (extrapolate the first segment), and start the route at the van
        y0 = vy[0] - (vy[1] - vy[0]) / (vx[1] - vx[0]) * vx[0]; vy = vy - y0
        vx = np.r_[0.0, vx]; vy = np.r_[0.0, vy]
        out.update(usable=True, reach_m=float(vx[-1]), path=np.column_stack((vx, vy)))
        return out

    def _edge_query(self,out,c,delta,heading_deg,localization_bound_m,heading_bound_deg):
        if not self.edge_registered:return
        h=float(self.edge_h[c]);a=float(self.edge_a[c]);b=float(self.edge_b[c]);bound=float(self.edge_bound[c]);extent=float(self.edge_extent[c])
        if (not self.edge_verified[c] or not math.isfinite(h) or not math.isfinite(bound) or bound<0 or
                not math.isfinite(extent) or extent<=0 or not (math.isfinite(a) or math.isfinite(b))):
            out['edges_reason']='registered edge metadata is incomplete or not verified';return
        hs=math.radians(h);normal_shift=-delta[0]*math.cos(hs)+delta[1]*math.sin(hs)
        extent-=abs(delta[0]*math.sin(hs)+delta[1]*math.cos(hs))
        if extent<=0:out['edges_reason']='query lies beyond the registered local edge extent';return
        angle=math.radians(h-heading_deg);nx,ny=math.sin(angle),math.cos(angle)
        if abs(ny)<.5:out['edges_reason']='edge normal nearly parallel to vehicle forward axis';return
        left=a+normal_shift if math.isfinite(a) else None;right=b+normal_shift if math.isfinite(b) else None
        out['registered_edges']={'normal_vehicle':[nx,ny],'left_normal_m':left,'right_normal_m':right,
            'left_at_vehicle_x0_m':left/ny if left is not None else None,'right_at_vehicle_x0_m':right/ny if right is not None else None,
            'extent_m':extent,'registration_error_bound_m':bound,'basis':'registered cell-normal lines; translated to query pose'}
        pose_ok=(type(localization_bound_m) in (int,float) and math.isfinite(localization_bound_m) and localization_bound_m>=0 and
                 bound+localization_bound_m<=MAX_EDGE_POSE_BOUND_M and type(heading_bound_deg) in (int,float) and
                 math.isfinite(heading_bound_deg) and 0<=heading_bound_deg<=MAX_EDGE_HEADING_BOUND_DEG)
        if not pose_ok:out['edges_reason']='query localization/heading uncertainty absent or exceeds edge bounds';return
        out.update(edges_usable=True,edges_reason=None,a_rel=left/ny if left is not None else None,b_rel=right/ny if right is not None else None)

    @staticmethod
    def tier_rel(a_rel, b_rel, cross_m, margin_m=1.0):
        """Tier offsets only when caller establishes a SHARED coordinate frame.
        Registered vehicle edges are not directly comparable to reanchored-path cross-track offsets.
        """
        if not _finite_number(cross_m):return 'unknown_edge'
        okA = _finite_number(a_rel); okB = _finite_number(b_rel)
        if not okA and not okB: return 'unknown_edge'
        if (okA and abs(cross_m - a_rel) <= margin_m) or (okB and abs(cross_m - b_rel) <= margin_m): return 'near_edge'
        if (okA and cross_m > a_rel) or (okB and cross_m < b_rel): return 'outside'
        return 'inside' if (okA and okB) else 'inside_one_edge'

    @staticmethod
    def tier(q, obj_x, obj_y, margin_m=1.0):
        """'inside' | 'near_edge' | 'outside' | 'unknown_edge' | 'beyond_path' | 'no_map' for an object in the vehicle frame."""
        if not q.get('usable'): return 'no_map'
        if not q.get('edges_usable'):return 'unknown_edge'
        xs = q['path'][:, 0]
        if obj_x > xs[-1] + STEP_M / 2: return 'beyond_path'
        edges=q.get('registered_edges')
        if not isinstance(edges,dict):return 'unknown_edge'
        if not all(_finite_number(v) for v in (obj_x,obj_y)):return 'unknown_edge'
        if abs(obj_x)>edges['extent_m']:return 'unknown_edge'
        nx,ny=edges['normal_vehicle'];rel=nx*obj_x+ny*obj_y
        return LiveMap.tier_rel(edges['left_normal_m'],edges['right_normal_m'],rel,margin_m)


_MAP = None


def live_map(path):
    """Process-wide singleton (the engine queries every frame)."""
    global _MAP
    if _MAP is None or _MAP[0] != path:
        try: _MAP = (path, LiveMap(path))
        except Exception as e: _MAP = (path, None); _MAP_ERR[0] = str(e)
    return _MAP[1]


_MAP_ERR = ['']

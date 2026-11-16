"""Versioned, deliberately narrow parameter interface for gated proposals."""
import hashlib,json,math
DEFAULT_POLICY={'cluster_cell_m':.55,'min_cluster_points':4}
POLICY_SCHEMA=1

def canonical_policy(value=None):
    value=DEFAULT_POLICY if value is None else value
    if not isinstance(value,dict) or set(value)!=set(DEFAULT_POLICY):raise ValueError('Unknown perception policy fields')
    cell=value['cluster_cell_m'];count=value['min_cluster_points']
    if isinstance(cell,bool) or not isinstance(cell,(int,float)) or not math.isfinite(cell) or not .3<=cell<=.8:raise ValueError('Cluster cell must be 0.3–0.8 m')
    if type(count) is not int or not 4<=count<=8:raise ValueError('Minimum cluster points must be an integer from 4 to 8')
    return {'cluster_cell_m':float(cell),'min_cluster_points':count}

def policy_digest(value):
    return hashlib.sha256(json.dumps(canonical_policy(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()

"""Bind learned artifacts to the actual feature, teacher and consumer code."""
import hashlib,json,sys,re
from pathlib import Path

FILES=('radar.py','learning/radar_forecast.py','learning/model_registry.py',
       'learning/radar_live.py','learning/datasets.py','learning/training.py')

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def build_contract(source_root):
    root=Path(source_root)
    payload={'schema_version':1,'purpose':'radar measurement forecast feature/teacher/inference contract',
             'files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}}
    return {**payload,'runtime_signature':_digest(payload)}

def load_contract(assets):
    path=Path(assets)/'learning-runtime-contract.json'
    if not path.exists() and not getattr(sys,'frozen',False):
        return build_contract(Path(__file__).resolve().parents[1])
    if path.stat().st_size>32_000:raise ValueError('Learning contract exceeds size limit')
    value=json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value,dict) or type(value.get('schema_version')) is not int or value.get('schema_version')!=1
            or not isinstance(value.get('files'),dict) or set(value['files'])!=set(FILES)):
        raise ValueError('Incompatible learning contract')
    if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in value['files'].values()):
        raise ValueError('Malformed learning source hashes')
    if value.get('runtime_signature')!=_digest({k:v for k,v in value.items() if k!='runtime_signature'}):raise ValueError('Learning contract integrity failed')
    return value

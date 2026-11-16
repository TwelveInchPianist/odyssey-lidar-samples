"""Integrity of immutable folder-bundle files; mutable settings remain outside it.

This detects damaged/mismatched package contents, not malicious administrator
replacement of both the manifest and program. The installer supplies the trust
boundary by verifying its complete payload before atomic directory replacement.
"""
import hashlib,json
from pathlib import Path,PurePosixPath
MANIFEST='BUNDLE-MANIFEST.json'
def digest(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(executable):
 exe=Path(executable).resolve();root=exe.parent;raw=(root/MANIFEST).read_bytes()
 data=json.loads(raw);files=data.get('files')
 if data.get('schema')!=1 or not isinstance(files,list) or not files:raise ValueError('Invalid bundle manifest')
 names=set();total=0
 for item in files:
  name=item.get('path');parts=PurePosixPath(name).parts if isinstance(name,str) else ()
  if not parts or '\\' in name or ':' in name or name.startswith('/') or any(x in ('..','.') for x in parts):raise ValueError('Invalid managed path')
  if name.casefold() in names:raise ValueError('Duplicate managed path')
  names.add(name.casefold());target=root.joinpath(*parts)
  if not (name==exe.name or name=='Open-OdysseyFCW.exe' or parts[0]=='_internal'):raise ValueError('Mutable/unexpected managed path')
  if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root):raise ValueError('Missing or redirected managed file: '+name)
  size=target.stat().st_size
  if type(item.get('size')) is not int or size!=item['size'] or digest(target)!=item.get('sha256'):raise ValueError('Bundle mismatch: '+name)
  total+=size
 expected={x for x in names if x.startswith('_internal/')}
 actual={p.relative_to(root).as_posix().casefold() for p in (root/'_internal').rglob('*') if p.is_file()}
 if actual!=expected:raise ValueError('Unexpected or missing internal bundle files')
 if exe.name.casefold() not in names:raise ValueError('Executable missing from bundle manifest')
 return {'bundle_sha256':hashlib.sha256(raw).hexdigest(),'bundle_files':len(files),'bundle_bytes':total,'bundle_state':'complete'}

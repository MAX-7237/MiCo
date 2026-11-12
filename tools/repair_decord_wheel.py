"""Repair the published decord 0.6.0 wheel's inconsistent internal Python tag.

The published filename is py3-none, but its WHEEL metadata declares cp36-cp36m.
Only WHEEL and RECORD change; runtime files and libdecord.so remain identical.
Run after requirements installation, before pip check. Other versions are untouched.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

NAME='decord-0.6.0-py3-none-manylinux2010_x86_64.whl'
TAG='decord-0.6.0.dist-info/WHEEL'
RECORD='decord-0.6.0.dist-info/RECORD'
OLD=b'Tag: cp36-cp36m-manylinux2010_x86_64'
NEW=b'Tag: py3-none-manylinux2010_x86_64'

def repair(original, destination):
    with zipfile.ZipFile(original) as archive:
        contents={n:archive.read(n) for n in archive.namelist()}
    if original.name!=NAME or OLD not in contents[TAG]:
        raise ValueError('Unexpected wheel; refusing to change its compatibility metadata')
    contents[TAG]=contents[TAG].replace(OLD,NEW)
    rows=[]
    for name,data in sorted(contents.items()):
        if name==RECORD: continue
        digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        rows.append([name,'sha256='+digest,str(len(data))])
    rows.append([RECORD,'',''])
    stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\n').writerows(rows)
    contents[RECORD]=stream.getvalue().encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items(): archive.writestr(name,data)
    with zipfile.ZipFile(original) as a,zipfile.ZipFile(destination) as b:
        changed=[name for name in a.namelist() if a.read(name)!=b.read(name)]
    assert set(changed)=={TAG,RECORD},changed
    proof={'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
           'fixed_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
           'changed_members':changed,'runtime_files_unchanged':True}
    print(json.dumps(proof,indent=2))
    return proof

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel',type=Path,help='Repair this original wheel without installing it')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.wheel:
        if a.output_dir is None:p.error('--wheel requires --output-dir')
        target=a.output_dir/NAME
        if target.resolve()==a.wheel.resolve():p.error('Use separate original and repaired directories')
        repair(a.wheel,target);return
    try: distribution=importlib.metadata.distribution('decord')
    except importlib.metadata.PackageNotFoundError:
        print('decord is not installed; no repair needed');return
    if distribution.version!='0.6.0' or OLD.decode() not in (distribution.read_text('WHEEL') or ''):
        print('No affected decord wheel metadata detected');return
    with tempfile.TemporaryDirectory(prefix='mico-decord-') as temp:
        root=Path(temp);original=root/'original';fixed=root/'fixed'/NAME
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','decord==0.6.0','--dest',str(original)],check=True)
        repair(original/NAME,fixed)
        # Replace only metadata for the exact installed version; retain pinned dependencies.
        subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps',str(fixed)],check=True)

if __name__=='__main__':main()

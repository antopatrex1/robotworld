#!/usr/bin/env python3
"""Fetch pinned public assets, or verify existing assets offline with --check.

Only Menagerie's G1/Shadow models, SOMA's walking sample and the requested public
Marble scene are installed. No API key, world generation or GPU stack is used.
Existing checkouts/files must match their pins; they are never reset or deleted.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_ROOT=Path(__file__).resolve().parents[1]
WALK='assets/motions/csv/Neutral_walk_forward_002__A057.csv'
WORLD='warm_living_room'
ALLOWED_DOWNLOAD_HOSTS={'media.githubusercontent.com','cdn.marble.worldlabs.ai'}


def git(directory,*arguments,offline=False):
    env=os.environ.copy()
    env['GIT_LFS_SKIP_SMUDGE']='1'
    env['GIT_TERMINAL_PROMPT']='0'
    if offline:env['GIT_NO_LAZY_FETCH']='1'
    result=subprocess.run(['git','-C',str(directory),*arguments],check=True,
                          capture_output=True,env=env)
    return result.stdout


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def download(url,destination,digest,size,check=False,pointer=None):
    if destination.exists():
        if destination.stat().st_size==size and sha256(destination)==digest:return
        if pointer is None or destination.read_bytes()!=pointer:
            raise ValueError(f'{destination} differs from its asset pin; preserving it. Use another --root.')
    if check:raise ValueError(f'Missing materialized asset: {destination}')
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError('Asset URL is outside the pinned public download hosts.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.download-',delete=False) as output:
            temporary=Path(output.name)
            with urlopen(Request(url,headers={'User-Agent':'robotworld-assets/1'}),timeout=90) as response:
                if urlparse(response.geturl()).hostname not in ALLOWED_DOWNLOAD_HOSTS:
                    raise ValueError('Unexpected download redirect host.')
                length=0
                while chunk:=response.read(1024*1024):
                    length+=len(chunk)
                    if length>size:raise ValueError('Downloaded asset exceeds its pinned size.')
                    output.write(chunk)
        if temporary.stat().st_size!=size or sha256(temporary)!=digest:
            raise ValueError('Downloaded asset failed its SHA-256/size check.')
        # Replacement is allowed only for the exact pinned LFS pointer.
        if destination.exists() and (pointer is None or destination.read_bytes()!=pointer):
            raise ValueError('Asset changed while downloading; preserving the existing file.')
        os.replace(temporary,destination)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def verify_repository(directory,pin,paths,check):
    if git(directory,'rev-parse','HEAD',offline=check).decode().strip()!=pin['commit']:
        raise ValueError(f'{directory} is on a different revision; preserving it. Use another --root.')
    entries=git(directory,'ls-tree','-r','-z','HEAD','--',*paths,offline=check).split(b'\0')
    seen=[]
    for entry in filter(None,entries):
        meta,name=entry.split(b'\t',1)
        mode,kind,oid=meta.decode().split()
        relative=name.decode();path=directory/relative
        if kind!='blob' or mode not in ('100644','100755'):
            raise ValueError(f'Unsupported asset tree entry: {relative}')
        seen.append(relative)
        if relative==WALK:
            pointer=git(directory,'show','HEAD:'+relative,offline=check)
            lfs=re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?',pointer)
            if lfs:
                repo=urlparse(pin['url']).path.removesuffix('.git').strip('/')
                url=f'https://media.githubusercontent.com/media/{repo}/{pin["commit"]}/{relative}'
                download(url,path,lfs[1].decode(),int(lfs[2]),check,pointer)
                continue
        if not path.is_file():raise ValueError(f'Missing pinned asset: {path}')
        contents=path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
        if actual!=oid:raise ValueError(f'{path} differs from its pinned Git object; preserving it.')
    for required in paths:
        if not any(p==required or p.startswith(required+'/') for p in seen):
            raise ValueError(f'Pinned repository is missing {required}')


def ensure_repository(root,name,pin,paths,check):
    destination=root/'vendor'/name
    if destination.exists():
        verify_repository(destination,pin,paths,check)
        print(f'Verified {name} at {pin["commit"][:12]}',flush=True)
        return
    if check:raise ValueError(f'Missing checkout: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix=f'.{name}-') as temporary:
        checkout=Path(temporary)/'checkout'
        checkout.mkdir()
        git(checkout,'init','--quiet')
        git(checkout,'remote','add','origin',pin['url'])
        git(checkout,'config','core.sparseCheckout','true')
        git(checkout,'config','core.sparseCheckoutCone','false')
        info=checkout/'.git/info/sparse-checkout'
        info.parent.mkdir(exist_ok=True)
        info.write_text(''.join('/'+path+'\n' for path in paths))
        print(f'Fetching pinned {name} assets…',flush=True)
        git(checkout,'fetch','--depth=1','--filter=blob:none','origin',pin['commit'])
        git(checkout,'-c','advice.detachedHead=false','checkout','--detach','FETCH_HEAD')
        verify_repository(checkout,pin,paths,check=False)
        checkout.rename(destination)


def write_missing(path,contents,check=False):
    if path.exists():return
    if check:raise ValueError(f'Missing generated file: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:output.write(contents)


def ensure_world(root,check):
    lock=json.loads((SOURCE_ROOT/'configs/public_world.lock.json').read_text())
    world,asset=lock['world'],lock['asset']
    folder=root/'data/worlds'/WORLD
    splat=folder/'scene.spz'
    download(world['generation_output']['spz_urls']['150k'],splat,asset['sha256'],asset['size'],check)
    metadata=root/'data/worlds/requested_public_world.json'
    write_missing(metadata,json.dumps(world,indent=2).encode(),check)
    existing=json.loads(metadata.read_text())
    if existing['id']!=world['id'] or existing['generation_output']['spz_urls']['150k']!=world['generation_output']['spz_urls']['150k']:
        raise ValueError('Existing public-world metadata does not match its pin; preserving it.')
    manifest_path=folder/'manifest.json'
    config_path=root/'configs/environments'/f'{WORLD}.json'
    if not manifest_path.exists() or not config_path.exists():
        if check:raise ValueError('Public-world manifest/config missing; run bootstrap without --check.')
        # Generate in a new temporary directory, then publish only missing files.
        # The importer cannot overwrite an existing manifest or environment edit.
        with tempfile.TemporaryDirectory(dir=folder.parent,prefix='.public-world-') as temporary:
            staging=Path(temporary)
            (staging/'data/worlds'/WORLD).mkdir(parents=True)
            (staging/'configs/environments').mkdir(parents=True)
            shutil.copyfile(splat,staging/'data/worlds'/WORLD/'scene.spz')
            spec=importlib.util.spec_from_file_location('robotworld_public_importer',SOURCE_ROOT/'scripts/import_public_scene.py')
            importer=importlib.util.module_from_spec(spec);spec.loader.exec_module(importer)
            importer.ROOT=staging
            # Use the checked-in minimal metadata, not any local additional fields.
            staged_metadata=staging/'public.json'
            staged_metadata.write_text(json.dumps(world))
            importer.prepare(staged_metadata)
            write_missing(manifest_path,(staging/'data/worlds'/WORLD/'manifest.json').read_bytes())
            write_missing(config_path,(staging/'configs/environments'/f'{WORLD}.json').read_bytes())
    manifest=json.loads(manifest_path.read_text())
    config=json.loads(config_path.read_text())
    if manifest['world_id']!=world['id'] or manifest['splat_file']!=f'data/worlds/{WORLD}/scene.spz':
        raise ValueError('Existing public-world manifest references another scene; preserving it.')
    if config.get('world_manifest')!=f'data/worlds/{WORLD}/manifest.json':
        raise ValueError('Existing public-world config references another manifest; preserving it.')
    print('Verified exact public Marble scene and 150k SPZ SHA-256',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=SOURCE_ROOT,help='Asset destination; useful for an isolated installation.')
    parser.add_argument('--lab-only',action='store_true',help='Install robot and walking assets without downloading the room.')
    parser.add_argument('--check',action='store_true',help='Verify installed assets offline without changing files.')
    args=parser.parse_args();root=args.root.resolve()
    if not args.check:root.mkdir(parents=True,exist_ok=True)
    sources=json.loads((SOURCE_ROOT/'sources.lock.json').read_text())
    ensure_repository(root,'mujoco_menagerie',sources['mujoco_menagerie'],['unitree_g1','shadow_hand'],args.check)
    ensure_repository(root,'soma-retargeter',sources['soma-retargeter'],['LICENSE',WALK],args.check)
    for name in ('lab','kitchen'):
        write_missing(root/'configs/environments'/f'{name}.json',
                      (SOURCE_ROOT/'configs/environments'/f'{name}.json').read_bytes(),args.check)
    if not args.lab_only:ensure_world(root,args.check)
    print('Assets ready. '+('Launch with --environment lab.' if args.lab_only else 'The default public scene is ready.'),flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        detail=error.stderr.decode(errors='replace').strip() if isinstance(error,subprocess.CalledProcessError) and error.stderr else str(error)
        print('Asset setup failed: '+detail,file=sys.stderr)
        sys.exit(1)

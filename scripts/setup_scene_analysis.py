#!/usr/bin/env python3
"""Download the pinned local scene detector; run after installing requirements-scene.txt."""
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt'
SHA256 = '1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3'


def main():
    destination = ROOT/'data/models/yolo11s-seg.pt'
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('Existing model differs from its pin; preserving it.')
        print('Scene-analysis model verified.')
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as output:
            temporary=Path(output.name)
            with urlopen(URL,timeout=60) as response:
                size=0
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>30*1024*1024:raise ValueError('Model download exceeds expected size')
                    output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('Model download failed checksum verification')
        temporary.replace(destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    print('Scene-analysis model ready.')


if __name__=='__main__':main()

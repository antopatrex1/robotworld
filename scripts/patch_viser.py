#!/usr/bin/env python3
"""Preserve every GUI message in Viser 1.1.0, including text before submit.

Run with the project's Python after installing requirements. The installed source
and shipped, compressed browser bundle are both patched; no npm build is needed.
Use --check to verify an existing installation without writing files.
"""
import argparse
import base64
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess

import viser
import zstandard


MARKER='robot-world: reliable-gui-v1'
SOURCE_OLD='''    if (viewerMutable.sendMessage === null) return;
    latestMessage = message;'''
SOURCE_NEW='''    if (viewerMutable.sendMessage === null) return;
    // robot-world: reliable-gui-v1
    // GUI commits must never replace a pending input value in this shared
    // throttle. Send GUI input and commit events in order; retain throttling
    // for the high-frequency camera and drag messages that need it.
    if (message.type.startsWith("Gui")) {
      flush();
      viewerMutable.sendMessage(message);
      return;
    }
    latestMessage = message;'''
JS_OLD='function NR(e,t){let n=!0,r=!1,i=null,a=null;function o(s){let c=e.mutable.current;c.sendMessage!==null&&(i=s,n?(c.sendMessage(s),r=!1,n=!1,a=setTimeout(()=>{a=null,n=!0,r&&i&&o(i)},t)):r=!0)}function s(){e.mutable.current.sendMessage!==null&&i!==null&&r&&(e.mutable.current.sendMessage(i),i=null,r=!1)}function c(){a!==null&&(clearTimeout(a),a=null),n=!0,r=!1,i=null}return{send:o,flush:s,cancel:c}}'
JS_NEW=JS_OLD.replace(
    'function o(s){let c=e.mutable.current;',
    'function o(s){/* '+MARKER+' */let c=e.mutable.current;'
    'if(c.sendMessage!==null&&s.type.startsWith("Gui")){'
    'i!==null&&r&&(c.sendMessage(i),i=null,r=!1);c.sendMessage(s);return}',
)


def replace_once(text,old,new,label):
    if new in text:
        return text
    if text.count(old)!=1:
        raise RuntimeError(f'{label} differs from the expected Viser 1.1.0 build; refusing to patch it.')
    return text.replace(old,new,1)


def verify_ordering(javascript):
    node=shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required to verify the browser patch before applying it.')
    # Exercise the real extracted function, with deterministic simulated timers.
    harness='''
import assert from 'node:assert/strict';
const timers=[];
globalThis.setTimeout=(callback)=>{timers.push(callback);return timers.length;};
globalThis.clearTimeout=()=>{};
FUNCTION
for (const submit of [
  {type:'GuiFormSubmitMessage',uuid:'prompt-form'},
  {type:'GuiUpdateMessage',uuid:'run-button',updates:{value:true}},
]) {
  const sent=[];
  const viewer={mutable:{current:{sendMessage:(message)=>sent.push(message)}}};
  const sender=NR(viewer,50);
  const first={type:'GuiUpdateMessage',uuid:'prompt',updates:{value:'grab the red mu'}};
  const last={type:'GuiUpdateMessage',uuid:'prompt',updates:{value:'grab the red mug'}};
  const dirty={type:'GuiFormDirtyMessage',uuid:'prompt-form'};
  sender.send(first);sender.send(last);sender.send(dirty);sender.send(submit);
  assert.deepEqual(sent,[first,last,dirty,submit]);
  while(timers.length) timers.shift()();
  assert.deepEqual(sent,[first,last,dirty,submit]);
}
const sent=[];
const viewer={mutable:{current:{sendMessage:(message)=>sent.push(message)}}};
const sender=NR(viewer,50);
sender.send({type:'ViewerCameraMessage',sequence:1});
sender.send({type:'ViewerCameraMessage',sequence:2});
sender.send({type:'ViewerCameraMessage',sequence:3});
assert.equal(sent.length,1,'Camera updates should remain throttled');
sender.send({type:'GuiFormSubmitMessage',uuid:'prompt-form'});
assert.deepEqual(sent.map(m=>m.sequence??m.type),[1,3,'GuiFormSubmitMessage']);
while(timers.length) timers.shift()();
assert.equal(sent.length,3,'Flushing must not duplicate the pending message');
viewer.mutable.current.sendMessage=null;
sender.send({type:'GuiUpdateMessage',uuid:'prompt',updates:{value:'stop'}});
assert.equal(sent.length,3);
'''.replace('FUNCTION',javascript)
    subprocess.run([node,'--input-type=module','-e',harness],check=True,capture_output=True,text=True)


def apply(check=False):
    if importlib.metadata.version('viser')!='1.1.0':
        raise RuntimeError('This patch is pinned to Viser 1.1.0; review it before using another version.')
    client=Path(viser.__file__).resolve().parent/'client'
    source_path=client/'src/WebsocketUtils.ts'
    bundle_path=client/'build/index.html'
    source=source_path.read_text()
    html=bundle_path.read_text()
    encoded=re.search(r'data-c="([^"]+)"',html)
    if encoded is None:
        raise RuntimeError('Expected a compressed Viser browser bundle.')
    javascript=zstandard.ZstdDecompressor().decompress(base64.b64decode(encoded[1])).decode()
    patched_source=replace_once(source,SOURCE_OLD,SOURCE_NEW,'Source')
    patched_javascript=replace_once(javascript,JS_OLD,JS_NEW,'Browser bundle')
    verify_ordering(JS_NEW)
    if check:
        if source!=patched_source or javascript!=patched_javascript:
            raise RuntimeError('The Viser GUI patch has not been applied. Run scripts/patch_viser.py.')
    elif source!=patched_source or javascript!=patched_javascript:
        # All shape checks and JavaScript behavior checks pass before any write.
        payload=patched_javascript.encode()
        compressed=base64.b64encode(zstandard.ZstdCompressor(level=10).compress(payload)).decode()
        patched_html=html[:encoded.start(1)]+compressed+html[encoded.end(1):]
        patched_html,count=re.subn(r'data-cs="\d+"',f'data-cs="{len(payload)}"',patched_html,count=1)
        if count!=1:
            raise RuntimeError('Missing decompressed JavaScript size; refusing to patch the bundle.')
        source_path.write_text(patched_source)
        bundle_path.write_text(patched_html)
    print(json.dumps({'viser':'1.1.0','gui_message_ordering':'verified','source_and_bundle':'patched','mode':'check' if check else 'apply'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    apply(parser.parse_args().check)

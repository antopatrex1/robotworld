#!/usr/bin/env python3
"""Prepare an already downloaded public Marble SPZ with approximate collisions."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_world.splats import decode_spz
from robot_world.scene import ROOT

def prepare(metadata_path, name='warm_living_room'):
    metadata=json.loads(Path(metadata_path).read_text())
    folder=ROOT/'data/worlds'/name
    output=metadata['generation_output']
    semantics=output['semantics_metadata']
    scale=semantics['metric_scale_factor']
    transform=np.eye(4)
    # Marble exports OpenCV Y-down. ground_plane_offset is in scaled metres.
    transform[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]])*scale
    transform[2,3]=semantics['ground_plane_offset']
    source=decode_spz(folder/'scene.spz')
    points=source['centers']@transform[:3,:3].T+transform[:3,3]
    valid=(points[:,2]>.18)&(points[:,2]<1.6)&(source['opacities'][:,0]>.5)
    bins={}
    cell=.20
    for x,y,z in points[valid]:
        key=tuple(np.floor(np.array([x,y])/cell).astype(int))
        count,height=bins.get(key,(0,0.))
        bins[key]=(count+1,max(height,float(z)))
    boxes=[]
    for (i,j),(count,height) in sorted(bins.items()):
        if count<8: continue
        boxes.append({'name':f'room_proxy_{i}_{j}', 'pos':f'{(i+.5)*cell} {(j+.5)*cell} {height/2}',
                      'size':f'{cell/2} {cell/2} {height/2}', 'rgba':'0.2 0.7 0.9 0'})
    manifest={'world_id':metadata['id'],'name':metadata['display_name'],
      'marble_url':'https://marble.worldlabs.ai/world/'+metadata['id'],
      'splat_file':str((folder/'scene.spz').relative_to(ROOT)), 'splat_url':output['spz_urls']['150k'],
      'world_to_sim':transform.tolist(),'units':'metres','source_up':'-Y','sim_up':'Z',
      'calibration':'Metadata metric scale and ground offset; approximate, visually checked.',
      'collision_mode':'Approximate 20 cm splat occupancy columns; source has no collision mesh.',
      'visual_meshes':[],'collision_boxes':boxes,'bounds':np.quantile(points,[.01,.99],axis=0).tolist()}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'configs/environments'/f'{name}.json').write_text(json.dumps({'name':metadata['display_name'],
      'floor_color':'0.31 0.27 0.23','floor_alt':'0.34 0.3 0.26',
      'world_manifest':str((folder/'manifest.json').relative_to(ROOT)),'furniture':[]},indent=2))
    print(f'{len(boxes)} approximate collision columns; bounds {manifest["bounds"]}')
    return manifest

if __name__=='__main__': prepare(ROOT/'data/worlds/requested_public_world.json')

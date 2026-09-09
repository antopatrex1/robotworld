"""Approximate, manually annotated tabletop snapshot registration.

This maps RGB points ON the table to XY metres. It neither aligns raw depth
nor estimates the camera's 3D pose. Native RGB/depth buffers remain unchanged.
"""
import json
import xml.etree.ElementTree as ET

import numpy as np


def homography(pixels, size):
    pixels = np.asarray(pixels, dtype=float)
    width, length = size
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all() or min(size) <= 0:
        raise ValueError('Four finite ordered paper corners and positive dimensions required')
    destination = [[0, 0], [width, 0], [width, length], [0, length]]
    rows, values = [], []
    for (x, y), (u, v) in zip(pixels, destination):
        rows.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        values.extend([u, v])
    try:
        return np.append(np.linalg.solve(rows, values), 1).reshape(3, 3)
    except np.linalg.LinAlgError as error:
        raise ValueError('Paper corners must form a nondegenerate quadrilateral') from error


def project(matrix, points):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError('Point lies at the tabletop projection horizon')
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def table_to_world(calibration, points):
    points = np.asarray(points, dtype=float)
    return (np.asarray(calibration['world_corner']) +
            points[:, :1] * np.asarray(calibration['world_u']) +
            points[:, 1:] * np.asarray(calibration['world_v']))


def apply_camera_layout(world, asset, root, config):
    calibration = json.loads((root / config['camera_layout']).read_text())
    if 'objects' in calibration:
        apply_detected_layout(world, asset, calibration, config)
        return
    matrix = homography(calibration['paper_pixels'], calibration['paper_size_m'])

    def mapped(pixels):
        return table_to_world(calibration, project(matrix, pixels))

    # Remove the demo props rather than leave invisible collision bodies behind.
    from .objects import OBJECTS
    for name in OBJECTS:
        body = world.find(f"body[@name='{name}']")
        if name != 'mug' and body is not None:
            world.remove(body)
    mug = world.find("body[@name='mug']")
    center = mapped([calibration['mug_base_pixel']])[0]
    center[2] += .04  # Existing cup's half-height; not measured from this image.
    mug.set('pos', ' '.join(map(str, center)))
    for geom in mug.findall('geom'):
        geom.set('rgba', '.035 .04 .045 1')

    def tile(name, pixels, color, thickness):
        corners = mapped(pixels)
        vertices = np.vstack([corners, corners + [0, 0, thickness]])
        faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                 [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                 [2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        ET.SubElement(asset, 'mesh', name=name+'_mesh',
                      vertex=' '.join(map(str, vertices.ravel())),
                      face=' '.join(map(str, np.asarray(faces).ravel())))
        ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name+'_mesh',
                      rgba=color, contype='0', conaffinity='0')

    tile('calibration_paper', calibration['paper_pixels'], '.95 .94 .88 1', .0005)
    tile('observed_card', calibration['card_pixels'], '.20 .50 .56 1', .001)
    tile('observed_connector', calibration['connector_pixels'], '.85 .87 .85 1', .002)
    cable = mapped(calibration['cable_pixels']) + [0, 0, .002]
    for i, (a, b) in enumerate(zip(cable[:-1], cable[1:])):
        ET.SubElement(world, 'geom', name=f'observed_cable_{i}', type='capsule',
                      fromto=' '.join(map(str, np.r_[a,b])), size='.0015',
                      rgba='.80 .84 .81 1', contype='0', conaffinity='0')
    # Calibration axes are a visible tabletop origin and orientation reference.
    origin = np.asarray(calibration['world_corner']) + [0, 0, .003]
    for axis, vector, color in [('u', calibration['world_u'], '.95 .3 .2 1'),
                                ('v', calibration['world_v'], '.2 .8 .4 1')]:
        end = origin + np.asarray(vector) * .07
        ET.SubElement(world, 'geom', name='paper_axis_'+axis, type='capsule',
                      fromto=' '.join(map(str, np.r_[origin, end])), size='.002',
                      rgba=color, contype='0', conaffinity='0')
    config['alignment_summary'] = {
        'method': calibration['method'], 'mug_table_xy_m': project(matrix, [calibration['mug_base_pixel']])[0].tolist(),
        'source_capture': calibration['source_capture'], 'limitations': calibration['limitations']}


def apply_detected_layout(world, asset, calibration, config):
    """Replace demo bodies with instances actually found in the new capture."""
    import copy
    from .objects import OBJECTS
    templates = {}
    for name in OBJECTS:
        body = world.find(f"body[@name='{name}']")
        if body is not None:
            templates[name] = copy.deepcopy(body)
            world.remove(body)
    matrix = homography(calibration['paper_pixels'], calibration['paper_size_m'])
    config['object_labels'] = {}
    config['graspable_objects'] = []
    for item in calibration['objects']:
        name, kind = item['id'], item['kind']
        xy = project(matrix,[item['base_pixel']])
        position = table_to_world(calibration,xy)[0]
        rgba = ' '.join(map(str,np.r_[np.asarray(item['color_rgb']) / 255.,1]))
        if kind in templates:
            body = copy.deepcopy(templates[kind])
            # Preserve the proven model dimensions while relocating its base.
            position[2] += float(body.get('pos').split()[2]) - .725
            for element in body.iter():
                if element.get('name'):
                    element.set('name',name+element.get('name')[len(kind):])
            for geom in body.findall('geom'):
                geom.set('rgba',rgba)
        else:
            body = ET.Element('body',name=name)
            ET.SubElement(body,'freejoint',name=name+'_free')
            position[2] += .018
            ET.SubElement(body,'geom',name=name+'_body',type='ellipsoid',size='.055 .033 .018',
                          rgba=rgba,mass='.08',friction='1 .01 .001')
        body.set('pos',' '.join(map(str,position)))
        world.append(body)
        config['object_labels'][name] = item['label']
        if kind in ('mug','bottle','apple'):
            config['graspable_objects'].append(name)
    width,length = calibration['paper_size_m']
    center = table_to_world(calibration,[[width/2,length/2]])[0]+[0,0,.0004]
    ET.SubElement(world,'geom',name='calibration_paper',type='box',
                  pos=' '.join(map(str,center)),size=f'{length/2} {width/2} .0004',
                  rgba='.95 .94 .88 1',contype='0',conaffinity='0')
    config['alignment_summary'] = {'method':calibration['method'],
        'source_capture':calibration['source_capture'], 'limitations':calibration['limitations'],
        'object_count':len(calibration['objects']), 'omitted':calibration.get('omitted',[])}

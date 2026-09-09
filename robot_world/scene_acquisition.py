"""Fresh RGB-D capture and local tabletop object analysis for Acquire Scene."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .table_alignment import homography, project, correct_table_orientation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/models/yolo11s-seg.pt'
CLASS_KIND = {'cup':'mug', 'bottle':'bottle', 'bowl':'bowl', 'apple':'apple',
              'orange':'ball', 'sports ball':'ball', 'book':'book',
              'remote':'remote', 'mouse':'mouse', 'cell phone':'proxy',
              'keyboard':'proxy', 'banana':'proxy', 'scissors':'proxy',
              'vase':'proxy', 'wine glass':'proxy', 'fork':'proxy',
              'knife':'proxy', 'spoon':'proxy'}


def capture_scene(socket_path, output_root=None):
    binary = ROOT / 'realsense/target/debug/realsense-studio'
    if not binary.is_file():
        raise ValueError('Build RealSense first, then start the camera helper.')
    output_root = Path(output_root or ROOT / 'realsense/captures')
    result = subprocess.run([str(binary), '--socket', str(socket_path), '--output', str(output_root), 'capture'],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError('Camera capture failed. Check the RealSense window and connection.')
    metadata = json.loads(result.stdout)
    if metadata['demo']:
        raise ValueError('Acquire Scene needs a real camera frame; the helper is in synthetic demo mode.')
    if abs(time.time() * 1000 - metadata['captured_unix_ms']) > 15000:
        raise ValueError('The camera frame is stale. Restart the camera helper and retry.')
    return metadata


def track_paper(rgb, reference):
    for level in (2, 3):
        corners = _track_paper_level(rgb, reference, level)
        if corners is not None:
            return corners
    return None


def _track_paper_level(rgb, reference, level):
    """Relocate previously verified corners in the new image, with round-trip checks."""
    import cv2
    capture = reference.get('source_capture')
    if not capture:
        return None
    source = Path(capture) / 'color.png'
    if not source.is_file():
        return None
    old = cv2.imread(str(source))
    if old is not None:
        old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if old is None or old.shape != gray.shape:
        return None
    points = np.asarray(reference['paper_pixels'], np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=level)
    moved, valid, error = cv2.calcOpticalFlowPyrLK(old, gray, points, None, **options)
    if moved is None or not valid.all() or np.max(error) > 30:
        return None
    back, valid_back, _ = cv2.calcOpticalFlowPyrLK(gray, old, moved, points.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
    if back is None or not valid_back.all() or np.max(np.linalg.norm(back-points, axis=2)) > 1.0:
        return None
    corners = moved.reshape(4, 2)
    height, width = gray.shape
    if not np.isfinite(corners).all() or np.max(np.linalg.norm(moved-points, axis=2)) > 100:
        return None
    if np.any(corners < 3) or np.any(corners[:, 0] > width-4) or np.any(corners[:, 1] > height-4):
        return None
    area = cv2.contourArea(corners)
    previous_area = cv2.contourArea(points)
    if not cv2.isContourConvex(corners) or not .7 < area/previous_area < 1.4:
        return None
    # The tracked area must still contain a bright, reasonably uniform sheet.
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask > 0]
    if len(pixels) < 500 or np.mean((pixels[:, 2] > 170) & (pixels[:, 1] < 155)) < .9:
        return None
    if np.std(pixels[:, 2]) > 22:
        return None
    return corners.astype(float)


def find_paper(rgb, reference, saturation=100):
    """Find new white paper corners near the known tabletop anchor, never reuse pixels."""
    import cv2
    height, width = rgb.shape[:2]
    if [width, height] != reference['image_size']:
        raise ValueError('Camera resolution changed; recalibrate the paper reference.')
    tracked = track_paper(rgb, reference)
    if tracked is not None:
        return tracked
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    reference_pixels = np.asarray(reference['paper_pixels'],dtype=int)
    left,top = np.maximum(reference_pixels.min(axis=0)-30,0)
    right,bottom = np.minimum(reference_pixels.max(axis=0)+30,[width-1,height-1])
    brightness = max(170, min(235, float(np.percentile(hsv[top:bottom+1,left:right+1,2],95))*.92))
    # Daylight gives the white sheet a blue cast; brightness separates it from wood.
    mask = cv2.inRange(hsv, np.array([0,0,int(brightness)]), np.array([179,saturation,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    previous = np.asarray(reference['paper_pixels'], dtype=np.float32)
    expected_center = previous.mean(axis=0)
    expected_area = abs(cv2.contourArea(previous))
    candidates = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if not .7 * expected_area < area < 3 * expected_area:
            continue
        perimeter = cv2.arcLength(hull, True)
        quad = cv2.approxPolyDP(hull, .025 * perimeter, True).reshape(-1,2)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        # Camera stays on the same side: top pair is the far edge.
        far, near = np.array_split(quad[np.argsort(quad[:,1])], 2)
        far = far[np.argsort(far[:,0])]
        near = near[np.argsort(near[:,0])]
        ordered = np.array([far[0], far[1], near[1], near[0]],dtype=float)
        center_distance = np.linalg.norm(ordered.mean(axis=0)-expected_center)
        if center_distance > 100 or np.any(ordered[:,0] < 3) or np.any(ordered[:,0] > width-4):
            continue
        if np.any(ordered[:,1] < 3) or np.any(ordered[:,1] > height-4):
            continue
        if np.linalg.norm(far[1]-far[0]) < 80 or ordered[2:,1].mean()-ordered[:2,1].mean() < 15:
            continue
        candidates.append((center_distance + 30*abs(np.log(area/expected_area)), ordered))
    if not candidates and saturation == 100:
        return find_paper(rgb, reference, saturation=150)
    if not candidates:
        raise ValueError('I could not locate the reference paper. Keep the 210 × 147 mm sheet flat at the same table corner, with all four corners visible.')
    candidates.sort(key=lambda item:item[0])
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<10:
        raise ValueError('More than one possible reference sheet is visible. Leave only the calibration sheet near the table corner.')
    return candidates[0][1]


def color_name(rgb):
    r,g,b = rgb
    if max(rgb)<65:return 'black'
    if min(rgb)>185:return 'white'
    if max(rgb)-min(rgb)<30:return 'gray'
    if b>r*1.1 and g>r*1.1:return 'blue'
    if r>g*1.25 and r>b*1.25:return 'red'
    if g>r*1.15 and g>b*1.1:return 'green'
    if r>b*1.5 and g>b*1.5:return 'yellow'
    return 'dark' if np.mean(rgb)<110 else 'colored'


def analyze_scene(metadata, reference, model=None):
    """Infer new objects, map only their table contact points, omit cropped items."""
    import cv2
    reference = correct_table_orientation(reference)
    rgb = np.asarray(Image.open(metadata['color']).convert('RGB'))
    corners = find_paper(rgb, reference)
    matrix = homography(corners, reference['paper_size_m'])
    if model is None:
        if not MODEL.exists():
            raise ValueError('The local scene-analysis model is missing. Run scripts/setup_scene_analysis.py once.')
        from ultralytics import YOLO
        model = YOLO(str(MODEL))
    prediction = model.predict(source=rgb[:,:,::-1].copy(), device='cpu', imgsz=640,
                               conf=.25, verbose=False, retina_masks=True, agnostic_nms=True)[0]
    objects, omitted, counts = [], [], {}
    height, width = rgb.shape[:2]
    if prediction.masks is not None:
        for box, polygon in zip(prediction.boxes, prediction.masks.xy):
            category = prediction.names[int(box.cls.item())]
            if category not in CLASS_KIND:
                continue
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            if min(x1,y1)<3 or x2>width-4 or y2>height-4:
                omitted.append(category+' (cropped)')
                continue
            polygon = np.asarray(polygon)
            if len(polygon)<3:
                continue
            bottom = polygon[polygon[:,1] >= np.percentile(polygon[:,1],90)]
            base = [float(np.median(bottom[:,0])), float(np.max(bottom[:,1]))]
            xy = project(matrix,[base])[0]
            # Reject background objects and footprints beyond this virtual table.
            if not (0.035 <= xy[0] <= .765 and .035 <= xy[1] <= .645):
                omitted.append(category+' (outside the mapped tabletop)')
                continue
            region = np.zeros((height,width),np.uint8)
            cv2.fillPoly(region,[polygon.astype(np.int32)],1)
            pixels = rgb[region.astype(bool)]
            color = np.median(pixels,axis=0).astype(int).tolist()
            confidence=float(box.conf.item())
            kind = CLASS_KIND[category] if confidence>=.5 else 'proxy'
            counts[kind] = counts.get(kind,0)+1
            name = kind if counts[kind]==1 else f'{kind}_{counts[kind]}'
            noun = ('mug' if category=='cup' else category) if confidence>=.5 else 'object'
            label = f'Observed {color_name(color)} {noun}'
            if any(item['label']==label for item in objects):
                label += f' {counts[kind]}'
            objects.append({'id':name, 'kind':kind, 'label':label, 'base_pixel':base,
                            'color_rgb':color, 'confidence':float(box.conf.item()),
                            'category':category, 'table_xy_m':xy.tolist()})
    result = {key:reference[key] for key in ('image_size','paper_size_m','world_corner','world_u','world_v')}
    result.update(method='local YOLO11 segmentation and newly detected paper corners; approximate tabletop snapshot',
                  paper_pixels=corners.tolist(), corner_order='far-left, far-right, near-right, near-left',
                  source_capture=metadata['directory'], captured_unix_ms=metadata['captured_unix_ms'],
                  objects=objects, omitted=omitted,
                  limitations=['Object shapes, sizes and masses are simulation approximations.',
                               'Paper-based placement is approximate and not full RGB-D calibration.',
                               'Unsupported objects such as cables and small flat cards may be omitted.',
                               'This is a snapshot, not continuous tracking.'])
    validate_layout(result)
    return result


def validate_layout(layout):
    """Reject malformed, overlapping or off-table proposals before changing the scene."""
    corners = np.asarray(layout['paper_pixels'],dtype=float)
    matrix = homography(corners,layout['paper_size_m'])
    if len(layout['objects'])>20:
        raise ValueError('Too many objects detected; use a less cluttered tabletop.')
    names = set()
    for item in layout['objects']:
        if item['id'] in names or item['kind'] not in set(CLASS_KIND.values()):
            raise ValueError('Invalid object identifiers in scene analysis')
        names.add(item['id'])
        position = project(matrix,[item['base_pixel']])[0]
        if not (.035 <= position[0] <= .765 and .035 <= position[1] <= .645):
            raise ValueError('An object could not be placed within the virtual table.')
        item['table_xy_m'] = position.tolist()
    for i,item in enumerate(layout['objects']):
        for other in layout['objects'][:i]:
            if np.linalg.norm(np.array(item['table_xy_m'])-other['table_xy_m'])<.09:
                raise ValueError('Two detected objects are too close for the current simulation shapes. Separate them slightly and acquire again.')


class SceneAcquirer:
    """One model instance; caller serializes jobs and commits on simulation thread."""
    def __init__(self):
        self.model = None

    def acquire(self, socket_path, reference, progress):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ValueError('Local analysis is not installed. Install requirements-scene.txt and run scripts/setup_scene_analysis.py.') from error
        progress('Capturing a fresh RGB-D frame…')
        metadata = capture_scene(socket_path)
        progress('Finding the paper and identifying tabletop objects…')
        if not MODEL.exists():
            raise ValueError('Scene-analysis model missing. Run scripts/setup_scene_analysis.py once.')
        if self.model is None:
            self.model = YOLO(str(MODEL))
        layout = analyze_scene(metadata, reference, self.model)
        directory = Path(metadata['directory'])
        destination = directory / 'scene_layout.json'
        destination.write_text(json.dumps(layout,indent=2))
        progress(f'Building the new table layout ({len(layout["objects"])} objects)…')
        from .scene import build_scene
        prepared = build_scene('camera_table', camera_layout=destination, output_path=directory/'scene.xml')
        return destination, layout, prepared

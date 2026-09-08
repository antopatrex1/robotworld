"""Read the fixed-point SPZ v2 format (Niantic SPZ specification).

Only degree-zero color is displayed. The actual source scene remains unmodified.
Spec: https://github.com/nianticlabs/spz (MIT).
"""
import gzip
import struct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def decode_spz(path):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(128_000_001)
    if len(raw) > 128_000_000 or len(raw) < 16:
        raise ValueError('SPZ is oversized or truncated')
    magic, version, count, degree, bits, flags, reserved = struct.unpack('<III4B', raw[:16])
    if magic != 0x5053474E or version != 2 or degree > 3 or bits > 24 or count > 2_000_000:
        raise ValueError('Only SPZ version 2 with supported dimensions is accepted')
    expected = 16 + count * (19 + 3 * ((degree+1)**2-1))
    if len(raw) != expected:
        raise ValueError('SPZ payload length does not match its header')
    offset = 16
    def take(width):
        nonlocal offset
        result = np.frombuffer(raw, dtype=np.uint8, count=count*width, offset=offset).reshape(count, width)
        offset += count*width
        return result.astype(np.float32)
    packed = take(9).astype(np.int32).reshape(count,3,3)
    fixed = packed[:,:,0] | packed[:,:,1] << 8 | packed[:,:,2] << 16
    fixed = (fixed ^ 0x800000) - 0x800000
    positions = fixed.astype(np.float32) / (1 << bits)
    alpha = take(1) / 255
    colors = np.clip(.5 + .28209479177387814 * ((take(3)/255-.5)/.15), 0, 1)
    scales = np.exp(take(3)/16 - 10)
    xyz = take(3)/127.5 - 1
    quat = np.column_stack([xyz, np.sqrt(np.maximum(0, 1-(xyz*xyz).sum(axis=1)))])
    rotations = Rotation.from_quat(quat).as_matrix()
    covariances = (rotations * scales[:,None,:]**2) @ rotations.transpose(0,2,1)
    return {'centers': positions, 'covariances': covariances.astype(np.float32),
            'rgbs': (colors*255).astype(np.uint8), 'opacities': alpha}


def load_world_splats(manifest, root):
    splats = decode_spz(Path(root) / manifest['splat_file'])
    transform = np.asarray(manifest['world_to_sim'])
    linear = transform[:3,:3]
    splats['centers'] = (splats['centers'] @ linear.T + transform[:3,3]).astype(np.float32)
    splats['covariances'] = (linear @ splats['covariances'] @ linear.T).astype(np.float32)
    return splats

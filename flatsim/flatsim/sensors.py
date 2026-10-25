"""ROS-free sensor math: beam patterns, point clouds from first-hit distances, the IMU reading."""

from __future__ import annotations

import math

import attrs
import numpy as np

GRAVITY = 9.80665
MAX_CAMERA_COLUMNS = 160
MAX_CAMERA_ROWS = 24

LIDAR_POINT = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'], 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'], 'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
CAMERA_POINT = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4'], 'offsets': [0, 4, 8], 'itemsize': 12})


def fan(samples: int, min_angle: float, max_angle: float) -> np.ndarray:
    """Angles of `samples` beams spread from min_angle to max_angle inclusive."""
    return np.linspace(min_angle, max_angle, samples)


@attrs.frozen
class RayPlan:
    """Planar fan about the base z axis that carries the rays of a mounted sensor: per (row, column) ray its beam, its rise per meter and its sensor-frame offset per meter along that beam."""

    angle_min: float
    angle_increment: float
    beams: int
    beam: np.ndarray
    slope: np.ndarray
    offset: np.ndarray


def _arc(heading: np.ndarray) -> tuple[float, float]:
    """Start and length of the shortest arc that holds every heading."""
    ordered = np.sort(np.mod(heading.ravel(), 2.0 * math.pi))
    gaps = np.diff(ordered, append=ordered[0] + 2.0 * math.pi)
    widest = int(np.argmax(gaps))
    return math.remainder(float(ordered[(widest + 1) % len(ordered)]), 2.0 * math.pi), 2.0 * math.pi - float(gaps[widest])


def ray_plan(rotation: np.ndarray, azimuth: np.ndarray, slopes: np.ndarray) -> RayPlan:
    """Fan, spaced by the azimuth step, for the rays of sensor-frame `azimuth` and rise per meter `slopes` from a frame with `rotation` in the base."""
    slopes = np.broadcast_to(slopes, (len(slopes), len(azimuth)))
    sensor = np.stack([np.broadcast_to(np.cos(azimuth), slopes.shape), np.broadcast_to(np.sin(azimuth), slopes.shape), slopes], axis=-1)
    base = sensor @ rotation.T
    heading = np.arctan2(base[..., 1], base[..., 0])
    slope = base[..., 2] / np.hypot(base[..., 0], base[..., 1])
    step = abs(float(azimuth[1] - azimuth[0])) if len(azimuth) > 1 else 0.0
    angle_min, span = _arc(heading)
    if step > 0.0:
        gap = 2.0 * math.pi - span
        beam = np.rint((np.mod(heading - angle_min + gap / 2.0, 2.0 * math.pi) - gap / 2.0) / step).astype(np.intp)
    else:
        beam = np.zeros(slopes.shape, dtype=np.intp)
    along = angle_min + step * beam
    offset = np.stack([np.cos(along), np.sin(along), slope], axis=-1) @ rotation
    return RayPlan(angle_min, step, int(beam.max()) + 1, beam, slope, offset)


def ring_slopes(samples: int, min_angle: float, max_angle: float) -> np.ndarray:
    """Rise per meter of each lidar ring as a (rings, 1) column, one level ring for a 2D lidar."""
    if samples <= 1:
        return np.zeros((1, 1))
    return np.tan(fan(samples, min_angle, max_angle))[:, None]


def slant_range(distance: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Range along each ray that travels horizontal `distance` with sensor-frame `offset` per meter."""
    return distance * np.linalg.norm(offset, axis=-1)


def scan_ranges(distance: np.ndarray, offset: np.ndarray, range_max: float) -> np.ndarray:
    """Slant range per ray of horizontal `distance` along `offset`, inf beyond range_max."""
    slant = slant_range(distance, offset)
    return np.where(slant <= range_max, slant, np.inf)


def lidar_cloud(distance: np.ndarray, offset: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """Sensor-frame LIDAR_POINT per (ring, beam) of horizontal `distance` along `offset` whose slant range lies within [range_min, range_max]."""
    slant = slant_range(distance, offset)
    ring, beam = np.nonzero((slant >= range_min) & (slant <= range_max))
    xyz = distance[ring, beam, None] * offset[ring, beam]
    points = np.zeros(len(xyz), dtype=LIDAR_POINT)
    points['x'], points['y'], points['z'] = xyz.T
    points['ring'] = ring
    return points


def camera_grid(horizontal_fov: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Column azimuths and (rows, columns) rise per meter of a pinhole image decimated to MAX_CAMERA_COLUMNS x MAX_CAMERA_ROWS."""
    half = horizontal_fov / 2.0
    azimuth = fan(min(width, MAX_CAMERA_COLUMNS), -half, half) if width > 1 else np.zeros(1)
    vertical = math.atan(math.tan(half) * height / width)
    elevation = fan(min(height, MAX_CAMERA_ROWS), -vertical, vertical) if height > 1 else np.zeros(1)
    return azimuth, np.tan(elevation)[:, None] * np.cos(azimuth)[None, :]


def camera_reach(offset: np.ndarray, far: float) -> float:
    """Horizontal ray length that covers depth `far` on every ray of sensor-frame `offset` per meter."""
    return far / float(offset[..., 0].min())


def camera_cloud(distance: np.ndarray, offset: np.ndarray, near: float, far: float) -> np.ndarray:
    """CAMERA_POINT (x forward, y left, z up) per (row, column) of horizontal `distance` along `offset` whose depth along x lies within [near, far]."""
    with np.errstate(invalid='ignore'):
        depth = distance * offset[..., 0]
    row, column = np.nonzero((depth >= near) & (depth <= far))
    xyz = distance[row, column, None] * offset[row, column]
    points = np.zeros(len(xyz), dtype=CAMERA_POINT)
    points['x'], points['y'], points['z'] = xyz.T
    return points


def imu_acceleration(ax: float, ay: float, mount_yaw: float) -> tuple[float, float, float]:
    """Specific force in a sensor frame yawed by `mount_yaw` from the body, given the body-frame acceleration."""
    cos, sin = math.cos(mount_yaw), math.sin(mount_yaw)
    return cos * ax + sin * ay, -sin * ax + cos * ay, GRAVITY

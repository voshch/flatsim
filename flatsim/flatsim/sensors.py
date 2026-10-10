"""ROS-free sensor math: beam patterns, point clouds from first-hit distances, the IMU reading."""

from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.80665
MAX_CAMERA_COLUMNS = 160
MAX_CAMERA_ROWS = 24

LIDAR_POINT = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'], 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'], 'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
CAMERA_POINT = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4'], 'offsets': [0, 4, 8], 'itemsize': 12})


def fan(samples: int, min_angle: float, max_angle: float) -> np.ndarray:
    """Angles of `samples` beams spread from min_angle to max_angle inclusive."""
    return np.linspace(min_angle, max_angle, samples)


def cast_fan(azimuth: np.ndarray, inverted: bool) -> tuple[float, float]:
    """(angle_min, angle_increment) that sweeps sensor-frame `azimuth` about the base z axis, mirrored for a sensor mounted upside down."""
    step = float(azimuth[1] - azimuth[0]) if len(azimuth) > 1 else 0.0
    return (-float(azimuth[0]), -step) if inverted else (float(azimuth[0]), step)


def ring_slopes(samples: int, min_angle: float, max_angle: float) -> np.ndarray:
    """Rise per meter of each lidar ring as a (rings, 1) column, one level ring for a 2D lidar."""
    if samples <= 1:
        return np.zeros((1, 1))
    return np.tan(fan(samples, min_angle, max_angle))[:, None]


def lidar_cloud(distance: np.ndarray, azimuth: np.ndarray, slopes: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """Sensor-frame LIDAR_POINT per (ring, beam) of `distance` whose slant range lies within [range_min, range_max]."""
    slant = distance * np.sqrt(1.0 + slopes**2)
    ring, beam = np.nonzero((slant >= range_min) & (slant <= range_max))
    reach = distance[ring, beam]
    points = np.zeros(len(reach), dtype=LIDAR_POINT)
    points['x'] = reach * np.cos(azimuth[beam])
    points['y'] = reach * np.sin(azimuth[beam])
    points['z'] = reach * slopes[ring, 0]
    points['ring'] = ring
    return points


def camera_grid(horizontal_fov: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Column azimuths and (rows, columns) rise per meter of a pinhole image decimated to MAX_CAMERA_COLUMNS x MAX_CAMERA_ROWS."""
    half = horizontal_fov / 2.0
    azimuth = fan(min(width, MAX_CAMERA_COLUMNS), -half, half) if width > 1 else np.zeros(1)
    vertical = math.atan(math.tan(half) * height / width)
    elevation = fan(min(height, MAX_CAMERA_ROWS), -vertical, vertical) if height > 1 else np.zeros(1)
    return azimuth, np.tan(elevation)[:, None] * np.cos(azimuth)[None, :]


def camera_reach(horizontal_fov: float, far: float) -> float:
    """Horizontal ray length that covers depth `far` at the edge of the field of view."""
    return far / math.cos(horizontal_fov / 2.0)


def camera_cloud(distance: np.ndarray, azimuth: np.ndarray, slopes: np.ndarray, near: float, far: float) -> np.ndarray:
    """CAMERA_POINT (x forward, y left, z up) per (row, column) of `distance` whose depth along x lies within [near, far]."""
    depth = distance * np.cos(azimuth)
    row, column = np.nonzero((depth >= near) & (depth <= far))
    reach = distance[row, column]
    points = np.zeros(len(reach), dtype=CAMERA_POINT)
    points['x'] = depth[row, column]
    points['y'] = reach * np.sin(azimuth[column])
    points['z'] = reach * slopes[row, column]
    return points


def imu_acceleration(ax: float, ay: float, mount_yaw: float) -> tuple[float, float, float]:
    """Specific force in a sensor frame yawed by `mount_yaw` from the body, given the body-frame acceleration."""
    cos, sin = math.cos(mount_yaw), math.sin(mount_yaw)
    return cos * ax + sin * ay, -sin * ax + cos * ay, GRAVITY

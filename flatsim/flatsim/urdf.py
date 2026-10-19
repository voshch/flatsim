"""Link offsets, joint names and gazebo sensor blocks read straight from a URDF string."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import attrs
import numpy as np

LIDAR_TYPES = ('gpu_lidar', 'lidar', 'gpu_ray', 'ray')
CAMERA_TYPES = ('camera', 'depth_camera', 'depth', 'rgbd_camera')

_MOVABLE = ('revolute', 'continuous', 'prismatic')


@attrs.frozen
class LidarBlock:
    """Ray pattern of a lidar sensor, one vertical sample being a 2D lidar."""

    samples: int
    min_angle: float
    max_angle: float
    vertical_samples: int
    vertical_min_angle: float
    vertical_max_angle: float
    range_min: float
    range_max: float


@attrs.frozen
class CameraBlock:
    """Pinhole geometry of a camera sensor."""

    horizontal_fov: float
    width: int
    height: int
    near: float
    far: float


@attrs.frozen
class SensorBlock:
    """One <sensor> element under a <gazebo reference> block, update_rate None when unset."""

    name: str
    type: str
    link: str
    update_rate: float | None
    lidar: LidarBlock | None = None
    camera: CameraBlock | None = None


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find('origin')
    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.0, 0.0, 0.0]
    roll, pitch, yaw = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.0, 0.0, 0.0]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transform[:3, 3] = xyz
    return transform


def _transform(urdf_xml: str, parent: str, child: str) -> np.ndarray:
    root = ET.fromstring(urdf_xml)
    links = {link.get('name') for link in root.findall('link')}
    for name in (parent, child):
        if name not in links:
            raise ValueError(f'link {name!r} is not in the URDF, known links: {sorted(str(link) for link in links)}')
    above: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall('joint'):
        child_tag, parent_tag = joint.find('child'), joint.find('parent')
        child_link = child_tag.get('link') if child_tag is not None else None
        parent_link = parent_tag.get('link') if parent_tag is not None else None
        if child_link is None or parent_link is None:
            raise ValueError(f'joint {joint.get("name")!r} needs a <parent link> and a <child link>')
        above[child_link] = (parent_link, _origin(joint))

    def from_root(link: str) -> np.ndarray:
        transform = np.eye(4)
        while link in above:
            link, origin = above[link]
            transform = origin @ transform
        return transform

    return np.linalg.inv(from_root(parent)) @ from_root(child)


def frame_offset(urdf_xml: str, parent: str, child: str) -> tuple[float, float, float, float]:
    """Pose (x, y, z, yaw) of link `child` in link `parent` with every joint at zero."""
    offset = _transform(urdf_xml, parent, child)
    return float(offset[0, 3]), float(offset[1, 3]), float(offset[2, 3]), math.atan2(offset[1, 0], offset[0, 0])


def frame_inverted(urdf_xml: str, parent: str, child: str) -> bool:
    """Whether the z axis of link `child` points down in link `parent`."""
    return bool(_transform(urdf_xml, parent, child)[2, 2] < 0.0)


def movable_joints(urdf_xml: str) -> list[str]:
    """Names of the joints robot_state_publisher needs a position for."""
    root = ET.fromstring(urdf_xml)
    return [joint.get('name', '') for joint in root.findall('joint') if joint.get('type') in _MOVABLE and joint.find('mimic') is None]


def _number(element: ET.Element, path: str, default: float) -> float:
    text = element.findtext(path)
    return float(text) if text is not None and text.strip() else default


def _child(sensor: ET.Element, *tags: str) -> ET.Element:
    """First present child among `tags`, the sensor itself without one."""
    return next((child for tag in tags if (child := sensor.find(tag)) is not None), sensor)


def _lidar(sensor: ET.Element) -> LidarBlock:
    block = _child(sensor, 'lidar', 'ray')
    return LidarBlock(
        samples=int(_number(block, 'scan/horizontal/samples', 1)),
        min_angle=_number(block, 'scan/horizontal/min_angle', -math.pi),
        max_angle=_number(block, 'scan/horizontal/max_angle', math.pi),
        vertical_samples=int(_number(block, 'scan/vertical/samples', 1)),
        vertical_min_angle=_number(block, 'scan/vertical/min_angle', 0.0),
        vertical_max_angle=_number(block, 'scan/vertical/max_angle', 0.0),
        range_min=_number(block, 'range/min', 0.0),
        range_max=_number(block, 'range/max', 100.0),
    )


def _camera(sensor: ET.Element) -> CameraBlock:
    block = _child(sensor, 'camera')
    return CameraBlock(
        horizontal_fov=_number(block, 'horizontal_fov', 1.047),
        width=int(_number(block, 'image/width', 320)),
        height=int(_number(block, 'image/height', 240)),
        near=_number(block, 'clip/near', 0.1),
        far=_number(block, 'clip/far', 100.0),
    )


def sensor_block(urdf_xml: str, name: str, link: str | None = None) -> SensorBlock | None:
    """The <sensor name=`name`> element of a <gazebo reference> block, the one on `link` when several share the name."""
    root = ET.fromstring(urdf_xml)
    found = [(ref, sensor) for gazebo in root.iter('gazebo') if (ref := gazebo.get('reference')) is not None for sensor in gazebo.iter('sensor') if sensor.get('name') == name]
    if not found:
        return None
    reference, sensor = next((entry for entry in found if entry[0] == link), found[0])
    kind = sensor.get('type', '')
    rate = _number(sensor, 'update_rate', 0.0) or _number(sensor, './/update_rate', 0.0)
    return SensorBlock(
        name=name,
        type=kind,
        link=reference,
        update_rate=rate if rate > 0.0 else None,
        lidar=_lidar(sensor) if kind in LIDAR_TYPES else None,
        camera=_camera(sensor) if kind in CAMERA_TYPES else None,
    )

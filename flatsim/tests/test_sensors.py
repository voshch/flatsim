"""Flat sensor math on real FlatWorld casts: lidar rings, camera grids, cloud layouts, the IMU reading."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flatsim.drive import Axis, DriveModel
from flatsim.sensors import CAMERA_POINT, GRAVITY, LIDAR_POINT, MAX_CAMERA_COLUMNS, MAX_CAMERA_ROWS, RayPlan, camera_cloud, camera_grid, camera_reach, fan, imu_acceleration, lidar_cloud, ray_plan, ring_slopes, scan_ranges
from flatsim.urdf import frame_rotation
from flatsim.world import Dynamics, FlatWorld, ScanSpec

DT = 1.0 / 30.0
FREE = DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(0.0, 0.0), angular=Axis(-2.0, 2.0))
VLP16 = (16, -0.261799, 0.261799)
PLANAR = (1, 0.0, 0.0)
D435 = (1.5184351666666667, 640, 480)
UPRIGHT = '0 0 0'
FLIPPED = '3.141592653589793 0 0'


def _mounted(rpy: str) -> np.ndarray:
    """Rotation of a sensor link fixed to the base at `rpy`."""
    urdf = f'<robot name="mount"><link name="base"/><link name="sensor"/><joint name="mount" type="fixed"><parent link="base"/><child link="sensor"/><origin rpy="{rpy}"/></joint></robot>'
    return frame_rotation(urdf, 'base', 'sensor')


def _plan(rpy: str, azimuth: np.ndarray, slopes: np.ndarray) -> RayPlan:
    """Plan of the given rays plus a last row in the sensor's own xy plane."""
    return ray_plan(_mounted(rpy), azimuth, np.vstack([np.broadcast_to(slopes, (len(slopes), len(azimuth))), np.zeros((1, len(azimuth)))]))


def _cast(world: FlatWorld, plan: RayPlan, mount: tuple[float, float], height: float, reach: float) -> np.ndarray:
    """Horizontal distance per ray of `plan` from one cast of robot `robot`."""
    spec = ScanSpec(mount=(*mount, 0.0), angle_min=plan.angle_min, angle_increment=plan.angle_increment, beams=plan.beams, range_min=0.0, range_max=reach, height=height)
    return world.cast('robot', spec).take(plan.beam).first(plan.slope)


def _lidar(
    world: FlatWorld,
    height: float,
    horizontal: tuple[int, float, float],
    vertical: tuple[int, float, float],
    limits: tuple[float, float],
    inverted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan ranges and cloud of one lidar on robot `robot`, both from a single cast."""
    azimuth = fan(*horizontal)
    slopes = ring_slopes(*vertical)
    plan = _plan(FLIPPED if inverted else UPRIGHT, azimuth, slopes)
    distance = _cast(world, plan, (0.0, 0.0), height, limits[1])
    return scan_ranges(distance[-1], plan.offset[-1], limits[1]), lidar_cloud(distance[:-1], plan.offset[:-1], *limits), azimuth, slopes


def _camera(world: FlatWorld, rpy: str, height: float, near: float, far: float) -> np.ndarray:
    """Cloud of a D435 at (0.2, 0) on robot `robot`."""
    plan = _plan(rpy, *camera_grid(*D435))
    distance = _cast(world, plan, (0.2, 0.0), height, camera_reach(plan.offset[:-1], far))
    return camera_cloud(distance[:-1], plan.offset[:-1], near, far)


def _boxes() -> FlatWorld:
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('crate', (0.2, 4.0), (2.1, 0.0, 0.0), (0.0, 0.5))
    world.add_box('wall', (0.2, 12.0), (4.1, 0.0, 0.0), (0.0, 2.5))
    return world


def test_point_layouts_are_the_published_ones():
    assert LIDAR_POINT.itemsize == 32
    assert {name: (kind.str, offset) for name, (kind, offset) in LIDAR_POINT.fields.items()} == {'x': ('<f4', 0), 'y': ('<f4', 4), 'z': ('<f4', 8), 'intensity': ('<f4', 16), 'ring': ('<u2', 24)}
    assert CAMERA_POINT.itemsize == 12
    assert {name: (kind.str, offset) for name, (kind, offset) in CAMERA_POINT.fields.items()} == {'x': ('<f4', 0), 'y': ('<f4', 4), 'z': ('<f4', 8)}


def test_planar_lidar_cloud_is_its_scan_as_points():
    world = _boxes()
    world.add_disc('ped', 0.3, (0.0, -0.35), (0.0, 1.8))
    world.add_disc('far', 0.3, (-9.0, 0.0), (0.0, 1.8))
    ranges, cloud, azimuth, _ = _lidar(world, 0.3, (360, -math.pi, math.pi), PLANAR, (0.1, 6.0))
    assert ranges.min() == pytest.approx(0.05, abs=1e-4)
    assert np.isinf(ranges[0])
    kept = (ranges >= 0.1) & (ranges <= 6.0)
    assert (ranges < 0.1).any() and kept.any() and np.isinf(ranges).any()
    assert len(cloud) == kept.sum()
    assert np.hypot(cloud['x'], cloud['y']) == pytest.approx(ranges[kept], abs=1e-5)
    assert np.arctan2(cloud['y'], cloud['x']) == pytest.approx(azimuth[kept], abs=1e-5)
    assert not cloud['z'].any() and not cloud['ring'].any() and not cloud['intensity'].any()
    assert len(cloud.tobytes()) == 32 * len(cloud)


def test_sixteen_rings_land_on_each_box_within_its_height():
    world = _boxes()
    height = 0.3
    ranges, cloud, azimuth, slopes = _lidar(world, height, (9, -0.2, 0.2), VLP16, (0.08, 12.0))
    assert slopes.shape == (16, 1)
    assert np.arctan(slopes[:, 0]) == pytest.approx(np.linspace(-0.261799, 0.261799, 16))
    assert ranges == pytest.approx(2.0 / np.cos(azimuth), abs=1e-4)

    expected = []
    for ring, slope in enumerate(slopes[:, 0]):
        for angle in azimuth:
            crate, wall = 2.0 / math.cos(angle), 4.0 / math.cos(angle)
            floor = -height / slope if slope < 0.0 else math.inf
            reach = min([distance for distance, top in ((crate, 0.5), (wall, 2.5)) if 0.0 <= height + distance * slope <= top] + [floor])
            if math.isfinite(reach):
                expected.append((reach * math.cos(angle), reach * math.sin(angle), reach * slope, ring))
    x, y, z, rings = (np.array(column) for column in zip(*expected, strict=True))
    assert len(cloud) == len(expected)
    assert cloud['x'] == pytest.approx(x, abs=1e-4)
    assert cloud['y'] == pytest.approx(y, abs=1e-4)
    assert cloud['z'] == pytest.approx(z, abs=1e-4)
    assert (cloud['ring'] == rings).all()

    on_floor = np.isclose(cloud['z'], -height, atol=1e-4)
    on_crate = ~on_floor & (cloud['x'] < 3.0)
    on_wall = ~on_floor & ~on_crate
    assert set(cloud['ring'][on_floor].tolist()) == set(range(4))
    assert np.hypot(cloud['x'][on_floor], cloud['y'][on_floor]).max() < 2.0
    assert ((cloud['z'][on_crate] + height >= 0.0) & (cloud['z'][on_crate] + height <= 0.5)).all()
    assert ((cloud['z'][on_wall] + height >= 0.0) & (cloud['z'][on_wall] + height <= 2.5)).all()
    assert set(cloud['ring'][on_crate].tolist()) == set(range(4, 11))
    assert set(cloud['ring'][on_wall].tolist()) == set(range(11, 16))
    slant = np.sqrt(cloud['x'] ** 2 + cloud['y'] ** 2 + cloud['z'] ** 2)
    assert slant[~on_floor].min() >= 2.0 and slant.max() <= 4.0 / math.cos(0.2) / math.cos(0.261799) + 1e-4


def test_slant_range_bounds_which_rings_report_a_tall_wall():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('wall', (0.2, 12.0), (4.1, 0.0, 0.0), (-5.0, 5.0))
    ahead, rings = (1, 0.0, 0.0), (5, -0.2, 0.2)
    assert _lidar(world, 0.0, ahead, rings, (0.08, 12.0))[1]['ring'].tolist() == [0, 1, 2, 3, 4]
    assert _lidar(world, 0.0, ahead, rings, (0.08, 4.05))[1]['ring'].tolist() == [1, 2, 3]
    cloud = _lidar(world, 0.0, ahead, rings, (4.01, 4.05))[1]
    assert cloud['ring'].tolist() == [1, 3]
    assert cloud['x'] == pytest.approx([4.0, 4.0], abs=1e-4)
    assert cloud['z'] == pytest.approx([-4.0 * math.tan(0.1), 4.0 * math.tan(0.1)], abs=1e-4)
    assert len(_lidar(world, 0.0, ahead, rings, (0.08, 3.9))[1]) == 0


def test_upside_down_lidar_mirrors_left_and_up_in_its_own_frame():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_disc('left', 0.3, (0.0, 2.0), (0.4, 0.6))
    world.add_disc('high', 0.3, (2.0, 0.0), (0.75, 1.5))
    quarter_turns, rings = (5, -math.pi, math.pi), (3, -0.2, 0.2)
    upright = _lidar(world, 0.5, quarter_turns, rings, (0.08, 12.0))
    flipped = _lidar(world, 0.5, quarter_turns, rings, (0.08, 12.0), inverted=True)
    rise = 1.7 * math.tan(0.2)

    assert np.isfinite(upright[0]).tolist() == [False, False, False, True, False]
    assert np.isfinite(flipped[0]).tolist() == [False, True, False, False, False]
    for ranges, cloud, side, up in ((*upright[:2], 1.0, 1.0), (*flipped[:2], -1.0, -1.0)):
        assert ranges[np.isfinite(ranges)] == pytest.approx([1.7], abs=1e-4)
        floor = np.isclose(cloud['z'] * up, -0.5, atol=1e-4)
        assert set(cloud['ring'][floor].tolist()) == {0 if up > 0 else 2}
        assert np.hypot(cloud['x'][floor], cloud['y'][floor]) == pytest.approx(0.5 / math.tan(0.2), abs=1e-4)
        cloud = cloud[~floor]
        points = sorted(zip(cloud['ring'].tolist(), cloud['x'].tolist(), cloud['y'].tolist(), cloud['z'].tolist(), strict=True))
        high_ring = 2 if up > 0 else 0
        assert [ring for ring, *_ in points] == sorted([1, high_ring])
        level = next(point for point in points if point[0] == 1)
        raised = next(point for point in points if point[0] == high_ring)
        assert level[1:] == pytest.approx((0.0, side * 1.7, 0.0), abs=1e-4)
        assert raised[1:] == pytest.approx((1.7, 0.0, up * rise), abs=1e-4)


def test_camera_grid_is_decimated_and_keeps_the_image_aspect():
    azimuth, slopes = camera_grid(*D435)
    assert len(azimuth) == MAX_CAMERA_COLUMNS == 160
    assert slopes.shape == (MAX_CAMERA_ROWS, MAX_CAMERA_COLUMNS) == (24, 160)
    assert (azimuth[0], azimuth[-1]) == pytest.approx((-D435[0] / 2.0, D435[0] / 2.0))
    top = math.tan(D435[0] / 2.0) * 480 / 640
    assert (slopes[0, 80] + slopes[0, 79]) / 2.0 == pytest.approx(-top, rel=1e-3)
    assert slopes[-1, 0] == pytest.approx(top * math.cos(D435[0] / 2.0))
    assert np.allclose(slopes, -slopes[::-1])

    azimuth, slopes = camera_grid(1.0, 80, 4)
    assert (len(azimuth), slopes.shape) == (80, (4, 80))
    azimuth, slopes = camera_grid(1.0, 1, 1)
    assert azimuth.tolist() == [0.0] and slopes.tolist() == [[0.0]]


def test_camera_cloud_lands_on_the_wall_within_its_height_or_the_floor_and_the_clip():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('wall', (0.2, 20.0), (3.1, 0.0, 0.0), (0.0, 2.5))
    cloud = _camera(world, UPRIGHT, 1.0, 0.105, 8.0)
    floor = np.isclose(cloud['z'], -1.0, atol=1e-4)
    wall = cloud[~floor]
    elevation = np.arctan(camera_grid(*D435)[1][:, 80] / math.cos(camera_grid(*D435)[0][80]))
    rise = 2.8 * np.tan(elevation)
    rows = ((rise >= -1.0) & (rise <= 1.5)).sum()
    assert 0 < rows < 24
    assert len(wall) == rows * 160
    assert floor.sum() == (rise < -1.0).sum() * 160 > 0
    assert (cloud['x'][floor] < 2.8).all()
    assert wall['x'] == pytest.approx(np.full(len(wall), 2.8), abs=1e-4)
    assert np.abs(wall['y']).max() == pytest.approx(2.8 * math.tan(D435[0] / 2.0), abs=1e-4)
    assert wall['z'].min() >= -1.0 and wall['z'].max() <= 1.5
    assert wall['z'].min() < -0.8 and wall['z'].max() > 1.3
    assert len(cloud.tobytes()) == 12 * len(cloud)

    near = _camera(world, UPRIGHT, 1.0, 0.105, 2.7)
    assert len(near) > 0 and np.isclose(near['z'], -1.0, atol=1e-4).all()
    assert len(_camera(world, UPRIGHT, 1.0, 2.9, 8.0)) == 0
    turned = _camera(world, '0 0 1.5707963267948966', 1.0, 0.105, 8.0)
    turned = turned[~np.isclose(turned['z'], -1.0, atol=1e-4)]
    assert len(turned) > 0
    assert turned['y'] == pytest.approx(np.full(len(turned), -2.8), abs=1e-4)


@pytest.mark.parametrize('rpy', ['0 -0.2576 0', FLIPPED, '3.141592653589793 0.3 0', '0.2 0.5 0.4', '0 1.2 0'])
def test_tilted_camera_floor_lies_flat_in_the_base_frame_and_within_the_field_of_view(rpy: str):
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    cloud = _camera(world, rpy, 0.5, 0.105, 8.0)
    assert len(cloud) > 0
    sensor = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1).astype(np.float64)
    base = sensor @ _mounted(rpy).T + (0.2, 0.0, 0.5)
    assert base[:, 2] == pytest.approx(np.zeros(len(base)), abs=1e-4)
    assert (sensor[:, 0] > 0.0).all()
    assert np.abs(sensor[:, 1] / sensor[:, 0]).max() <= math.tan(D435[0] / 2.0) + 0.02
    assert np.abs(sensor[:, 2] / sensor[:, 0]).max() <= math.tan(D435[0] / 2.0) * 480 / 640 + 0.02


def test_camera_pitched_down_sees_the_floor_on_every_ray():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    assert len(_camera(world, '0 1.2 0', 0.5, 0.105, 8.0)) == MAX_CAMERA_ROWS * MAX_CAMERA_COLUMNS


def test_pitched_planar_lidar_returns_hold_their_height_in_the_base_frame():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('wall', (0.2, 20.0), (3.1, 0.0, 0.0), (0.0, 2.0))
    rpy, height = '0 0.1 0', 0.5
    azimuth = fan(181, -math.pi / 2, math.pi / 2)
    plan = _plan(rpy, azimuth, ring_slopes(*PLANAR))
    distance = _cast(world, plan, (0.0, 0.0), height, 12.0)
    ranges = scan_ranges(distance[-1], plan.offset[-1], 12.0)
    assert ranges[90] == pytest.approx(3.0 / math.cos(0.1), abs=1e-4)
    assert np.isinf(ranges[[0, -1]]).all()
    cloud = lidar_cloud(distance[:-1], plan.offset[:-1], 0.08, 12.0)
    base = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1).astype(np.float64) @ _mounted(rpy).T + (0.0, 0.0, height)
    on_floor = np.isclose(base[:, 2], 0.0, atol=1e-4)
    on_wall = np.isclose(base[:, 0], 3.0, atol=1e-4)
    assert on_wall.any() and (on_floor | on_wall).all()
    assert ((base[on_wall, 2] >= 0.0) & (base[on_wall, 2] <= 2.0)).all()
    ahead = base[np.abs(base[:, 1]) < 1e-6]
    assert ahead == pytest.approx(np.array([[3.0, 0.0, height - 3.0 * math.tan(0.1)]]), abs=1e-4)


def test_imu_reads_a_speed_up_and_a_steady_turn_in_its_own_frame():
    limits = DriveModel(linear=Axis(-0.5, 0.5, 2.5), lateral=Axis(0.0, 0.0), angular=Axis(-1.5, 1.5, 3.2))
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, limits)
    assert imu_acceleration(0.0, 0.0, 0.3) == (0.0, 0.0, GRAVITY)
    world.command('robot', (10.0, 0.0, 0.0))
    world.advance(2 * DT)
    state = world.robot_state('robot')
    assert imu_acceleration(state.ax, state.ay, 0.0) == pytest.approx((2.5, 0.0, 9.80665))
    assert imu_acceleration(state.ax, state.ay, math.pi / 2) == pytest.approx((0.0, -2.5, 9.80665))

    for i in range(3, 40):
        world.command('robot', (10.0, 0.0, 10.0))
        world.advance(i * DT)
    state = world.robot_state('robot')
    assert (state.vx, state.wz) == (0.5, 1.5)
    assert imu_acceleration(state.ax, state.ay, 0.0) == pytest.approx((0.0, 0.75, 9.80665))
    assert imu_acceleration(state.ax, state.ay, math.pi) == pytest.approx((0.0, -0.75, 9.80665))
    assert imu_acceleration(state.ax, state.ay, math.pi / 4) == pytest.approx((0.75 * math.sin(math.pi / 4), 0.75 * math.cos(math.pi / 4), 9.80665))

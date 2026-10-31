"""Warp against Box2D on the same scenes, within fixed tolerances."""

from __future__ import annotations

import math
import typing

import attrs
import numpy as np
import pytest

from flatsim.drive import Axis, DriveModel
from flatsim.physics import Physics, make_world, resolve
from flatsim.world import Dynamics, ScanSpec

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from flatsim.physics import World

    Scene = Callable[[World], np.ndarray]

DT = 1.0 / 30.0
DRIVE = DriveModel(linear=Axis(-1.0, 1.0, 2.0), lateral=Axis(-0.5, 0.5, 2.0), angular=Axis(-2.0, 2.0, 4.0), lag=0.1, traction=3.0)
STIFF = DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(0.0, 0.0), angular=Axis(-2.0, 2.0))
SQUARE = [(0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2)]
OFFSET = [(0.4, 0.2), (0.4, -0.2), (-0.2, -0.2), (-0.2, 0.2)]
FAN = ScanSpec(mount=(0.1, 0.0, 0.0), angle_min=-math.pi, angle_increment=2 * math.pi / 360, beams=360, range_min=0.05, range_max=10.0, height=0.3)


def _both(scene: Scene, dynamics: Dynamics = Dynamics.RIGID) -> tuple[np.ndarray, np.ndarray]:
    return scene(make_world(DT, dynamics, Physics.BOX2D)), scene(make_world(DT, dynamics, Physics.WARP))


def _pose(world: World, name: str) -> np.ndarray:
    state = world.robot_state(name)
    return np.array([state.x, state.y, state.yaw])


def _wall(world: World, name: str, start: tuple[float, float], end: tuple[float, float], z: tuple[float, float] = (0.0, 2.5)) -> None:
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / length * 0.05, (x1 - x0) / length * 0.05
    world.add_outline(name, [[(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]], z)


def _drive(world: World, plan: list[tuple[float, dict[str, tuple[float, float, float]]]]) -> np.ndarray:
    """Hold each command set for its seconds, recording every robot's pose and twist each step."""
    out = []
    for seconds, commands in plan:
        for _ in range(round(seconds / DT)):
            for name, twist in commands.items():
                world.command(name, twist)
            world.advance(world.time + DT)
            for name in commands:
                state = world.robot_state(name)
                out.append([state.x, state.y, state.yaw, state.vx, state.vy, state.wz, state.ax, state.ay])
    return np.array(out)


@pytest.mark.parametrize('dynamics', [Dynamics.RIGID, Dynamics.KINEMATIC])
@pytest.mark.parametrize('footprint', [None, OFFSET])
def test_free_drive_follows_the_drive_model_alike(dynamics: Dynamics, footprint: list[tuple[float, float]] | None):
    def scene(world: World) -> np.ndarray:
        world.add_robot('robot', (1.0, -2.0, 0.3), 0.0, DRIVE, footprint=footprint, radius=0.25)
        return _drive(world, [(1.5, {'robot': (1.0, 0.3, 0.8)}), (1.0, {'robot': (-0.5, -0.4, -2.0)}), (1.0, {'robot': (0.0, 0.0, 0.0)})])

    box2d, warp = _both(scene, dynamics)
    assert np.abs(warp[:, :3] - box2d[:, :3]).max() < 1e-3
    assert np.abs(warp[:, 3:6] - box2d[:, 3:6]).max() < 1e-3
    assert np.abs(warp[:, 6:] - box2d[:, 6:]).max() < 0.05


@pytest.mark.parametrize('footprint', [None, SQUARE, OFFSET])
def test_head_on_wall_stops_the_robot_alike(footprint: list[tuple[float, float]] | None):
    def scene(world: World) -> np.ndarray:
        _wall(world, 'wall', (2.5, -3.0), (2.5, 3.0))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=footprint, radius=0.2)
        return _drive(world, [(4.0, {'robot': (1.0, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.02
    assert abs(warp[-1, 0] - box2d[-1, 0]) < 0.01
    assert abs(warp[-1, 3]) < 0.05
    assert abs(warp[-1, 2]) < 0.01


@pytest.mark.parametrize('footprint', [None, SQUARE])
def test_oblique_wall_slides_the_robot_alike(footprint: list[tuple[float, float]] | None):
    def scene(world: World) -> np.ndarray:
        _wall(world, 'wall', (2.0, -5.0), (2.0, 5.0))
        world.add_robot('robot', (0.5, 0.0, math.radians(30.0)), 0.0, STIFF, footprint=footprint, radius=0.2)
        return _drive(world, [(4.0, {'robot': (1.0, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.03
    assert np.abs(warp[:, 2] - box2d[:, 2]).max() < 0.05


def test_corner_hit_turns_the_robot_alike():
    def scene(world: World) -> np.ndarray:
        _wall(world, 'stub', (2.0, 0.15), (2.0, 3.0))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=OFFSET)
        return _drive(world, [(3.0, {'robot': (0.6, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    turned = box2d[-1, 2]
    assert abs(turned) > 0.05
    assert abs(warp[-1, 2] - turned) < 0.1
    assert np.abs(warp[-1, :2] - box2d[-1, :2]).max() < 0.05


def test_moving_box_pushes_a_resting_robot_alike():
    def scene(world: World) -> np.ndarray:
        world.add_box('door', (0.2, 2.0), (-1.0, 0.0, 0.0), (0.0, 2.0))
        world.move('door', (-1.0, 0.0, 0.0), velocity=(0.5, 0.0))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        return _drive(world, [(3.0, {'robot': (0.0, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    assert box2d[-1, 0] > 0.3
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.02


def test_robot_shoving_a_peer_moves_both_alike():
    def scene(world: World) -> np.ndarray:
        world.add_robot('pusher', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        world.add_robot('parked', (1.0, 0.05, 0.0), 0.0, STIFF, radius=0.25)
        return _drive(world, [(3.0, {'pusher': (0.8, 0.0, 0.0), 'parked': (0.0, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    assert np.abs(warp[-2:, :2] - box2d[-2:, :2]).max() < 0.03
    assert np.abs(warp[-2:, 2] - box2d[-2:, 2]).max() < 0.1


@pytest.mark.parametrize('dynamics', [Dynamics.RIGID, Dynamics.KINEMATIC])
def test_walking_disc_never_moves_the_robot_in_either(dynamics: Dynamics):
    def scene(world: World) -> np.ndarray:
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, radius=0.3)
        world.add_disc('ped', 0.3, (-2.0, 0.0), (0.0, 1.8))
        world.move('ped', (-2.0, 0.0, 0.0), velocity=(1.0, 0.0))
        return _drive(world, [(4.0, {'robot': (0.0, 0.0, 0.0)})])

    box2d, warp = _both(scene, dynamics)
    assert np.abs(box2d[:, :6]).max() == 0.0
    assert np.abs(warp[:, :6]).max() == 0.0


def test_kinematic_robot_passes_walls_alike():
    def scene(world: World) -> np.ndarray:
        _wall(world, 'wall', (2.0, -3.0), (2.0, 3.0))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        return _drive(world, [(4.0, {'robot': (1.0, 0.0, 0.0)})])

    box2d, warp = _both(scene, Dynamics.KINEMATIC)
    assert np.abs(warp[:, :3] - box2d[:, :3]).max() < 1e-3


def test_levels_and_floor_crossings_alike():
    def scene(world: World) -> np.ndarray:
        world.set_levels([0.0, 3.0])
        _wall(world, 'ground', (1.5, -3.0), (1.5, 3.0), (0.0, 2.5))
        _wall(world, 'upper', (3.0, -3.0), (3.0, 3.0), (3.0, 5.5))
        world.add_robot('robot', (0.0, 0.0, 0.0), 3.0, STIFF, footprint=SQUARE)
        first = _drive(world, [(4.0, {'robot': (1.0, 0.0, 0.0)})])
        world.set_robot_pose('robot', (0.0, 0.5, 0.0), 0.0)
        return np.vstack([first, _drive(world, [(3.0, {'robot': (1.0, 0.0, 0.0)})])])

    box2d, warp = _both(scene)
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.02


def test_scans_and_contacts_from_one_pose_alike():
    poses = {'a': (0.0, 0.0, 0.2), 'b': (1.4, 0.3, 1.0), 'c': (-1.5, 1.0, -2.0), 'd': (0.3, -0.42, 0.0)}

    def scene(world: World, poses: dict[str, tuple[float, float, float]]) -> tuple[list[np.ndarray], list[list[str]]]:
        world.set_levels([0.0, 3.0])
        _wall(world, 'north', (-4.0, 3.0), (4.0, 3.0))
        _wall(world, 'east', (4.0, -3.0), (4.0, 3.0), (0.0, 6.0))
        _wall(world, 'upstairs', (0.0, 2.0), (3.0, 2.0), (3.0, 5.0))
        world.add_box('crate', (0.5, 0.8), (2.5, -1.0, 0.4), (0.0, 0.6))
        world.add_box('sign', (0.3, 1.0), (-2.5, -1.5, 0.0), (1.0, 2.0))
        world.add_disc('ped', 0.3, (-1.0, -1.5), (0.0, 1.8))
        world.add_disc('slab', 0.4, (1.0, 1.5), (-0.04, 0.01))
        for i, (name, pose) in enumerate(poses.items()):
            world.add_robot(name, pose, 0.0, STIFF, footprint=SQUARE if i % 2 else None, radius=0.25, height=0.8)
        scans, touching = [], []
        for name in poses:
            hits = world.cast(name, FAN)
            scans.extend(hits.first(slope) for slope in (0.0, 0.2, -0.05, -0.3))
            scans.append(world.scan(name, attrs.evolve(FAN, height=1.5)))
            touching.append([other for other, _, _ in world.contacts(name)])
        return scans, touching

    placed = make_world(DT, Dynamics.RIGID, Physics.BOX2D)
    box2d = scene(placed, poses)
    actual = {name: tuple(_pose(placed, name).tolist()) for name in poses}
    warp = scene(make_world(DT, Dynamics.RIGID, Physics.WARP), actual)
    for b, w in zip(box2d[0], warp[0], strict=True):
        finite = ~np.isinf(b)
        assert np.array_equal(finite, ~np.isinf(w))
        assert np.abs(b[finite] - w[finite]).max(initial=0.0) < 1e-4
    assert box2d[1] == warp[1]


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_first_many_gives_the_first_hit_of_each_ray_of_a_tilted_fan(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    world.set_levels([0.0, 3.0])
    _wall(world, 'north', (-4.0, 3.0), (4.0, 3.0))
    world.add_box('crate', (0.5, 0.8), (2.5, 0.0, 0.0), (0.0, 0.6))
    world.add_disc('ped', 0.3, (-1.5, 0.0), (0.0, 1.8))
    world.add_robot('a', (0.0, 0.0, 0.0), 0.0, STIFF, radius=0.25)
    world.add_robot('b', (0.0, 2.0, 0.5), 0.0, STIFF, footprint=SQUARE)
    beam = np.array([[0, 90, 180, 270], [0, 90, 180, 270], [359, 90, 180, 270]])
    slope = np.array([[0.0], [0.5], [-0.2]])
    spec = attrs.evolve(FAN, mount=(0.0, 0.0, 0.0), angle_min=0.0)
    other = attrs.evolve(FAN, beams=4, angle_increment=math.pi / 2)
    plain = np.arange(4)
    got = world.first_many([('a', spec, beam, slope), ('b', other, plain, np.array(0.0))])
    assert got[0].shape == beam.shape
    assert got[1].shape == plain.shape
    for name, fan, rays, rise, distance in (('a', spec, beam, slope, got[0]), ('b', other, plain, np.array(0.0), got[1])):
        want = world.cast(name, fan).take(rays).first(rise)
        finite = ~np.isinf(want)
        assert np.array_equal(finite, ~np.isinf(distance))
        assert np.abs(want[finite] - distance[finite]).max(initial=0.0) < 1e-4


def test_crowd_in_a_corridor_stays_out_of_the_walls_and_advances_alike():
    def scene(world: World) -> np.ndarray:
        _wall(world, 'south', (-8.0, -1.2), (8.0, -1.2))
        _wall(world, 'north', (-8.0, 1.2), (8.0, 1.2))
        commands = {}
        for k in range(6):
            lane = -0.6 + 0.3 * (k % 5)
            world.add_robot(f'east{k}', (-2.0 - 0.7 * k, lane, 0.0), 0.0, STIFF, footprint=SQUARE if k % 2 else None, radius=0.2)
            world.add_robot(f'west{k}', (2.0 + 0.7 * k, -lane, math.pi), 0.0, STIFF, footprint=None if k % 2 else SQUARE, radius=0.2)
            commands[f'east{k}'] = commands[f'west{k}'] = (0.8, 0.0, 0.0)
        return _drive(world, [(8.0, commands)])

    box2d, warp = _both(scene)
    for run in (box2d, warp):
        assert np.abs(run[:, 1]).max() < 0.97

    def progress(run: np.ndarray) -> float:
        return float(np.abs(run[-12:, 0] - run[:12, 0]).mean())

    assert abs(progress(warp) - progress(box2d)) < 0.1 * progress(box2d)


def test_auto_takes_warp_exactly_when_a_cuda_device_is_visible():
    import warp as wp

    assert resolve(Physics.AUTO) is (Physics.WARP if wp.is_cuda_available() else Physics.BOX2D)
    assert resolve(Physics.BOX2D) is Physics.BOX2D
    assert resolve(Physics.WARP) is Physics.WARP


def test_warp_repeats_itself_byte_for_byte():
    def run() -> tuple[np.ndarray, np.ndarray]:
        world = make_world(DT, Dynamics.RIGID, Physics.WARP)
        rng = np.random.default_rng(3)
        for k in range(30):
            x, y = rng.uniform(-8.0, 8.0, 2)
            sx, sy = rng.uniform(0.5, 2.0, 2)
            world.add_outline(f'room{k}', [[(x, y), (x + sx, y), (x + sx, y + sy), (x, y + sy)]], (0.0, 2.5))
        for k in range(6):
            world.add_box(f'box{k}', (0.6, 0.8), (*rng.uniform(-8.0, 8.0, 2), rng.uniform(-3.0, 3.0)), (0.0, 1.0))
        names = [f'r{k}' for k in range(20)]
        for k, name in enumerate(names):
            world.add_robot(name, (*rng.uniform(-8.0, 8.0, 2), rng.uniform(-3.0, 3.0)), 0.0, DRIVE, footprint=OFFSET if k % 2 else None, radius=0.25)
        states = []
        for i in range(120):
            for name in names:
                world.command(name, tuple(rng.uniform((-0.3, -0.3, -1.5), (1.0, 0.3, 1.5))))
            world.advance((i + 1) * DT)
            states.extend(_pose(world, name) for name in names)
        return np.array(states), np.array([world.cast(name, FAN).first() for name in names])

    first, second = run(), run()
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_wall_on_a_cell_boundary_stops_the_robot(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    world.add_outline('far', [[(-1.1, 5.0), (-0.9, 5.0), (-0.9, 5.2), (-1.1, 5.2)]], (0.0, 2.5))
    world.add_outline('wall', [[(-0.1, 0.0), (-0.1, 3.0)]], (0.0, 2.5))
    world.add_robot('robot', (-1.0, 1.5, 0.0), 0.0, STIFF, footprint=SQUARE)
    _drive(world, [(3.0, {'robot': (1.0, 0.0, 0.0)})])
    assert world.robot_state('robot').x == pytest.approx(-0.3, abs=0.02)


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_entity_added_again_stands_at_its_new_pose(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, radius=0.25)
    world.add_box('box', (0.2, 2.0), (1.0, 0.0, 0.0), (0.0, 2.0))
    world.add_disc('ped', 0.3, (-1.0, 0.0), (0.0, 1.8))
    world.advance(DT)
    world.remove('box')
    world.add_box('box', (0.2, 2.0), (3.0, 0.0, 0.0), (0.0, 2.0))
    world.add_disc('ped', 0.3, (-4.0, 0.0), (0.0, 1.8))
    world.advance(2 * DT)
    east = attrs.evolve(FAN, mount=(0.0, 0.0, 0.0), angle_min=0.0, angle_increment=math.pi, beams=2, height=0.0)
    assert world.scan('robot', east).tolist() == pytest.approx([2.9, 3.7], abs=1e-3)


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_first_many_follows_new_rays_and_keeps_old_results(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    _wall(world, 'east', (2.0, -3.0), (2.0, 3.0))
    _wall(world, 'north', (-3.0, 1.0), (3.0, 1.0))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, radius=0.25)
    spec = attrs.evolve(FAN, mount=(0.0, 0.0, 0.0), angle_min=0.0, angle_increment=math.pi / 2, beams=4)
    kept = world.first_many([('robot', spec, np.array([0]), np.zeros(1))])[0].copy()
    alias = world.first_many([('robot', spec, np.array([0]), np.zeros(1))])[0]
    for beam, want in ((1, 0.95), (0, 1.95), (1, 0.95)):
        assert world.first_many([('robot', spec, np.array([beam]), np.zeros(1))])[0] == pytest.approx(want, abs=1e-3)
    assert alias == pytest.approx(kept, abs=1e-6)
    both = world.first_many([('robot', spec, np.array([1, 0, 1]), np.zeros(3))])[0]
    assert both == pytest.approx([0.95, 1.95, 0.95], abs=1e-3)


def test_coincident_robots_part_alike():
    def scene(world: World) -> np.ndarray:
        world.add_robot('a', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        world.add_robot('b', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        world.add_robot('c', (4.0, 0.0, 0.0), 0.0, STIFF, radius=0.25)
        world.add_robot('d', (4.0, 0.0, 0.0), 0.0, STIFF, radius=0.25)
        return _drive(world, [(1.0, {name: (0.0, 0.0, 0.0) for name in 'abcd'})])

    box2d, warp = _both(scene)
    assert np.abs(box2d[-4:, :2].mean(axis=0) - warp[-4:, :2].mean(axis=0)).max() < 0.02
    assert np.abs(warp[-2:, :2] - (4.0, 0.0)).max() < 1e-6
    assert np.abs(warp[-4, 0] + warp[-3, 0]) < 0.05


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_outline_moved_to_another_level_stops_robots_only_there(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    world.set_levels([0.0, 3.0])
    world.add_outline('wall', [[(1.0, -3.0), (1.0, 3.0)]], (0.0, 2.5))
    world.add_robot('ground', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
    world.add_robot('upper', (0.0, 0.0, 0.0), 3.0, STIFF, footprint=SQUARE)
    world.advance(DT)
    world.move('wall', (0.0, 0.0, 0.0), z=(3.0, 5.5))
    _drive(world, [(2.0, {'ground': (1.0, 0.0, 0.0), 'upper': (1.0, 0.0, 0.0)})])
    assert world.robot_state('upper').x == pytest.approx(0.8, abs=0.02)
    assert world.robot_state('ground').x == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize('physics', [Physics.BOX2D, Physics.WARP], ids=lambda physics: physics.value)
def test_far_away_box_leaves_rays_and_contact_working(physics: Physics):
    world = make_world(DT, Dynamics.RIGID, physics)
    _wall(world, 'east', (2.0, -3.0), (2.0, 3.0))
    world.add_box('parked', (0.5, 0.5), (0.0, 3.0, 0.0), (0.0, 2.0))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
    world.move('parked', (1.0e5, 1.0e5, 0.0))
    _drive(world, [(3.0, {'robot': (1.0, 0.0, 0.0)})])
    assert world.robot_state('robot').x == pytest.approx(1.75, abs=0.02)
    assert world.scan('robot', attrs.evolve(FAN, mount=(0.0, 0.0, 0.0), height=0.0))[180] == pytest.approx(0.2, abs=0.02)


def test_corner_passing_a_wall_end_at_a_small_gap_alike():
    def scene(world: World) -> np.ndarray:
        world.add_outline('stub', [[(2.0, 0.21), (2.5, 0.21), (2.5, 3.0), (2.0, 3.0)]], (0.0, 2.5))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        return _drive(world, [(4.0, {'robot': (0.6, 0.0, 0.0)})])

    box2d, warp = _both(scene)
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.03
    assert np.abs(warp[:, 2] - box2d[:, 2]).max() < 0.05


def test_spin_is_capped_alike():
    fast = DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(0.0, 0.0), angular=Axis(-100.0, 100.0))

    def scene(world: World) -> np.ndarray:
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, fast, radius=0.25)
        return _drive(world, [(1.0, {'robot': (0.0, 0.0, 30.0)})])

    box2d, warp = _both(scene)
    assert box2d[-1, 5] == pytest.approx(0.25 * math.pi / DT, abs=1e-2)
    assert warp[-1, 5] == pytest.approx(box2d[-1, 5], abs=1e-2)


def test_wall_stops_the_robot_alike_at_a_long_step():
    slow = 0.05

    def scene(world: World) -> np.ndarray:
        _wall(world, 'wall', (2.5, -3.0), (2.5, 3.0))
        world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, STIFF, footprint=SQUARE)
        out = []
        for _ in range(80):
            world.command('robot', (1.0, 0.0, 0.0))
            world.advance(world.time + slow)
            out.append(_pose(world, 'robot'))
        return np.array(out)

    box2d = scene(make_world(slow, Dynamics.RIGID, Physics.BOX2D))
    warp = scene(make_world(slow, Dynamics.RIGID, Physics.WARP))
    assert np.abs(warp[:, :2] - box2d[:, :2]).max() < 0.02

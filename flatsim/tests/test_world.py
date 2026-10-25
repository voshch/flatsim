"""FlatWorld kernel: rays, heights, levels, drive limits, acceleration and contact."""

from __future__ import annotations

import math

import attrs
import numpy as np
import pytest

from flatsim.drive import Axis, DriveModel
from flatsim.world import MAX_LEVELS, Dynamics, FlatWorld, ScanSpec

DT = 1.0 / 30.0
FREE = DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(0.0, 0.0), angular=Axis(-2.0, 2.0))
HOLONOMIC = DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(-1.0, 1.0), angular=Axis(-2.0, 2.0))
LIMITED = DriveModel(linear=Axis(-0.5, 0.5, 2.5), lateral=Axis(0.0, 0.0), angular=Axis(-1.5, 1.5, 3.2))
FAN = ScanSpec(mount=(0.0, 0.0, 0.0), angle_min=-math.pi, angle_increment=2 * math.pi / 640, beams=640, range_min=0.0, range_max=12.0)
AHEAD = ScanSpec(mount=(0.0, 0.0, 0.0), angle_min=0.0, angle_increment=0.1, beams=1, range_min=0.0, range_max=12.0)
SQUARE = [(0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2)]


def _wall(world: FlatWorld, name: str, x: float, z: tuple[float, float] = (0.0, 2.5)) -> None:
    world.add_outline(name, [[(x, -3.0), (x, 3.0)]], z)


def _ahead(world: FlatWorld, name: str = 'robot', height: float = 0.0) -> float:
    return float(world.scan(name, attrs.evolve(AHEAD, height=height))[0])


def test_scan_origin_and_heading_follow_the_mount():
    world = FlatWorld(DT)
    _wall(world, 'east', 2.0)
    world.add_outline('north', [[(-3.0, 1.0), (3.0, 1.0)]], (0.0, 2.5))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    forward = ScanSpec(mount=(0.5, 0.0, 0.0), angle_min=0.0, angle_increment=0.1, beams=1, range_min=0.0, range_max=12.0)
    left = ScanSpec(mount=(0.0, 0.25, math.pi / 2), angle_min=0.0, angle_increment=0.1, beams=1, range_min=0.0, range_max=12.0)
    assert world.scan('robot', forward)[0] == pytest.approx(1.5, abs=1e-4)
    assert world.scan('robot', left)[0] == pytest.approx(0.75, abs=1e-4)


def test_robot_sees_and_hits_only_its_own_level():
    world = FlatWorld(DT)
    world.set_levels([0.0, 3.0])
    _wall(world, 'ground', 2.0, (0.0, 2.5))
    _wall(world, 'upper', 1.0, (3.0, 5.5))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    assert _ahead(world) == pytest.approx(2.0, abs=1e-4)
    world.command('robot', (1.0, 0.0, 0.0))
    for i in range(1, 12):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(i * 0.4)
    assert world.robot_state('robot').x == pytest.approx(1.8, abs=0.02)

    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 3.0)
    assert world.level(3.0) == 1
    assert _ahead(world) == pytest.approx(1.0, abs=1e-4)
    for i in range(1, 12):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(world.time + 0.4)
    assert world.robot_state('robot').x == pytest.approx(0.8, abs=0.02)


def test_wall_spanning_two_levels_is_seen_from_both():
    world = FlatWorld(DT)
    world.set_levels([0.0, 3.0])
    _wall(world, 'shaft', 2.0, (0.0, 6.0))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    assert _ahead(world) == pytest.approx(2.0, abs=1e-4)
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 3.0)
    assert _ahead(world) == pytest.approx(2.0, abs=1e-4)


def test_levels_set_after_spawn_refilter_existing_shapes():
    world = FlatWorld(DT)
    _wall(world, 'upper', 1.0, (3.0, 5.5))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    assert _ahead(world, height=3.5) == pytest.approx(1.0, abs=1e-4)
    world.set_levels([0.0, 3.0])
    assert math.isinf(_ahead(world, height=3.5))


def test_scan_sees_only_entities_spanning_the_sensor_height():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('crate', (0.2, 2.0), (1.0, 0.0, 0.0), (0.0, 0.3))
    world.add_box('sign', (0.2, 2.0), (2.0, 0.0, 0.0), (1.2, 2.0))
    _wall(world, 'wall', 3.0)
    assert _ahead(world) == pytest.approx(0.9, abs=1e-4)
    assert _ahead(world, height=0.25) == pytest.approx(0.9, abs=1e-4)
    assert _ahead(world, height=0.35) == pytest.approx(3.0, abs=1e-4)
    assert _ahead(world, height=1.5) == pytest.approx(1.9, abs=1e-4)
    assert _ahead(world, height=2.2) == pytest.approx(3.0, abs=1e-4)
    assert math.isinf(_ahead(world, height=2.6))
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 0.1)
    assert _ahead(world, height=0.25) == pytest.approx(3.0, abs=1e-4)
    world.move('crate', (1.0, 0.0, 0.0), z=(0.0, 0.5))
    assert _ahead(world, height=0.25) == pytest.approx(0.9, abs=1e-4)


def test_cast_lists_every_entity_on_a_beam_nearest_first_with_its_extent():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.5, FREE)
    world.add_box('wall', (0.2, 6.0), (3.1, 0.0, 0.0), (-0.5, 2.5))
    world.add_disc('ped', 0.3, (2.0, 0.0), (0.5, 2.3))
    world.add_box('crate', (0.2, 2.0), (1.0, 0.0, 0.0), (0.5, 0.8))
    hits = world.cast('robot', attrs.evolve(AHEAD, height=0.25))
    assert hits.distance.shape == (1, 3)
    assert hits.distance[0] == pytest.approx([0.9, 1.7, 3.0], abs=1e-4)
    assert hits.lo[0] == pytest.approx([-0.25, -0.25, -1.25])
    assert hits.hi[0] == pytest.approx([0.05, 1.55, 1.75])
    assert hits.first()[0] == pytest.approx(0.9, abs=1e-4)
    assert hits.first(0.1)[0] == pytest.approx(1.7, abs=1e-4)
    assert hits.floor == pytest.approx(-0.75)
    assert hits.first(-0.3)[0] == pytest.approx(2.5, abs=1e-4)
    assert math.isinf(hits.first(1.0)[0])
    assert hits.first(np.array([[0.0], [0.1], [-0.3]]))[:, 0] == pytest.approx([0.9, 1.7, 2.5], abs=1e-4)
    side = ScanSpec(mount=(0.0, 0.0, 0.0), angle_min=math.pi / 2, angle_increment=0.1, beams=2, range_min=0.0, range_max=12.0)
    empty = world.cast('robot', side)
    assert empty.distance.shape == (2, 1)
    assert np.isinf(empty.first()).all()


def test_ground_slab_neither_blocks_a_ray_nor_stops_a_rigid_robot():
    world = FlatWorld(DT)
    world.add_box('cabin_floor', (2.0, 2.0), (1.5, 0.0, 0.0), (-0.04, 0.01))
    world.add_box('upper_cabin_floor', (2.0, 2.0), (1.5, 0.0, 0.0), (2.96, 3.01))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    assert _ahead(world) == pytest.approx(0.5, abs=1e-4)
    world.set_levels([0.0, 3.0])
    assert math.isinf(_ahead(world))
    for i in range(1, 11):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(i * 0.4)
    assert world.robot_state('robot').x == pytest.approx(4.0, abs=1e-3)
    assert world.contacts('robot') == []

    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 3.0)
    assert math.isinf(_ahead(world))
    world.move('upper_cabin_floor', (1.5, 0.0, 0.0), z=(2.96, 3.06))
    assert _ahead(world) == pytest.approx(0.5, abs=1e-4)
    for _ in range(10):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(world.time + 0.4)
    assert world.robot_state('robot').x == pytest.approx(0.3, abs=0.02)
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 0.0)
    assert math.isinf(_ahead(world))


def test_wall_ending_just_above_the_next_floor_stays_on_the_lower_level_only():
    world = FlatWorld(DT)
    world.set_levels([0.0, 3.0])
    _wall(world, 'wall', 2.0, (0.0, 3.01))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    assert _ahead(world) == pytest.approx(2.0, abs=1e-4)
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 3.0)
    assert math.isinf(_ahead(world))


def test_more_levels_than_supported_are_rejected():
    with pytest.raises(ValueError, match=str(MAX_LEVELS)):
        FlatWorld(DT).set_levels([3.0 * i for i in range(MAX_LEVELS + 1)])


def test_box_blocks_a_ray_until_moved_or_removed():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_box('door', (0.1, 2.0), (1.5, 0.0, 0.0), (0.0, 2.0))
    assert _ahead(world) == pytest.approx(1.45, abs=1e-4)
    world.move('door', (1.5, 0.0, math.pi / 2))
    assert _ahead(world) == pytest.approx(0.5, abs=1e-4)
    world.move('door', (1.5, 3.0, 0.0))
    assert math.isinf(_ahead(world))
    world.move('door', (1.5, 0.0, 0.0))
    assert world.remove('door')
    assert 'door' not in world
    assert math.isinf(_ahead(world))


def test_disc_keeps_its_velocity_between_moves():
    world = FlatWorld(DT)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.add_disc('ped', 0.3, (6.0, 0.0), (0.0, 1.8))
    assert _ahead(world) == pytest.approx(5.7, abs=1e-4)
    world.move('ped', (6.0, 0.0, 0.0), velocity=(-1.0, 0.0))
    world.advance(2.0)
    assert _ahead(world) == pytest.approx(3.7, abs=1e-3)


def test_command_is_clamped_by_velocity_and_acceleration_limits():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, LIMITED)
    speed, distance = 0.0, 0.0
    for i in range(1, 16):
        world.command('robot', (10.0, 0.0, 0.0))
        world.advance(i * DT)
        speed = min(0.5, speed + 2.5 * DT)
        distance += speed * DT
        state = world.robot_state('robot')
        assert state.vx == pytest.approx(speed, abs=1e-6)
        assert state.x == pytest.approx(distance, abs=1e-5)
    assert speed == 0.5

    turn = 0.0
    for i in range(16, 40):
        world.command('robot', (0.0, 0.0, -9.0))
        world.advance(i * DT)
        turn = max(-1.5, turn - 3.2 * DT)
        assert world.robot_state('robot').wz == pytest.approx(turn, abs=1e-6)
    assert turn == -1.5


def test_diff_drive_ignores_lateral_command_and_holonomic_follows_it():
    for drive, sideways in ((FREE, 0.0), (HOLONOMIC, 1.0)):
        world = FlatWorld(DT, Dynamics.KINEMATIC)
        world.add_robot('robot', (0.0, 0.0, math.pi / 2), 0.0, drive)
        world.command('robot', (0.0, 1.0, 0.0))
        world.advance(0.3)
        state = world.robot_state('robot')
        assert state.vy == pytest.approx(sideways)
        assert state.x == pytest.approx(-0.3 * sideways, abs=1e-5)
        assert state.y == pytest.approx(0.0, abs=1e-5)


def test_command_expires_after_the_timeout():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.command('robot', (1.0, 0.0, 0.0))
    world.advance(FREE.timeout + 2.0)
    state = world.robot_state('robot')
    assert state.vx == 0.0
    assert FREE.timeout - DT <= state.x <= FREE.timeout + 2 * DT


def test_non_finite_command_is_dropped():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.command('robot', (0.4, 0.0, 0.0))
    world.command('robot', (math.nan, 0.0, 0.0))
    world.advance(0.2)
    assert world.robot_state('robot').vx == pytest.approx(0.4)


@pytest.mark.parametrize(('dynamics', 'stops'), [(Dynamics.RIGID, True), (Dynamics.KINEMATIC, False)])
def test_wall_stops_a_rigid_robot_and_lets_a_kinematic_one_through(dynamics: Dynamics, stops: bool):
    world = FlatWorld(DT, dynamics)
    world.add_outline('wall', [[(2.0, -3.0), (2.1, -3.0), (2.1, 3.0), (2.0, 3.0)]], (0.0, 2.5))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    for i in range(1, 11):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(i * 0.4)
    state = world.robot_state('robot')
    if stops:
        assert state.x == pytest.approx(1.8, abs=0.02)
        assert abs(state.vx) < 0.05
    else:
        assert state.x == pytest.approx(4.0, abs=1e-3)


@pytest.mark.parametrize('dynamics', [Dynamics.RIGID, Dynamics.KINEMATIC])
def test_walking_disc_touches_a_robot_without_pushing_it(dynamics: Dynamics):
    world = FlatWorld(DT, dynamics)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, radius=0.3)
    world.add_disc('ped', 0.3, (-2.0, 0.0), (0.0, 1.8))
    world.move('ped', (-2.0, 0.0, 0.0), velocity=(1.0, 0.0))
    world.advance(1.8)
    assert [name for name, _, _ in world.contacts('robot')] == ['ped']
    world.advance(3.0)
    state = world.robot_state('robot')
    assert (state.x, state.y, state.vx) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_rigid_robot_still_reports_a_disc_after_a_level_change():
    world = FlatWorld(DT, Dynamics.RIGID)
    world.set_levels([0.0, 3.0])
    world.add_robot('robot', (0.0, 0.0, 0.0), 3.0, FREE, radius=0.3)
    world.add_disc('ped', 0.3, (0.0, 0.5), (3.0, 4.8))
    assert [name for name, _, _ in world.contacts('robot')] == ['ped']
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 0.0)
    assert world.contacts('robot') == []


def test_robots_on_one_level_see_each_other_but_never_themselves():
    world = FlatWorld(DT)
    world.set_levels([0.0, 3.0])
    world.add_robot('a', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    world.add_robot('b', (2.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    inside = ScanSpec(mount=(-0.1, 0.0, 0.0), angle_min=0.0, angle_increment=0.1, beams=1, range_min=0.0, range_max=12.0)
    assert world.scan('a', inside)[0] == pytest.approx(1.9, abs=1e-4)
    world.set_robot_pose('b', (2.0, 0.0, 0.0), 3.0)
    assert math.isinf(world.scan('a', inside)[0])
    world.set_robot_pose('b', (2.0, 0.0, 0.0), 0.0)
    assert world.remove('b')
    assert math.isinf(world.scan('a', inside)[0])


def test_peer_robot_is_seen_up_to_its_height():
    world = FlatWorld(DT)
    world.add_robot('a', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    world.add_robot('b', (2.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    world.add_robot('tall', (4.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE, height=1.4)
    assert _ahead(world, 'a', height=0.5) == pytest.approx(1.8, abs=1e-4)
    assert _ahead(world, 'a', height=0.6) == pytest.approx(3.8, abs=1e-4)
    assert math.isinf(_ahead(world, 'a', height=1.5))
    world.set_robot_pose('b', (2.0, 0.0, 0.0), 0.2)
    assert _ahead(world, 'a', height=0.6) == pytest.approx(1.8, abs=1e-4)
    assert _ahead(world, 'a', height=0.1) == pytest.approx(3.8, abs=1e-4)


def test_acceleration_follows_a_commanded_speed_up_and_vanishes_at_speed():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, LIMITED)
    assert (world.robot_state('robot').ax, world.robot_state('robot').ay) == (0.0, 0.0)
    for i in range(1, 7):
        world.command('robot', (10.0, 0.0, 0.0))
        world.advance(i * DT)
        state = world.robot_state('robot')
        assert state.ax == pytest.approx(2.5, abs=1e-9)
        assert state.ay == 0.0
    for i in range(7, 12):
        world.command('robot', (10.0, 0.0, 0.0))
        world.advance(i * DT)
    state = world.robot_state('robot')
    assert state.vx == 0.5
    assert (state.ax, state.ay) == (0.0, 0.0)
    world.command('robot', (0.0, 0.0, 0.0))
    world.advance(12 * DT)
    assert world.robot_state('robot').ax == pytest.approx(-2.5, abs=1e-9)


@pytest.mark.parametrize(('dynamics', 'tolerance'), [(Dynamics.RIGID, 0.05), (Dynamics.KINEMATIC, 1e-9)])
def test_steady_turn_leaves_only_the_centripetal_acceleration(dynamics: Dynamics, tolerance: float):
    world = FlatWorld(DT, dynamics)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, HOLONOMIC, radius=0.3)
    for i in range(1, 31):
        world.command('robot', (0.6, 0.0, 1.5))
        world.advance(i * DT)
    state = world.robot_state('robot')
    assert state.wz == pytest.approx(1.5)
    assert state.ax == pytest.approx(0.0, abs=tolerance)
    assert state.ay == pytest.approx(0.6 * 1.5, abs=tolerance)
    for i in range(31, 61):
        world.command('robot', (0.0, 0.4, -1.0))
        world.advance(i * DT)
    state = world.robot_state('robot')
    assert state.ax == pytest.approx(0.4 * 1.0, abs=tolerance)
    assert state.ay == pytest.approx(0.0, abs=tolerance)
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 0.0)
    assert (world.robot_state('robot').ax, world.robot_state('robot').ay) == (0.0, 0.0)


def test_rigid_robot_reports_the_wall_it_rests_against():
    world = FlatWorld(DT, Dynamics.RIGID)
    world.add_outline('wall', [[(2.0, -3.0), (2.1, -3.0), (2.1, 3.0), (2.0, 3.0)]], (0.0, 2.5))
    world.add_disc('ped', 0.3, (0.0, 5.0), (0.0, 1.8))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, footprint=SQUARE)
    assert world.contacts('robot') == []
    for i in range(1, 11):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(i * 0.4)
    ((name, point, normal),) = world.contacts('robot')
    assert name == 'wall'
    assert point == pytest.approx((2.0, 0.0), abs=0.01)
    assert normal == pytest.approx((1.0, 0.0), abs=1e-6)
    world.advance(world.time + 2.0)
    assert [name for name, _, _ in world.contacts('robot')] == ['wall']
    world.set_robot_pose('robot', (0.0, 0.0, 0.0), 0.0)
    assert world.contacts('robot') == []


def test_kinematic_robot_reports_what_its_shape_overlaps():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.set_levels([0.0, 3.0])
    world.add_box('door', (0.1, 2.0), (1.0, 0.0, 0.0), (0.0, 2.0))
    world.add_box('upstairs', (0.1, 2.0), (0.3, 0.0, 0.0), (3.0, 5.0))
    world.add_disc('ped', 0.3, (0.0, 0.55), (0.0, 1.8))
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE, radius=0.3)
    world.add_robot('peer', (0.0, -0.5, 0.0), 0.0, FREE, footprint=SQUARE)
    touches = {name: (point, normal) for name, point, normal in world.contacts('robot')}
    assert sorted(touches) == ['ped', 'peer']
    assert touches['ped'][0] == pytest.approx((0.0, 0.275), abs=1e-6)
    assert touches['ped'][1] == pytest.approx((0.0, 1.0), abs=1e-6)
    assert touches['peer'][0] == pytest.approx((0.0, -0.3), abs=1e-6)
    assert touches['peer'][1] == pytest.approx((0.0, -1.0), abs=1e-6)
    ((name, _, normal),) = world.contacts('peer')
    assert name == 'robot'
    assert normal == pytest.approx((0.0, 1.0), abs=1e-6)

    world.remove('ped')
    world.remove('peer')
    seen: list[list[str]] = []
    for i in range(1, 46):
        world.command('robot', (1.0, 0.0, 0.0))
        world.advance(i * DT)
        seen.append([name for name, _, _ in world.contacts('robot')])
    assert seen[0] == [] and seen[-1] == []
    touching = [i for i, names in enumerate(seen) if names == ['door']]
    assert touching == list(range(touching[0], touching[-1] + 1))
    assert (touching[0] + 1) * DT == pytest.approx(0.65, abs=DT)
    assert (touching[-1] + 1) * DT == pytest.approx(1.35, abs=DT)


def test_advance_takes_only_whole_steps_and_rebases_on_a_clock_that_ran_backwards():
    world = FlatWorld(DT, Dynamics.KINEMATIC)
    world.add_robot('robot', (0.0, 0.0, 0.0), 0.0, FREE)
    world.command('robot', (1.0, 0.0, 0.0))
    assert world.advance(2.5 * DT) == 2
    assert world.time == pytest.approx(2 * DT)
    assert world.robot_state('robot').x == pytest.approx(2 * DT, abs=1e-6)
    assert world.advance(2.9 * DT) == 0
    assert world.advance(3 * DT) == 1
    assert world.time == 3 * DT
    assert world.advance(6 * DT + 1e-7) == 3
    assert world.advance(1.0 * DT) == 0
    assert world.time == 1.0 * DT


def test_outline_drops_edges_box2d_refuses_and_moves_again():
    world = FlatWorld(1 / 30)
    ring = [(0.0, 0.0), (0.003, 0.0), (1.0, 0.0), (1.0, 0.5), (0.001, 0.5), (0.0, 0.5), (0.0, 0.002)]
    for x in (0.0, 0.5, 1.0):
        world.add_outline('lid', [[(px + x, py) for px, py in ring]], (0.6, 0.8))
    shapes = world._entities['lid'].shapes
    assert len(shapes) == 4
    assert sorted(world._owners) == sorted(shape.id for shape in shapes)
    assert world.remove('lid')
    assert world._owners == {}

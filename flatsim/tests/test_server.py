"""flatsim_server on a real graph: its clock, stepping, its one world and the topics of a spawned testbot."""

from __future__ import annotations

import math
import time

import pytest

from conftest import ACCEL, DT, TIMEOUT, TOP_SPEED, TURN_GAIN, ahead, seconds, until

WALL_X = 3.0
LIDAR_X = 0.1
REAR_X = 0.2


def test_paused_server_steps_exactly_on_request(flatsim) -> None:
    bench = flatsim()
    start = bench.step(0)
    assert start == 0.0
    until(lambda: len(bench.clock) >= 3, 5.0, 'idle clock of a paused server')
    assert set(bench.clock) == {0.0}

    assert bench.step(3) == pytest.approx(3 * DT, abs=1e-9)
    until(lambda: bool(bench.clock) and bench.clock[-1] == pytest.approx(3 * DT, abs=1e-9), 5.0, 'clock of the third step')
    stepped = sorted(set(bench.clock))
    assert stepped == pytest.approx([0.0, DT, 2 * DT, 3 * DT], abs=1e-9)

    refused = bench.control(False, 2)
    assert not refused.success
    assert seconds(refused.time) == pytest.approx(3 * DT, abs=1e-9)
    time.sleep(0.2)
    assert bench.clock[-1] == pytest.approx(3 * DT, abs=1e-9)


def test_unpaused_server_runs_at_its_real_time_factor(flatsim) -> None:
    import rclpy.parameter

    bench = flatsim(paused=False, real_time_factor=2.0)
    until(lambda: len(bench.clock) > 5, 5.0, 'free-running clock')
    first, wall = bench.clock[-1], time.monotonic()
    time.sleep(1.0)
    rate = (bench.clock[-1] - first) / (time.monotonic() - wall)
    assert rate == pytest.approx(2.0, rel=0.15)

    paused = bench.control(True)
    assert paused.success
    frozen = seconds(paused.time)
    time.sleep(0.3)
    assert bench.clock[-1] == pytest.approx(frozen, abs=1e-9)

    bench.server.set_parameters([rclpy.parameter.Parameter('real_time_factor', value=0.0)])
    assert bench.control(False).success
    time.sleep(0.5)
    reached = seconds(bench.control(True).time)
    assert reached - frozen > 10.0


def test_cmd_vel_drives_odom_tf_and_joint_states(flatsim) -> None:
    bench = flatsim()
    robot = bench.spawn()
    first = robot.odom[-1]
    assert (first.header.frame_id, first.child_frame_id) == ('testbot/odom', 'testbot/base_link')
    assert robot.joints[-1].name == ['left_wheel_joint', 'right_wheel_joint']
    assert [(t.header.frame_id, t.child_frame_id) for t in robot.tf[-1].transforms] == [('map', 'testbot/odom'), ('testbot/odom', 'testbot/base_link')]

    steps = 24
    robot.command(x=10.0, wz=1.0)
    bench.settle(robot, bench.step(steps))
    speed, travelled = 0.0, 0.0
    for _ in range(steps):
        speed = min(TOP_SPEED, speed + ACCEL * DT)
        travelled += speed * DT
    last = robot.odom[-1]
    assert last.twist.twist.linear.x == pytest.approx(TOP_SPEED, abs=1e-6)
    assert last.twist.twist.angular.z == pytest.approx(1.0, abs=1e-6)
    assert robot.imu[-1].angular_velocity.z == pytest.approx(TURN_GAIN, abs=1e-6)
    yaw = 2.0 * math.atan2(last.pose.pose.orientation.z, last.pose.pose.orientation.w)
    assert yaw == pytest.approx(TURN_GAIN * steps * DT, abs=1e-3)
    assert math.hypot(last.pose.pose.position.x, last.pose.pose.position.y) < travelled
    base = robot.tf[-1].transforms[1].transform
    assert (base.translation.x, base.translation.y) == pytest.approx((last.pose.pose.position.x, last.pose.pose.position.y), abs=1e-9)

    expired = round(TIMEOUT / DT) + 5
    bench.settle(robot, bench.step(expired))
    stopped = robot.odom[-1].twist.twist
    assert (stopped.linear.x, stopped.angular.z) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_straight_drive_follows_the_acceleration_limit(flatsim) -> None:
    bench = flatsim()
    robot = bench.spawn()
    rest = robot.imu[-1]
    assert rest.header.frame_id == 'testbot/imu_link'
    assert (rest.linear_acceleration.x, rest.linear_acceleration.y, rest.linear_acceleration.z) == pytest.approx((0.0, 0.0, 9.80665))

    robot.command(x=10.0)
    bench.settle(robot, bench.step(3))
    assert robot.imu[-1].linear_acceleration.x == pytest.approx(ACCEL, abs=1e-3)

    bench.settle(robot, bench.step(21))
    speed, travelled = 0.0, 0.0
    for _ in range(24):
        speed = min(TOP_SPEED, speed + ACCEL * DT)
        travelled += speed * DT
    assert robot.odom[-1].pose.pose.position.x == pytest.approx(travelled, abs=2e-3)
    assert robot.odom[-1].pose.pose.position.y == pytest.approx(0.0, abs=1e-6)


def test_scans_see_boxes_and_discs(flatsim) -> None:
    from flatsim_msgs.msg import Disc, Discs
    from flatsim_msgs.srv import DeleteEntities

    bench = flatsim()
    bench.box('wall', WALL_X)
    bench.box('back', -2.0)
    robot = bench.spawn()
    scan = robot.scans[-1]
    assert len(scan.ranges) == 361
    assert (scan.range_min, scan.range_max) == pytest.approx((0.1, 20.0))
    assert scan.header.frame_id == 'testbot/lidar_link'
    gap = WALL_X - 0.05 - LIDAR_X
    assert ahead(scan) == pytest.approx(gap, abs=1e-3)
    rear = robot.rear[-1]
    assert len(rear.ranges) == 101
    assert rear.range_max == pytest.approx(8.0)
    assert ahead(rear) == pytest.approx(2.0 - 0.05 - REAR_X, abs=1e-3)

    def seen() -> float:
        bench.republish()
        time.sleep(0.05)
        return ahead(robot.scans[-1])

    walker = Disc(name='walker', radius=0.3, x=1.5, y=0.0, z_min=0.0, z_max=1.8)
    bench.discs.publish(Discs(discs=[walker]))
    until(lambda: seen() == pytest.approx(1.5 - 0.3 - LIDAR_X, abs=1e-3), 5.0, 'scan hitting the disc')
    walker.y = 4.0
    bench.discs.publish(Discs(discs=[walker]))
    until(lambda: seen() == pytest.approx(gap, abs=1e-3), 5.0, 'scan back on the wall after the disc moved')
    walker.y = 0.0
    bench.discs.publish(Discs(discs=[walker]))
    until(lambda: seen() < gap - 1.0, 5.0, 'scan hitting the disc again')
    bench.discs.publish(Discs(removed=['walker']))
    until(lambda: seen() == pytest.approx(gap, abs=1e-3), 5.0, 'scan back on the wall after the disc was removed')

    assert list(bench.call('delete_entities', DeleteEntities.Request(names=['wall', 'walker', 'testbot'])).removed) == [True, False, False]
    assert math.isinf(seen())


def test_outlines_and_moved_boxes_block_the_scan(flatsim) -> None:
    from flatsim_msgs.msg import EntityPose, EntityPoses, Outline, Ring
    from flatsim_msgs.srv import AddOutlines

    bench = flatsim()
    square = Ring(x=[4.0, 6.0, 6.0, 4.0], y=[-1.0, -1.0, 1.0, 1.0])
    assert bench.call('add_outlines', AddOutlines.Request(outlines=[Outline(name='pillar', rings=[square], z_min=0.0, z_max=2.0)])).success
    bench.box('door', 2.0)
    robot = bench.spawn()
    assert ahead(robot.scans[-1]) == pytest.approx(2.0 - 0.05 - LIDAR_X, abs=1e-3)

    aside = EntityPose(name='door', x=2.0, y=10.0, yaw=0.0, z_min=0.0, z_max=2.0)
    bench.poses.publish(EntityPoses(poses=[aside, EntityPose(name='nothing'), EntityPose(name='testbot', x=9.0)]))

    def seen() -> float:
        bench.republish()
        time.sleep(0.05)
        return ahead(robot.scans[-1])

    until(lambda: seen() == pytest.approx(4.0 - LIDAR_X, abs=1e-3), 5.0, 'scan on the outline behind the moved box')
    assert robot.odom[-1].pose.pose.position.x == pytest.approx(0.0)


def test_concurrent_calls_beyond_the_default_queue_all_get_answers(flatsim) -> None:
    from flatsim_msgs.msg import Box
    from flatsim_msgs.srv import SpawnBoxes

    bench = flatsim(paused=False)
    futures = [bench.clients['spawn_boxes'].call_async(SpawnBoxes.Request(boxes=[Box(name=f'door_{i}', size_x=1.0, size_y=0.1, x=float(i), y=5.0, z_max=2.0)])) for i in range(60)]
    until(lambda: all(future.done() for future in futures), 10.0, 'answers to 60 concurrent calls')
    assert all(future.result().success for future in futures)


def test_lidar_keeps_its_update_rate_and_fills_a_sixteen_ring_cloud(flatsim) -> None:
    import numpy as np
    import sensor_msgs.msg

    bench = flatsim()
    bench.box('wall', WALL_X)
    robot = bench.spawn()
    scans, imus, rears = len(robot.scans), len(robot.imu), len(robot.rear)
    bench.settle(robot, bench.step(30))
    stamps = [seconds(scan.header.stamp) for scan in robot.scans[scans:]]
    assert len(stamps) == 10
    assert all(later - earlier == pytest.approx(3 * DT, abs=1e-6) for earlier, later in zip(stamps, stamps[1:], strict=False))
    assert robot.scans[-1].scan_time == pytest.approx(3 * DT)
    assert len(robot.imu) - imus == 30
    assert len(robot.rear) - rears == 30

    cloud = robot.clouds[-1]
    scan = robot.scans[-1]
    assert cloud.header.stamp == scan.header.stamp
    assert cloud.header.frame_id == 'testbot/lidar_link'
    assert (cloud.height, cloud.point_step, cloud.is_bigendian) == (1, 32, False)
    assert cloud.row_step == 32 * cloud.width == len(cloud.data)
    float32, uint16 = sensor_msgs.msg.PointField.FLOAT32, sensor_msgs.msg.PointField.UINT16
    assert [(field.name, field.offset, field.datatype, field.count) for field in cloud.fields] == [('x', 0, float32, 1), ('y', 4, float32, 1), ('z', 8, float32, 1), ('intensity', 16, float32, 1), ('ring', 24, uint16, 1)]
    layout = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'], 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'], 'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
    points = np.frombuffer(bytes(cloud.data), dtype=layout)
    assert set(points['ring'].tolist()) == set(range(16))
    front = points[(points['ring'] == 8) & (np.abs(points['y']) < 0.02)]
    assert len(front) > 0
    assert front['x'] == pytest.approx(np.full(len(front), ahead(scan)), abs=1e-3)
    assert front['z'] == pytest.approx(front['x'] * math.tan(0.261799 / 15), abs=1e-3)


def test_rigid_robot_stops_at_a_box_and_reports_the_contact(flatsim) -> None:
    bench = flatsim()
    bench.box('wall', WALL_X)
    robot = bench.spawn()
    for _ in range(5):
        robot.command(x=10.0)
        bench.settle(robot, bench.step(25))
    assert robot.odom[-1].pose.pose.position.x == pytest.approx(WALL_X - 0.05 - 0.2, abs=0.02)
    assert abs(robot.odom[-1].twist.twist.linear.x) < 0.05
    until(lambda: bool(robot.bumps), 5.0, 'contact with the box')
    touch = robot.bumps[-1].contacts[0]
    assert (touch.collision1.name, touch.collision2.name) == ('testbot', 'wall')
    assert touch.normals[0].x == pytest.approx(1.0, abs=1e-3)


def test_kinematic_robot_passes_a_box(flatsim) -> None:
    bench = flatsim(dynamics='kinematic')
    bench.box('wall', WALL_X)
    robot = bench.spawn()
    for _ in range(5):
        robot.command(x=10.0)
        bench.settle(robot, bench.step(25))
    assert robot.odom[-1].pose.pose.position.x > WALL_X + 0.5


def test_republish_shows_the_scan_of_a_teleported_robot(flatsim) -> None:
    from flatsim_msgs.srv import SetRobotPose

    bench = flatsim()
    bench.box('wall', WALL_X)
    robot = bench.spawn()
    time.sleep(0.2)
    gap = ahead(robot.scans[-1])
    scans, transforms = len(robot.scans), len(robot.tf)
    assert bench.call('set_robot_pose', SetRobotPose.Request(name='testbot', x=1.0)).success
    assert not bench.call('set_robot_pose', SetRobotPose.Request(name='nobody', x=1.0)).success
    until(lambda: len(robot.tf) == transforms + 1, 5.0, 'transform of the teleport')
    assert robot.tf[-1].transforms[1].transform.translation.x == pytest.approx(1.0)
    time.sleep(0.2)
    assert len(robot.scans) == scans
    bench.republish()
    until(lambda: len(robot.scans) == scans + 1, 5.0, 'scan after the republish')
    assert ahead(robot.scans[-1]) == pytest.approx(gap - 1.0, abs=1e-3)
    assert robot.odom[-1].pose.pose.position.x == pytest.approx(1.0)


def test_levels_hide_the_entities_of_other_floors(flatsim) -> None:
    from flatsim_msgs.msg import Floor
    from flatsim_msgs.srv import AddFloors, DeleteEntities, SetRobotPose

    bench = flatsim()
    assert bench.call('add_floors', AddFloors.Request(floors=[Floor(name='ground', z=0.0), Floor(name='upper', z=3.0)])).success
    bench.box('upstairs', 2.0, z=(3.0, 5.0))
    bench.box('downstairs', WALL_X, z=(0.0, 2.0))
    robot = bench.spawn()
    assert ahead(robot.scans[-1]) == pytest.approx(WALL_X - 0.05 - LIDAR_X, abs=1e-3)
    assert bench.call('set_robot_pose', SetRobotPose.Request(name='testbot', z=3.0)).success
    bench.republish()
    until(lambda: ahead(robot.scans[-1]) == pytest.approx(2.0 - 0.05 - LIDAR_X, abs=1e-3), 5.0, 'scan on the upper floor')
    assert robot.odom[-1].pose.pose.position.z == pytest.approx(3.0)

    assert list(bench.call('delete_entities', DeleteEntities.Request(names=['ground', 'ground'])).removed) == [True, False]
    assert bench.call('set_robot_pose', SetRobotPose.Request(name='testbot', z=0.0)).success
    bench.republish()
    until(lambda: math.isinf(ahead(robot.scans[-1])), 5.0, 'empty scan, the box below the only floor left belongs to no level')


def test_envs_share_one_world_and_leave_by_prefix(flatsim) -> None:
    from flatsim_msgs.msg import Floor
    from flatsim_msgs.srv import AddFloors, DeletePrefix, DeleteRobot

    bench = flatsim()
    bench.box('env_0/wall', WALL_X)
    assert bench.call('add_floors', AddFloors.Request(floors=[Floor(name='env_0/floor', z=0.0), Floor(name='env_1/floor', z=0.0)])).success
    near = bench.spawn('env_0/testbot')
    far = bench.spawn('env_1/testbot', y=20.0)
    assert ahead(near.scans[-1]) == pytest.approx(WALL_X - 0.05 - LIDAR_X, abs=1e-3)
    assert math.isinf(ahead(far.scans[-1]))

    near.command(x=10.0)
    reached = bench.step(6)
    bench.settle(near, reached)
    bench.settle(far, reached)
    assert near.stamp() == pytest.approx(far.stamp(), abs=1e-9)
    assert near.odom[-1].pose.pose.position.x > 0.04
    assert far.odom[-1].pose.pose.position.x == pytest.approx(0.0, abs=1e-9)

    heard = len(far.scans)
    bench.republish('env_0/')
    until(lambda: len(near.scans) > 1 and seconds(near.scans[-1].header.stamp) == pytest.approx(reached), 5.0, 'republished scan of env_0')
    time.sleep(0.2)
    assert len(far.scans) == heard

    assert bench.call('delete_prefix', DeletePrefix.Request(prefix='env_0/')).removed == 3
    until(lambda: bench.probe.count_publishers(f'{near.namespace}/odom') == 0, 5.0, 'topics of env_0 to close')
    assert bench.call('delete_prefix', DeletePrefix.Request(prefix='env_0/')).removed == 0
    heard = len(near.odom)
    bench.settle(far, bench.step(3))
    assert len(near.odom) == heard

    assert bench.call('delete_robot', DeleteRobot.Request(name='env_1/testbot')).success
    assert not bench.call('delete_robot', DeleteRobot.Request(name='env_1/testbot')).success
    until(lambda: bench.probe.count_publishers(f'{far.namespace}/odom') == 0, 5.0, 'topics of the deleted robot to close')


def test_names_are_unique_across_entities_floors_and_robots(flatsim) -> None:
    from flatsim_msgs.msg import Floor, Outline, Ring
    from flatsim_msgs.srv import AddFloors, AddOutlines

    bench = flatsim()
    robot = bench.spawn()
    square = Ring(x=[4.0, 6.0, 6.0, 4.0], y=[-1.0, -1.0, 1.0, 1.0])
    refused = bench.call('add_outlines', AddOutlines.Request(outlines=[Outline(name='testbot', rings=[square], z_min=0.0, z_max=2.0)]))
    assert not refused.success and 'testbot' in refused.message
    assert not bench.call('add_floors', AddFloors.Request(floors=[Floor(name='testbot', z=0.0)])).success
    bench.box('door', 2.0)
    clash = robot.request()
    clash.name = 'door'
    assert not bench.call('spawn_robot', clash).success


def test_respawn_replaces_the_robot(flatsim) -> None:
    bench = flatsim()
    robot = bench.spawn()
    again = bench.call('spawn_robot', robot.request(x=2.0))
    assert again.success, again.message
    until(lambda: bench.probe.count_publishers(f'{robot.namespace}/odom') == 1, 5.0, 'one odom publisher after the respawn')

    def placed() -> bool:
        bench.republish()
        time.sleep(0.05)
        return robot.odom[-1].pose.pose.position.x == pytest.approx(2.0)

    until(placed, 5.0, 'odom of the respawned robot')

    broken = robot.request()
    broken.name = 'broken'
    broken.urdf = '<robot'
    refused = bench.call('spawn_robot', broken)
    assert not refused.success and refused.message
    bench.step(2)


def test_destroying_the_server_stops_its_stepping_thread(flatsim) -> None:
    bench = flatsim(paused=False)
    until(lambda: len(bench.clock) > 3, 5.0, 'free-running clock')
    bench.server.destroy_node()
    assert not bench.server._thread.is_alive()

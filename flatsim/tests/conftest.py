"""A flatsim server and a probe node in one process, each spun on its own thread."""

from __future__ import annotations

import math
import pathlib
import threading
import time
import typing
from collections.abc import Callable, Iterator

import pytest

DT = 1.0 / 30.0
URDF = (pathlib.Path(__file__).parent / 'testbot.urdf').read_text()
TOP_SPEED = 1.0
ACCEL = 2.0
TURN_GAIN = 1.25
TIMEOUT = 1.0


def until(pred: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() >= deadline:
            raise AssertionError(f'timed out after {timeout}s waiting for {what}')
        time.sleep(0.005)


def seconds(stamp: typing.Any) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def ahead(scan: typing.Any) -> float:
    return scan.ranges[round(-scan.angle_min / scan.angle_increment)]


class Robot:
    """Everything the probe hears from one spawned testbot."""

    def __init__(self, bench: Bench, name: str) -> None:
        import geometry_msgs.msg
        import nav_msgs.msg
        import ros_gz_interfaces.msg
        import sensor_msgs.msg
        import tf2_msgs.msg

        self.bench = bench
        self.name = name
        self.namespace = f'/{name}'
        self.tf_topic = f'{self.namespace}/tf'
        self.prefix = f'{name}/'
        self.odom: list[nav_msgs.msg.Odometry] = []
        self.tf: list[tf2_msgs.msg.TFMessage] = []
        self.joints: list[sensor_msgs.msg.JointState] = []
        self.scans: list[sensor_msgs.msg.LaserScan] = []
        self.rear: list[sensor_msgs.msg.LaserScan] = []
        self.clouds: list[sensor_msgs.msg.PointCloud2] = []
        self.depth: list[sensor_msgs.msg.PointCloud2] = []
        self.imu: list[sensor_msgs.msg.Imu] = []
        self.bumps: list[ros_gz_interfaces.msg.Contacts] = []
        probe = bench.probe
        self.subscriptions = [
            probe.create_subscription(nav_msgs.msg.Odometry, f'{self.namespace}/odom', self.odom.append, 200),
            probe.create_subscription(tf2_msgs.msg.TFMessage, self.tf_topic, self.tf.append, 200),
            probe.create_subscription(sensor_msgs.msg.JointState, f'{self.namespace}/joint_states', self.joints.append, 200),
            probe.create_subscription(sensor_msgs.msg.LaserScan, f'{self.namespace}/scan', self.scans.append, 200),
            probe.create_subscription(sensor_msgs.msg.LaserScan, f'{self.namespace}/rear_scan', self.rear.append, 200),
            probe.create_subscription(sensor_msgs.msg.PointCloud2, f'{self.namespace}/points', self.clouds.append, 200),
            probe.create_subscription(sensor_msgs.msg.PointCloud2, f'{self.namespace}/depth_points', self.depth.append, 200),
            probe.create_subscription(sensor_msgs.msg.Imu, f'{self.namespace}/imu', self.imu.append, 200),
            probe.create_subscription(ros_gz_interfaces.msg.Contacts, f'{self.namespace}/bumper', self.bumps.append, 200),
        ]
        self.cmd_vel = probe.create_publisher(geometry_msgs.msg.Twist, f'{self.namespace}/cmd_vel', 10)

    def request(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> typing.Any:
        from flatsim_msgs.msg import Axis, DriveModel, Fan, SensorSpec
        from flatsim_msgs.srv import SpawnRobot

        def sensor(name: str, kind: str, frame: str, backing: str = '', fan: list[Fan] | None = None) -> SensorSpec:
            return SensorSpec(name=name, type=kind, topic=f'{self.namespace}/{name}', frame=frame, sensor=backing, fan=fan or [])

        request = SpawnRobot.Request(name=self.name, robot_namespace=self.namespace, tf_topic=self.tf_topic, frame_prefix=self.prefix, odom_frame='odom', base_frame='base_link')
        request.x, request.y, request.z, request.yaw = x, y, z, yaw
        request.radius = 0.2
        request.drive = DriveModel(
            linear=Axis(low=-TOP_SPEED, high=TOP_SPEED, accel=ACCEL),
            lateral=Axis(low=0.0, high=0.0, accel=math.inf),
            angular=Axis(low=-2.0, high=2.0, accel=math.inf),
            gain=[1.0, 1.0, TURN_GAIN],
            lag=0.0,
            traction=math.inf,
            pivot=0.0,
            timeout=TIMEOUT,
        )
        request.urdf = URDF
        request.sensors = [
            sensor('scan', SensorSpec.LASERSCAN, 'lidar_link', 'lidar'),
            sensor('points', SensorSpec.POINTCLOUD, 'lidar_link', 'lidar'),
            sensor('depth_points', SensorSpec.POINTCLOUD, 'camera_link', 'depth'),
            sensor('imu', SensorSpec.IMU, 'imu_link', 'imu'),
            sensor('rear_scan', SensorSpec.LASERSCAN, 'rear_link', fan=[Fan(angle_min=-1.0, angle_increment=0.02, beams=101, range_max=8.0, update_rate=0.0)]),
            sensor('bumper', SensorSpec.CONTACT, 'base_link'),
        ]
        return request

    def stamp(self) -> float:
        return seconds(self.odom[-1].header.stamp) if self.odom else -math.inf

    def command(self, x: float = 0.0, wz: float = 0.0) -> None:
        """Send a twist and give the server time to take it."""
        import geometry_msgs.msg

        until(lambda: self.cmd_vel.get_subscription_count() == 1, 5.0, 'cmd_vel subscription')
        self.cmd_vel.publish(geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=x), angular=geometry_msgs.msg.Vector3(z=wz)))
        time.sleep(0.1)


class Bench:
    def __init__(self, server: typing.Any, probe: typing.Any) -> None:
        from flatsim_msgs.msg import Discs, EntityPoses
        from flatsim_msgs.srv import AddFloors, AddOutlines, ControlWorld, DeleteEntities, DeletePrefix, DeleteRobot, Republish, SetRobotPose, SpawnBoxes, SpawnRobot
        from rosgraph_msgs.msg import Clock
        import rclpy.qos

        self.server = server
        self.probe = probe
        self.clock: list[float] = []
        probe.create_subscription(Clock, '/clock', lambda msg: self.clock.append(seconds(msg.clock)), 1000)
        self.discs = probe.create_publisher(Discs, '/flatsim/discs', 100)
        self.poses = probe.create_publisher(EntityPoses, '/flatsim/poses', 100)
        services = {
            'control_world': ControlWorld,
            'add_floors': AddFloors,
            'add_outlines': AddOutlines,
            'spawn_boxes': SpawnBoxes,
            'delete_entities': DeleteEntities,
            'delete_prefix': DeletePrefix,
            'spawn_robot': SpawnRobot,
            'set_robot_pose': SetRobotPose,
            'delete_robot': DeleteRobot,
            'republish': Republish,
        }
        queue = rclpy.qos.QoSProfile(depth=100)
        self.clients = {name: probe.create_client(service, f'/flatsim/{name}', qos_profile=queue) for name, service in services.items()}
        for client in self.clients.values():
            assert client.wait_for_service(timeout_sec=10.0), client.srv_name

    def call(self, service: str, request: typing.Any) -> typing.Any:
        future = self.clients[service].call_async(request)
        until(future.done, 10.0, f'{service} response')
        return future.result()

    def control(self, pause: bool, steps: int = 0) -> typing.Any:
        from flatsim_msgs.srv import ControlWorld

        return self.call('control_world', ControlWorld.Request(pause=pause, multi_step=steps))

    def step(self, steps: int = 1) -> float:
        response = self.control(True, steps)
        assert response.success
        return seconds(response.time)

    def republish(self, prefix: str = '') -> None:
        from flatsim_msgs.srv import Republish

        assert self.call('republish', Republish.Request(prefix=prefix)).success

    def box(self, name: str, x: float, y: float = 0.0, size: tuple[float, float] = (0.1, 6.0), z: tuple[float, float] = (0.0, 2.0)) -> None:
        from flatsim_msgs.msg import Box
        from flatsim_msgs.srv import SpawnBoxes

        box = Box(name=name, size_x=size[0], size_y=size[1], x=x, y=y, yaw=0.0, z_min=z[0], z_max=z[1])
        assert self.call('spawn_boxes', SpawnBoxes.Request(boxes=[box])).success

    def spawn(self, name: str = 'testbot', **pose: float) -> Robot:
        """Spawn a testbot and wait until the probe hears all of its topics."""
        robot = Robot(self, name)
        response = self.call('spawn_robot', robot.request(**pose))
        assert response.success, response.message
        deadline = time.monotonic() + 10.0
        while not (robot.odom and robot.tf and robot.joints and robot.scans and robot.rear and robot.clouds and robot.depth and robot.imu):
            assert time.monotonic() < deadline, 'testbot topics did not all arrive'
            self.republish(name)
            time.sleep(0.1)
        return robot

    def settle(self, robot: Robot, time_reached: float) -> None:
        until(lambda: robot.stamp() >= time_reached - 1e-6, 5.0, f'odom at {time_reached:.3f}')
        time.sleep(0.05)


@pytest.fixture()
def flatsim() -> Iterator[Callable[..., Bench]]:
    """Factory of a paused server (or one with the given parameters) and its probe."""
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('flatsim_msgs')
    import rclpy.executors
    import rclpy.node
    import rclpy.parameter

    from flatsim.server import FlatsimServer

    rclpy.init()
    executors: list[rclpy.executors.SingleThreadedExecutor] = []
    threads: list[threading.Thread] = []
    nodes: list[rclpy.node.Node] = []

    def spin(node: rclpy.node.Node) -> None:
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        executors.append(executor)
        threads.append(thread)
        nodes.append(node)

    def start(**parameters: object) -> Bench:
        overrides = {'physics_dt': DT, 'paused': True, **parameters}
        server = FlatsimServer(parameter_overrides=[rclpy.parameter.Parameter(name, value=value) for name, value in overrides.items()])
        probe = rclpy.node.Node('flatsim_probe')
        spin(server)
        spin(probe)
        return Bench(server, probe)

    try:
        yield start
    finally:
        for executor in executors:
            executor.shutdown()
        for thread in threads:
            thread.join(timeout=5.0)
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()

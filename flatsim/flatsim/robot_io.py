"""ROS side of one robot: its topics, sensor schedule and messages."""

from __future__ import annotations

import array
import math
import typing
from collections.abc import Callable

import attrs
import builtin_interfaces.msg
import geometry_msgs.msg
import nav_msgs.msg
import numpy as np
import ros_gz_interfaces.msg
import sensor_msgs.msg
import tf2_msgs.msg

from flatsim.drive import Axis, DriveModel
from flatsim.sensors import camera_cloud, camera_grid, camera_reach, cast_fan, fan, imu_acceleration, lidar_cloud, ring_slopes
from flatsim.urdf import frame_inverted, frame_offset, movable_joints, sensor_block
from flatsim.world import FlatWorld, RobotState, ScanSpec, Touch, Twist2D
from flatsim_msgs.msg import DriveModel as DriveModelMsg
from flatsim_msgs.msg import SensorSpec

if typing.TYPE_CHECKING:
    import rclpy.node
    import rclpy.publisher
    import rclpy.subscription

    from flatsim_msgs.srv import SpawnRobot

DEFAULT_TF_TOPIC = '/tf'
DEFAULT_WORLD_FRAME = 'map'

_DUE_EPS = 1e-4
_QUEUE = 10
_TF_QUEUE = 100
_POINT_FIELDS = {'<f4': sensor_msgs.msg.PointField.FLOAT32, '<u2': sensor_msgs.msg.PointField.UINT16}


@attrs.define
class Rate:
    """Publish schedule locked to whole periods of sim time."""

    period: float
    last_stamp: float = -math.inf

    def due(self, now: float) -> bool:
        if not math.isfinite(self.last_stamp) or now < self.last_stamp:
            return True
        return math.floor(now / self.period + _DUE_EPS) > math.floor(self.last_stamp / self.period + _DUE_EPS)


def period(update_rate: float | None, dt: float) -> float:
    """Sensor publish period in whole physics steps, one step without an update rate."""
    return max(1, round(1.0 / update_rate / dt)) * dt if update_rate else dt


def drive_model(msg: DriveModelMsg) -> DriveModel:
    return DriveModel(
        linear=Axis(msg.linear.low, msg.linear.high, msg.linear.accel),
        lateral=Axis(msg.lateral.low, msg.lateral.high, msg.lateral.accel),
        angular=Axis(msg.angular.low, msg.angular.high, msg.angular.accel),
        gain=(msg.gain[0], msg.gain[1], msg.gain[2]),
        lag=msg.lag,
        traction=msg.traction,
        pivot=msg.pivot,
        timeout=msg.timeout,
    )


@attrs.define
class _Output:
    publisher: rclpy.publisher.Publisher
    frame_id: str


@attrs.define
class _Rays:
    """One backing lidar or camera, a single cast per update feeds its scan and its cloud."""

    spec: ScanSpec
    azimuth: np.ndarray
    slopes: np.ndarray
    up: float
    near: float
    far: float
    camera: bool
    rate: Rate
    scan: _Output | None = None
    cloud: _Output | None = None


@attrs.define
class _Imu:
    output: _Output
    mount_yaw: float
    rate: Rate


@attrs.define
class _Bumper:
    output: _Output
    touching: bool = False


def _yaw_quaternion(yaw: float) -> geometry_msgs.msg.Quaternion:
    return geometry_msgs.msg.Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def _laser_scan(rays: _Rays, ranges: np.ndarray, frame_id: str, stamp: builtin_interfaces.msg.Time) -> sensor_msgs.msg.LaserScan:
    scan = sensor_msgs.msg.LaserScan()
    scan.header.stamp = stamp
    scan.header.frame_id = frame_id
    scan.angle_min = float(rays.azimuth[0])
    scan.angle_max = float(rays.azimuth[-1])
    scan.angle_increment = rays.up * rays.spec.angle_increment
    scan.time_increment = 0.0
    scan.scan_time = rays.rate.period
    scan.range_min = rays.near
    scan.range_max = rays.far
    scan.ranges = ranges.astype(np.float32).tolist()
    return scan


def _point_cloud(points: np.ndarray, frame_id: str, stamp: builtin_interfaces.msg.Time) -> sensor_msgs.msg.PointCloud2:
    """Unorganized cloud whose fields mirror the structured dtype of `points`."""
    cloud = sensor_msgs.msg.PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [sensor_msgs.msg.PointField(name=name, offset=offset, datatype=_POINT_FIELDS[kind.str], count=1) for name, (kind, offset) in points.dtype.fields.items()]
    cloud.is_bigendian = False
    cloud.point_step = points.dtype.itemsize
    cloud.row_step = points.dtype.itemsize * len(points)
    cloud.data = array.array('B', points.tobytes())
    cloud.is_dense = True
    return cloud


def _imu_reading(imu: _Imu, state: RobotState, stamp: builtin_interfaces.msg.Time) -> sensor_msgs.msg.Imu:
    reading = sensor_msgs.msg.Imu()
    reading.header.stamp = stamp
    reading.header.frame_id = imu.output.frame_id
    reading.orientation = _yaw_quaternion(state.yaw + imu.mount_yaw)
    reading.angular_velocity.z = state.wz
    x, y, z = imu_acceleration(state.ax, state.ay, imu.mount_yaw)
    reading.linear_acceleration = geometry_msgs.msg.Vector3(x=x, y=y, z=z)
    return reading


def _contacts(name: str, touches: list[Touch], z: float, frame_id: str, stamp: builtin_interfaces.msg.Time) -> ros_gz_interfaces.msg.Contacts:
    contacts = ros_gz_interfaces.msg.Contacts()
    contacts.header.stamp = stamp
    contacts.header.frame_id = frame_id
    for other, point, normal in touches:
        contact = ros_gz_interfaces.msg.Contact()
        contact.collision1.name = name
        contact.collision2.name = other
        contact.positions = [geometry_msgs.msg.Vector3(x=point[0], y=point[1], z=z)]
        contact.normals = [geometry_msgs.msg.Vector3(x=normal[0], y=normal[1])]
        contacts.contacts.append(contact)
    return contacts


@attrs.define
class RobotIO:
    """Topics of one robot in a world, `name` being its key there."""

    name: str
    drive: DriveModel
    world_frame: str
    odom_frame: str
    base_frame: str
    joints: list[str]
    tf: rclpy.publisher.Publisher
    odom: rclpy.publisher.Publisher
    joint_states: rclpy.publisher.Publisher
    cmd_vel: rclpy.subscription.Subscription
    rays: list[_Rays]
    imus: list[_Imu]
    bumpers: list[_Bumper]

    def _transforms(self, state: RobotState, stamp: builtin_interfaces.msg.Time) -> tf2_msgs.msg.TFMessage:
        world_to_odom = geometry_msgs.msg.TransformStamped()
        world_to_odom.header.stamp = stamp
        world_to_odom.header.frame_id = self.world_frame
        world_to_odom.child_frame_id = self.odom_frame
        world_to_odom.transform.rotation.w = 1.0
        odom_to_base = geometry_msgs.msg.TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = self.odom_frame
        odom_to_base.child_frame_id = self.base_frame
        odom_to_base.transform.translation.x = state.x
        odom_to_base.transform.translation.y = state.y
        odom_to_base.transform.translation.z = state.z
        odom_to_base.transform.rotation = _yaw_quaternion(state.yaw)
        return tf2_msgs.msg.TFMessage(transforms=[world_to_odom, odom_to_base])

    def _odometry(self, state: RobotState, stamp: builtin_interfaces.msg.Time) -> nav_msgs.msg.Odometry:
        odom = nav_msgs.msg.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = state.x
        odom.pose.pose.position.y = state.y
        odom.pose.pose.position.z = state.z
        odom.pose.pose.orientation = _yaw_quaternion(state.yaw)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = self.drive.encoded((state.vx, state.vy, state.wz))
        return odom

    def _joint_state(self, stamp: builtin_interfaces.msg.Time) -> sensor_msgs.msg.JointState:
        joint_state = sensor_msgs.msg.JointState(name=self.joints, position=[0.0] * len(self.joints))
        joint_state.header.stamp = stamp
        return joint_state

    def _publish_rays(self, world: FlatWorld, rays: _Rays, stamp: builtin_interfaces.msg.Time) -> None:
        hits = world.cast(self.name, rays.spec)
        if rays.scan is not None:
            rays.scan.publisher.publish(_laser_scan(rays, hits.first(), rays.scan.frame_id, stamp))
        if rays.cloud is not None:
            points = (camera_cloud if rays.camera else lidar_cloud)(hits.first(rays.up * rays.slopes), rays.azimuth, rays.slopes, rays.near, rays.far)
            rays.cloud.publisher.publish(_point_cloud(points, rays.cloud.frame_id, stamp))

    def publish_transforms(self, world: FlatWorld, stamp: builtin_interfaces.msg.Time) -> None:
        self.tf.publish(self._transforms(world.robot_state(self.name), stamp))

    def publish(self, world: FlatWorld, stamp: builtin_interfaces.msg.Time, every: bool) -> None:
        """Publish the robot's state and its due sensors (all of them with `every`) at the world's time."""
        now = world.time
        state = world.robot_state(self.name)
        self.tf.publish(self._transforms(state, stamp))
        self.odom.publish(self._odometry(state, stamp))
        self.joint_states.publish(self._joint_state(stamp))
        for rays in self.rays:
            if every or rays.rate.due(now):
                rays.rate.last_stamp = now
                self._publish_rays(world, rays, stamp)
        for imu in self.imus:
            if every or imu.rate.due(now):
                imu.rate.last_stamp = now
                imu.output.publisher.publish(_imu_reading(imu, state, stamp))
        if self.bumpers:
            touches = world.contacts(self.name)
            for bumper in self.bumpers:
                if touches or bumper.touching:
                    bumper.output.publisher.publish(_contacts(self.name, touches, state.z, bumper.output.frame_id, stamp))
                bumper.touching = bool(touches)

    def close(self, node: rclpy.node.Node) -> None:
        node.destroy_subscription(self.cmd_vel)
        outputs = [output for rays in self.rays for output in (rays.scan, rays.cloud) if output is not None] + [sensor.output for sensor in (*self.imus, *self.bumpers)]
        for publisher in (self.tf, self.odom, self.joint_states, *(output.publisher for output in outputs)):
            node.destroy_publisher(publisher)


@attrs.define
class _Opener:
    node: rclpy.node.Node
    request: SpawnRobot.Request
    dt: float
    opened: list[rclpy.publisher.Publisher] = attrs.field(factory=list)

    def _warn(self, text: str) -> None:
        self.node.get_logger().warning(f'robot {self.request.name!r}: {text}')

    def publisher(self, msg_type: type, topic: str, depth: int = _QUEUE) -> rclpy.publisher.Publisher:
        publisher = self.node.create_publisher(msg_type, topic, depth)
        self.opened.append(publisher)
        return publisher

    def _output(self, sensor: SensorSpec, msg_type: type) -> _Output:
        return _Output(self.publisher(msg_type, sensor.topic), self.request.frame_prefix + sensor.frame)

    def _mount(self, sensors: list[SensorSpec], frame: str) -> tuple[float, float, float, float, bool] | None:
        """Pose (x, y, z, yaw) of a sensor frame in the base frame and whether it hangs upside down, None for a frame the URDF lacks."""
        try:
            return *frame_offset(self.request.urdf, self.request.base_frame, frame), frame_inverted(self.request.urdf, self.request.base_frame, frame)
        except ValueError as e:
            self._warn(f'{e}, sensors {[sensor.name for sensor in sensors]} are not simulated')
            return None

    def _rate(self, update_rate: float | None) -> Rate:
        return Rate(period=period(update_rate, self.dt))

    def rays(self, backing: str, frame: str, specs: dict[str, SensorSpec]) -> _Rays | None:
        """Ray sensor behind the laserscan and pointcloud specs of one backing URDF sensor, from the laserscan's fan when the URDF has none."""
        laserscan = specs.get(SensorSpec.LASERSCAN)
        pointcloud = specs.get(SensorSpec.POINTCLOUD)
        block = sensor_block(self.request.urdf, backing, frame)
        if block is not None and block.lidar is not None:
            lidar = block.lidar
            azimuth = fan(lidar.samples, lidar.min_angle, lidar.max_angle)
            slopes = ring_slopes(lidar.vertical_samples, lidar.vertical_min_angle, lidar.vertical_max_angle)
            near, far, reach, rate = lidar.range_min, lidar.range_max, lidar.range_max, block.update_rate
        elif block is not None and block.camera is not None and pointcloud is not None:
            camera = block.camera
            azimuth, slopes = camera_grid(camera.horizontal_fov, camera.width, camera.height)
            near, far, reach, rate = camera.near, camera.far, camera_reach(camera.horizontal_fov, camera.far), block.update_rate
            laserscan = None
        elif block is None and laserscan is not None and laserscan.fan:
            beams = laserscan.fan[0]
            azimuth = beams.angle_min + beams.angle_increment * np.arange(beams.beams)
            slopes = ring_slopes(1, 0.0, 0.0)
            near, far, reach, rate = 0.0, beams.range_max, beams.range_max, beams.update_rate or None
        else:
            self._warn(f'no lidar or camera sensor {backing!r} in the URDF to back {[sensor.name for sensor in specs.values()]}, not simulated')
            return None
        mount = self._mount(list(specs.values()), frame)
        if mount is None:
            return None
        x, y, z, yaw, inverted = mount
        angle_min, angle_increment = cast_fan(azimuth, inverted)
        return _Rays(
            spec=ScanSpec(mount=(x, y, yaw), angle_min=angle_min, angle_increment=angle_increment, beams=len(azimuth), range_min=near, range_max=reach, height=z),
            azimuth=azimuth,
            slopes=slopes,
            up=-1.0 if inverted else 1.0,
            near=near,
            far=far,
            camera=block is not None and block.camera is not None,
            rate=self._rate(rate),
            scan=self._output(laserscan, sensor_msgs.msg.LaserScan) if laserscan is not None else None,
            cloud=self._output(pointcloud, sensor_msgs.msg.PointCloud2) if pointcloud is not None else None,
        )

    def imu(self, sensor: SensorSpec) -> _Imu | None:
        mount = self._mount([sensor], sensor.frame)
        if mount is None:
            return None
        block = sensor_block(self.request.urdf, sensor.sensor, sensor.frame) if sensor.sensor else None
        return _Imu(self._output(sensor, sensor_msgs.msg.Imu), mount[3], self._rate(block.update_rate if block is not None else None))

    def bumper(self, sensor: SensorSpec) -> _Bumper:
        return _Bumper(self._output(sensor, ros_gz_interfaces.msg.Contacts))


def open_robot(node: rclpy.node.Node, request: SpawnRobot.Request, dt: float, on_command: Callable[[Twist2D], None]) -> RobotIO:
    """Create the topics of a spawn request, sensors the URDF cannot back are skipped with a warning."""
    opener = _Opener(node, request, dt)
    namespace = request.robot_namespace.rstrip('/')
    cmd_vel = node.create_subscription(geometry_msgs.msg.Twist, f'{namespace}/cmd_vel', lambda msg: on_command((msg.linear.x, msg.linear.y, msg.angular.z)), _QUEUE)
    try:
        ranged: dict[tuple[str, str], dict[str, SensorSpec]] = {}
        imus: list[_Imu] = []
        bumpers: list[_Bumper] = []
        for sensor in request.sensors:
            if sensor.type in (SensorSpec.LASERSCAN, SensorSpec.POINTCLOUD):
                ranged.setdefault((sensor.sensor or sensor.name, sensor.frame), {})[sensor.type] = sensor
            elif sensor.type == SensorSpec.IMU:
                imu = opener.imu(sensor)
                if imu is not None:
                    imus.append(imu)
            elif sensor.type == SensorSpec.CONTACT:
                bumpers.append(opener.bumper(sensor))
        rays = [opener.rays(backing, frame, specs) for (backing, frame), specs in ranged.items()]
        return RobotIO(
            name=request.name,
            drive=drive_model(request.drive),
            world_frame=request.world_frame or DEFAULT_WORLD_FRAME,
            odom_frame=request.frame_prefix + request.odom_frame,
            base_frame=request.frame_prefix + request.base_frame,
            joints=movable_joints(request.urdf) if request.urdf else [],
            tf=opener.publisher(tf2_msgs.msg.TFMessage, request.tf_topic or DEFAULT_TF_TOPIC, _TF_QUEUE),
            odom=opener.publisher(nav_msgs.msg.Odometry, f'{namespace}/odom'),
            joint_states=opener.publisher(sensor_msgs.msg.JointState, f'{namespace}/joint_states'),
            cmd_vel=cmd_vel,
            rays=[opened for opened in rays if opened is not None],
            imus=imus,
            bumpers=bumpers,
        )
    except Exception:
        node.destroy_subscription(cmd_vel)
        for publisher in opener.opened:
            node.destroy_publisher(publisher)
        raise

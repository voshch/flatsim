"""flatsim_server: owns sim time and the world, behind the services and topics of flatsim_msgs."""

from __future__ import annotations

import contextlib
import signal
import threading
import time
import traceback
import typing
from collections.abc import Iterator

import builtin_interfaces.msg
import rcl_interfaces.msg
import rclpy
import rclpy.executors
import rclpy.node
import rclpy.parameter
import rclpy.qos
import rosgraph_msgs.msg

from flatsim.scene import Scene
from flatsim.world import Dynamics, Twist2D
from flatsim_msgs.msg import Discs, EntityPoses
from flatsim_msgs.srv import AddFloors, AddOutlines, ControlWorld, DeleteEntities, DeletePrefix, DeleteRobot, Republish, SetRobotPose, SpawnBoxes, SpawnRobot

NODE_NAME = 'flatsim'
DEFAULT_DT = 0.0333
IDLE_CLOCK_PERIOD = 0.01

_TOPIC_QUEUE = 100
_SERVICE_QOS = rclpy.qos.QoSProfile(depth=100)
_NANO = 1_000_000_000


def stamp_of(seconds: float) -> builtin_interfaces.msg.Time:
    nanoseconds = round(seconds * _NANO)
    return builtin_interfaces.msg.Time(sec=nanoseconds // _NANO, nanosec=nanoseconds % _NANO)


class _Turns:
    """World lock the stepping loop yields to whenever a callback waits for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callers = 0
        self._quiet = threading.Condition()

    @contextlib.contextmanager
    def callback(self) -> Iterator[None]:
        with self._quiet:
            self._callers += 1
        try:
            with self._lock:
                yield
        finally:
            with self._quiet:
                self._callers -= 1
                self._quiet.notify_all()

    @contextlib.contextmanager
    def loop(self) -> Iterator[None]:
        with self._quiet:
            self._quiet.wait_for(lambda: not self._callers)
        with self._lock:
            yield


class _Pace:
    """Wall-clock deadline of the next free-running step."""

    def __init__(self) -> None:
        self._deadline = time.monotonic()

    def reset(self) -> None:
        self._deadline = time.monotonic()

    def wait(self, dt: float, real_time_factor: float) -> float:
        """Seconds to sleep before the next step, a late loop catches up by one step at most."""
        now = time.monotonic()
        if real_time_factor <= 0.0:
            self._deadline = now
            return 0.0
        step = dt / real_time_factor
        self._deadline = max(self._deadline + step, now - step)
        return max(0.0, self._deadline - now)


class FlatsimServer(rclpy.node.Node):
    """Steps the world on its clock: free-running at `real_time_factor` when unpaused, on request when paused."""

    def __init__(self, *, parameter_overrides: list[rclpy.parameter.Parameter] | None = None) -> None:
        super().__init__(NODE_NAME, parameter_overrides=parameter_overrides)
        fixed = rcl_interfaces.msg.ParameterDescriptor(read_only=True)
        self._dt = float(typing.cast(float, self.declare_parameter('physics_dt', DEFAULT_DT, fixed).value))
        if self._dt <= 0.0:
            raise ValueError(f'physics_dt must be positive, got {self._dt}')
        dynamics = str(self.declare_parameter('dynamics', Dynamics.RIGID.value, fixed).value)
        try:
            self._scene = Scene(self._dt, Dynamics(dynamics))
        except ValueError as e:
            raise ValueError(f'dynamics must be one of {[choice.value for choice in Dynamics]}, got {dynamics!r}') from e
        self._paused = bool(self.declare_parameter('paused', False, fixed).value)
        self._real_time_factor = float(typing.cast(float, self.declare_parameter('real_time_factor', 1.0).value))
        self.add_on_set_parameters_callback(self._on_parameters)

        self._ticks = 0
        self._turns = _Turns()
        self._stop = threading.Event()
        self._clock_publisher = self.create_publisher(rosgraph_msgs.msg.Clock, '/clock', rclpy.qos.QoSProfile(depth=10))

        self.create_service(ControlWorld, '~/control_world', self._control_world, qos_profile=_SERVICE_QOS)
        self.create_service(AddFloors, '~/add_floors', self._add_floors, qos_profile=_SERVICE_QOS)
        self.create_service(AddOutlines, '~/add_outlines', self._add_outlines, qos_profile=_SERVICE_QOS)
        self.create_service(SpawnBoxes, '~/spawn_boxes', self._spawn_boxes, qos_profile=_SERVICE_QOS)
        self.create_service(DeleteEntities, '~/delete_entities', self._delete_entities, qos_profile=_SERVICE_QOS)
        self.create_service(DeletePrefix, '~/delete_prefix', self._delete_prefix, qos_profile=_SERVICE_QOS)
        self.create_service(SpawnRobot, '~/spawn_robot', self._spawn_robot, qos_profile=_SERVICE_QOS)
        self.create_service(SetRobotPose, '~/set_robot_pose', self._set_robot_pose, qos_profile=_SERVICE_QOS)
        self.create_service(DeleteRobot, '~/delete_robot', self._delete_robot, qos_profile=_SERVICE_QOS)
        self.create_service(Republish, '~/republish', self._republish, qos_profile=_SERVICE_QOS)
        self.create_subscription(Discs, '~/discs', self._on_discs, rclpy.qos.QoSProfile(depth=_TOPIC_QUEUE))
        self.create_subscription(EntityPoses, '~/poses', self._on_poses, rclpy.qos.QoSProfile(depth=_TOPIC_QUEUE))

        self._thread = threading.Thread(target=self._run, name='flatsim_steps', daemon=True)
        self._thread.start()
        self.get_logger().info(f'flatsim up: dt {self._dt}, {dynamics}, real_time_factor {self._real_time_factor:g}, {"paused" if self._paused else "running"}')

    @property
    def now(self) -> float:
        return self._ticks * self._dt

    def _on_parameters(self, parameters: list[rclpy.parameter.Parameter]) -> rcl_interfaces.msg.SetParametersResult:
        for parameter in parameters:
            if parameter.name == 'real_time_factor':
                self._real_time_factor = float(typing.cast(float, parameter.value))
        return rcl_interfaces.msg.SetParametersResult(successful=True)

    def _advance(self, steps: int) -> None:
        """Take `steps` physics steps, each followed by its clock message, the caller holds the world lock."""
        for _ in range(steps):
            self._ticks += 1
            now = self.now
            stamp = stamp_of(now)
            self._scene.advance(now, stamp)
            self._clock_publisher.publish(rosgraph_msgs.msg.Clock(clock=stamp))

    def _run(self) -> None:
        pace = _Pace()
        while not self._stop.is_set():
            try:
                with self._turns.loop():
                    running = not self._paused
                    if running:
                        self._advance(1)
                    else:
                        self._clock_publisher.publish(rosgraph_msgs.msg.Clock(clock=stamp_of(self.now)))
            except Exception:
                if self._stop.is_set():
                    return
                self.get_logger().error(f'step failed, pausing\n{traceback.format_exc()}')
                self._paused = True
                continue
            if running:
                delay = pace.wait(self._dt, self._real_time_factor)
                if delay > 0.0:
                    self._stop.wait(delay)
            else:
                pace.reset()
                self._stop.wait(IDLE_CLOCK_PERIOD)

    def _control_world(self, request: ControlWorld.Request, response: ControlWorld.Response) -> ControlWorld.Response:
        with self._turns.callback():
            response.success = request.pause or not request.multi_step
            if response.success:
                self._paused = request.pause
                self._advance(request.multi_step)
                if not request.multi_step:
                    self._clock_publisher.publish(rosgraph_msgs.msg.Clock(clock=stamp_of(self.now)))
            response.time = stamp_of(self.now)
        return response

    def _add_floors(self, request: AddFloors.Request, response: AddFloors.Response) -> AddFloors.Response:
        with self._turns.callback():
            try:
                self._scene.add_floors(request.floors)
                response.success = True
            except ValueError as e:
                response.message = str(e)
        return response

    def _add_outlines(self, request: AddOutlines.Request, response: AddOutlines.Response) -> AddOutlines.Response:
        with self._turns.callback():
            try:
                self._scene.add_outlines(request.outlines)
                response.success = True
            except ValueError as e:
                response.message = str(e)
        return response

    def _spawn_boxes(self, request: SpawnBoxes.Request, response: SpawnBoxes.Response) -> SpawnBoxes.Response:
        with self._turns.callback():
            try:
                self._scene.spawn_boxes(request.boxes)
                response.success = True
            except ValueError as e:
                response.message = str(e)
        return response

    def _delete_entities(self, request: DeleteEntities.Request, response: DeleteEntities.Response) -> DeleteEntities.Response:
        with self._turns.callback():
            response.removed = self._scene.delete(request.names)
        return response

    def _delete_prefix(self, request: DeletePrefix.Request, response: DeletePrefix.Response) -> DeletePrefix.Response:
        with self._turns.callback():
            response.removed = self._scene.delete_prefix(self, request.prefix)
        return response

    def _on_command(self, robot: str, twist: Twist2D) -> None:
        with self._turns.callback():
            self._scene.command(robot, twist)

    def _spawn_robot(self, request: SpawnRobot.Request, response: SpawnRobot.Response) -> SpawnRobot.Response:
        robot = request.name
        with self._turns.callback():
            try:
                self._scene.spawn_robot(self, request, lambda twist: self._on_command(robot, twist))
                response.success = True
            except Exception as e:
                response.message = repr(e)
                self.get_logger().error(f'spawn of robot {robot!r} failed\n{traceback.format_exc()}')
        return response

    def _set_robot_pose(self, request: SetRobotPose.Request, response: SetRobotPose.Response) -> SetRobotPose.Response:
        with self._turns.callback():
            response.success = self._scene.set_robot_pose(request.name, (request.x, request.y, request.yaw), request.z, stamp_of(self.now))
        return response

    def _delete_robot(self, request: DeleteRobot.Request, response: DeleteRobot.Response) -> DeleteRobot.Response:
        with self._turns.callback():
            response.success = self._scene.delete_robot(self, request.name)
        return response

    def _republish(self, request: Republish.Request, response: Republish.Response) -> Republish.Response:
        with self._turns.callback():
            self._scene.publish(stamp_of(self.now), every=True, prefix=request.prefix)
            response.success = True
        return response

    def _on_discs(self, msg: Discs) -> None:
        with self._turns.callback():
            self._scene.update_discs(msg)

    def _on_poses(self, msg: EntityPoses) -> None:
        with self._turns.callback():
            self._scene.move(msg.poses)

    def destroy_node(self) -> None:
        self._stop.set()
        self._thread.join()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = FlatsimServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

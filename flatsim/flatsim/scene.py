"""The one world of the server with the topics of its robots."""

from __future__ import annotations

import typing
from collections.abc import Callable, Collection

from flatsim.robot_io import RobotIO, open_robot
from flatsim.world import Dynamics, FlatWorld, Twist2D

if typing.TYPE_CHECKING:
    import builtin_interfaces.msg
    import rclpy.node

    from flatsim_msgs.msg import Box, Discs, EntityPose, Floor, Outline
    from flatsim_msgs.srv import SpawnRobot


class Scene:
    """Entities, floors and robots of the world, every name unique across them."""

    def __init__(self, dt: float, dynamics: Dynamics) -> None:
        self.world = FlatWorld(dt, dynamics)
        self.robots: dict[str, RobotIO] = {}
        self._floors: dict[str, float] = {}

    def _entity(self, name: str) -> bool:
        return name in self.world and name not in self.robots

    def _free(self, names: list[str]) -> None:
        taken = [name for name in names if name in self.robots]
        if taken:
            raise ValueError(f'{taken} are robots')

    def add_floors(self, floors: Collection[Floor]) -> None:
        self._free([floor.name for floor in floors])
        levels = {**self._floors, **{floor.name: floor.z for floor in floors}}
        self.world.set_levels(levels.values())
        self._floors = levels

    def add_outlines(self, outlines: Collection[Outline]) -> None:
        self._free([outline.name for outline in outlines])
        for outline in outlines:
            self.world.add_outline(outline.name, [list(zip(ring.x, ring.y, strict=True)) for ring in outline.rings], (outline.z_min, outline.z_max))

    def spawn_boxes(self, boxes: Collection[Box]) -> None:
        self._free([box.name for box in boxes])
        for box in boxes:
            self.world.add_box(box.name, (box.size_x, box.size_y), (box.x, box.y, box.yaw), (box.z_min, box.z_max))

    def move(self, poses: Collection[EntityPose]) -> None:
        for pose in poses:
            if self._entity(pose.name):
                self.world.move(pose.name, (pose.x, pose.y, pose.yaw), z=(pose.z_min, pose.z_max))

    def delete(self, names: Collection[str]) -> list[bool]:
        removed: list[bool] = []
        floors = len(self._floors)
        for name in names:
            if name in self._floors:
                del self._floors[name]
                removed.append(True)
            else:
                removed.append(self._entity(name) and self.world.remove(name))
        if len(self._floors) != floors:
            self.world.set_levels(self._floors.values())
        return removed

    def delete_prefix(self, node: rclpy.node.Node, prefix: str) -> int:
        """Remove every entity, floor and robot whose name starts with `prefix`, returns how many."""
        robots = [name for name in self.robots if name.startswith(prefix)]
        for name in robots:
            self.delete_robot(node, name)
        entities = [name for name in self.world.names if name.startswith(prefix)]
        floors = [name for name in self._floors if name.startswith(prefix)]
        return len(robots) + sum(self.delete([*entities, *floors]))

    def update_discs(self, discs: Discs) -> None:
        for disc in discs.discs:
            if disc.name in self.robots:
                continue
            extent = (disc.z_min, disc.z_max)
            if disc.name not in self.world:
                self.world.add_disc(disc.name, disc.radius, (disc.x, disc.y), extent)
            self.world.move(disc.name, (disc.x, disc.y, 0.0), velocity=(disc.vx, disc.vy), z=extent)
        self.delete(discs.removed)

    def spawn_robot(self, node: rclpy.node.Node, request: SpawnRobot.Request, on_command: Callable[[Twist2D], None]) -> None:
        """Replace robot `request.name` by a body at the requested pose with fresh topics."""
        if self._entity(request.name) or request.name in self._floors:
            raise ValueError(f'{request.name!r} is not a robot')
        self.delete_robot(node, request.name)
        io = open_robot(node, request, self.world.dt, on_command)
        try:
            footprint = list(zip(request.footprint.x, request.footprint.y, strict=True))
            self.world.add_robot(io.name, (request.x, request.y, request.yaw), request.z, io.drive, footprint or None, request.radius)
        except Exception:
            io.close(node)
            raise
        self.robots[io.name] = io

    def command(self, name: str, twist: Twist2D) -> None:
        if name in self.robots:
            self.world.command(name, twist)

    def set_robot_pose(self, name: str, pose: tuple[float, float, float], z: float, stamp: builtin_interfaces.msg.Time) -> bool:
        io = self.robots.get(name)
        if io is None:
            return False
        self.world.set_robot_pose(name, pose, z)
        io.publish_transforms(self.world, stamp)
        return True

    def delete_robot(self, node: rclpy.node.Node, name: str) -> bool:
        io = self.robots.pop(name, None)
        if io is None:
            return False
        self.world.remove(name)
        io.close(node)
        return True

    def advance(self, now: float, stamp: builtin_interfaces.msg.Time) -> None:
        """Step the world to `now` and publish what is due."""
        if self.world.advance(now):
            self.publish(stamp, every=False)

    def publish(self, stamp: builtin_interfaces.msg.Time, every: bool, prefix: str = '') -> None:
        for name, io in self.robots.items():
            if name.startswith(prefix):
                io.publish(self.world, stamp, every)

    def close(self, node: rclpy.node.Node) -> None:
        for io in self.robots.values():
            io.close(node)
        self.robots.clear()

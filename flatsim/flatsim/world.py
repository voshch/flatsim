"""ROS-free 2D world of bodies, outlines, discs and levels, the only module that imports pyb2d3."""

from __future__ import annotations

import bisect
import enum
import math
from collections.abc import Iterable, Sequence

import attrs
import numpy as np
import pyb2d3 as b2

from flatsim.drive import DriveModel

Point = tuple[float, float]
Pose2D = tuple[float, float, float]
Twist2D = tuple[float, float, float]
Touch = tuple[str, Point, Point]

ROBOT_HEIGHT = 0.5
GROUND_SLAB = 0.05

_WALL = 1
_DISC = 2
_ROBOT = 4
_QUERY = 8
_LEVEL_EPS = 1e-3
_STEP_EPS = 1e-4
_LINEAR_SLOP = 0.005  # Box2D returns an invalid shape for a segment no longer than this
_SUBSTEPS = 4
_MAX_HULL = 8
_CONTACT_MARGIN = 0.01
_ARC_EPS = 1e-9
_BOX_EDGES = [1, 2, 3, 0]


class Dynamics(enum.Enum):
    RIGID = 'rigid'
    KINEMATIC = 'kinematic'


@attrs.frozen
class ScanSpec:
    """One planar ray fan, mounted at (x, y, yaw) in the robot base frame and `height` above its z."""

    mount: Pose2D
    angle_min: float
    angle_increment: float
    beams: int
    range_min: float
    range_max: float
    height: float = 0.0


@attrs.frozen
class RayHits:
    """Entities each beam of a fan crosses, nearest first: (beams, k) horizontal distances and z extents relative to the sensor, padded with inf, and the floor z relative to the sensor."""

    distance: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    floor: float = -math.inf

    def take(self, beam: np.ndarray) -> RayHits:
        """Hits of the beams `beam` indexes, in its shape."""
        return RayHits(self.distance[beam], self.lo[beam], self.hi[beam], self.floor)

    def first(self, slope: np.ndarray | float = 0.0) -> np.ndarray:
        """Horizontal distance to the first entity whose z extent holds each beam at `slope` (rise per meter, broadcast over beams), or to the floor for a downward beam, inf for none."""
        slope = np.asarray(slope, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            rise = self.distance * slope[..., None]
        seen = (self.lo <= rise) & (rise <= self.hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            ground = np.where((slope < 0.0) & (self.floor < 0.0), self.floor / slope, np.inf)
        return np.minimum(np.where(seen, self.distance, np.inf).min(axis=-1), ground)


@attrs.frozen
class RobotState:
    """World pose, body-frame twist and body-frame acceleration over the last step of one robot."""

    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    wz: float
    ax: float = 0.0
    ay: float = 0.0


@attrs.define
class _Entity:
    z: tuple[float, float]
    segments: np.ndarray | None = None
    corners: np.ndarray | None = None
    radius: float = 0.0
    pose: Pose2D = (0.0, 0.0, 0.0)
    velocity: Point = (0.0, 0.0)
    drifting: bool = False
    levels: range = range(0)
    bodies: list[b2.Body] = attrs.field(factory=list)
    shapes: list[b2.Shape] = attrs.field(factory=list)


@attrs.define
class _Robot:
    body: b2.Body
    shape: b2.Shape
    hull: b2.Hull | None
    corners: np.ndarray | None
    radius: float
    level: int
    z: float
    height: float
    drive: DriveModel
    command: Twist2D = (0.0, 0.0, 0.0)
    setpoint: Twist2D = (0.0, 0.0, 0.0)
    command_time: float = -math.inf
    twist: Twist2D = (0.0, 0.0, 0.0)
    acceleration: Point = (0.0, 0.0)


@attrs.frozen
class _Primitives:
    """Segments (two-sided or polygon entry edges) and circles in world coordinates, each with its owner index and the first and last level it stands on."""

    start: np.ndarray
    end: np.ndarray
    one_sided: np.ndarray
    segment_owner: np.ndarray
    segment_first: np.ndarray
    segment_last: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    circle_owner: np.ndarray
    circle_first: np.ndarray
    circle_last: np.ndarray


def _ccw(corners: np.ndarray) -> np.ndarray:
    area = np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1])
    return corners if area >= 0.0 else corners[::-1].copy()


def _placed(corners: np.ndarray, transform: b2.Transform) -> np.ndarray:
    cos, sin = transform.q.c, transform.q.s
    return corners @ np.array([[cos, sin], [-sin, cos]]) + (transform.p.x, transform.p.y)


def _primitives(segments: list[tuple[np.ndarray, bool, int, range]], circles: list[tuple[np.ndarray, float, int, range]]) -> _Primitives:
    """Pack (points, one sided, owner, levels) segment groups and (center, radius, owner, levels) circles."""
    counts = [len(points) for points, _, _, _ in segments]
    stacked = np.concatenate([points for points, _, _, _ in segments]) if segments else np.zeros((0, 2, 2))
    return _Primitives(
        start=stacked[:, 0],
        end=stacked[:, 1],
        one_sided=np.repeat(np.array([one for _, one, _, _ in segments], dtype=bool), counts),
        segment_owner=np.repeat(np.array([owner for _, _, owner, _ in segments], dtype=np.intp), counts),
        segment_first=np.repeat(np.array([levels.start for _, _, _, levels in segments], dtype=np.intp), counts),
        segment_last=np.repeat(np.array([levels.stop - 1 for _, _, _, levels in segments], dtype=np.intp), counts),
        center=np.array([center for center, _, _, _ in circles], dtype=np.float64).reshape(-1, 2),
        radius=np.array([radius for _, radius, _, _ in circles], dtype=np.float64),
        circle_owner=np.array([owner for _, _, owner, _ in circles], dtype=np.intp),
        circle_first=np.array([levels.start for _, _, _, levels in circles], dtype=np.intp),
        circle_last=np.array([levels.stop - 1 for _, _, _, levels in circles], dtype=np.intp),
    )


def _edges_of(corners: np.ndarray) -> np.ndarray:
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


def _beam_pairs(start: np.ndarray, span: np.ndarray, theta0: float, increment: float, beams: int) -> tuple[np.ndarray, np.ndarray]:
    """(candidate, beam) pairs whose beam points into the arc [start, start + span] of each candidate, every pair once for arcs under a full turn."""
    candidates = np.arange(len(start))
    if beams == 1 or increment == 0.0:
        return candidates, np.zeros(len(start), dtype=np.intp)
    flip = increment < 0.0
    if flip:
        theta0, increment = theta0 + increment * (beams - 1), -increment
    rel = np.mod(start - theta0, 2.0 * math.pi)
    lows, highs = [], []
    turns = math.ceil(increment * (beams - 1) / (2.0 * math.pi))
    for shift in 2.0 * math.pi * np.arange(-1, turns + 1):
        lows.append(np.clip(np.ceil((rel + shift - _ARC_EPS) / increment), 0, beams))
        highs.append(np.clip(np.floor((rel + shift + span + _ARC_EPS) / increment), -1, beams - 1))
    low = np.concatenate(lows).astype(np.intp)
    high = np.concatenate(highs).astype(np.intp)
    owner = np.tile(candidates, len(lows))
    count = np.maximum(high - low + 1, 0)
    total = int(count.sum())
    which = np.repeat(owner, count)
    beam = np.repeat(low - np.cumsum(count) + count, count) + np.arange(total)
    return which, (beams - 1 - beam if flip else beam)


_NO_HITS = (np.zeros(0, dtype=np.intp), np.zeros(0), np.zeros(0, dtype=np.intp))


def _segment_hits(primitives: _Primitives, origin: np.ndarray, rays: np.ndarray, theta0: float, increment: float, reach: float, level: int, caster: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(beam, fraction, owner) of every segment crossing, polygon edges counting only where a ray enters."""
    start = primitives.start - origin
    span = primitives.end - primitives.start
    length = (span**2).sum(axis=1)
    along = np.clip(-(start * span).sum(axis=1) / np.where(length > 0.0, length, 1.0), 0.0, 1.0)
    closest = start + along[:, None] * span
    keep = np.flatnonzero((primitives.segment_first <= level) & (level <= primitives.segment_last) & (primitives.segment_owner != caster) & ((closest**2).sum(axis=1) <= reach * reach))
    if not len(keep):
        return _NO_HITS
    start, span, end = start[keep], span[keep], start[keep] + span[keep]
    first = np.arctan2(start[:, 1], start[:, 0])
    turn = np.mod(np.arctan2(end[:, 1], end[:, 0]) - first + math.pi, 2.0 * math.pi) - math.pi
    which, beam = _beam_pairs(np.where(turn < 0.0, first + turn, first), np.abs(turn), theta0, increment, len(rays))
    ray, rel, edge = rays[beam], start[which], span[which]
    den = ray[:, 0] * edge[:, 1] - ray[:, 1] * edge[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (rel[:, 0] * edge[:, 1] - rel[:, 1] * edge[:, 0]) / den
        u = (rel[:, 0] * ray[:, 1] - rel[:, 1] * ray[:, 0]) / den
    hit = (den != 0.0) & ~(primitives.one_sided[keep][which] & (den > 0.0)) & (t > 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return beam[hit], t[hit], primitives.segment_owner[keep][which][hit]


def _circle_hits(primitives: _Primitives, origin: np.ndarray, rays: np.ndarray, theta0: float, increment: float, reach: float, level: int, caster: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(beam, fraction, owner) where a ray enters a circle, none from inside one."""
    rel = primitives.center - origin
    gap = np.sqrt((rel**2).sum(axis=1))
    keep = np.flatnonzero((primitives.circle_first <= level) & (level <= primitives.circle_last) & (primitives.circle_owner != caster) & (gap > primitives.radius) & (gap - primitives.radius <= reach))
    if not len(keep):
        return _NO_HITS
    rel, gap, radius = rel[keep], gap[keep], primitives.radius[keep]
    half = np.arcsin(np.clip(radius / gap, 0.0, 1.0))
    which, beam = _beam_pairs(np.arctan2(rel[:, 1], rel[:, 0]) - half, 2.0 * half, theta0, increment, len(rays))
    ray, offset = rays[beam], -rel[which]
    a = (ray**2).sum(axis=1)
    b = (offset * ray).sum(axis=1)
    disc = b * b - a * ((offset**2).sum(axis=1) - radius[which] ** 2)
    t = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
    hit = (disc >= 0.0) & (t > 0.0) & (t <= 1.0)
    return beam[hit], t[hit], primitives.circle_owner[keep][which][hit]


def _core(shape: b2.Shape) -> tuple[np.ndarray, float]:
    """World-frame vertices of a shape's convex core and its rounding radius."""
    typed = shape.cast()
    if shape.type == b2.ShapeType.circle:
        local, radius = [(typed.circle.center.x, typed.circle.center.y)], typed.circle.radius
    elif shape.type == b2.ShapeType.segment:
        local, radius = [(typed.segment.point1.x, typed.segment.point1.y), (typed.segment.point2.x, typed.segment.point2.y)], 0.0
    else:
        local, radius = list(typed.polygon.vertices), typed.polygon.radius
    transform = shape.body.transform
    cos, sin = transform.q.c, transform.q.s
    return np.asarray(local, dtype=np.float64) @ np.array([[cos, sin], [-sin, cos]]) + (transform.p.x, transform.p.y), float(radius)


def _edges(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end points of a core's edges, a point being its own edge."""
    if len(core) <= 2:
        return core[:1], core[-1:]
    return core, np.roll(core, -1, axis=0)


def _axes(core: np.ndarray) -> np.ndarray:
    """Outward unit normals of a core's edges, both sides of a segment, none for a point."""
    start, end = _edges(core)
    span = end - start
    length = np.linalg.norm(span, axis=1)
    start, span, length = start[length > 0], span[length > 0], length[length > 0]
    normals = np.stack([span[:, 1], -span[:, 0]], axis=1) / length[:, None]
    if len(core) <= 2:
        return np.concatenate([normals, -normals])
    inward = ((core.mean(axis=0) - start) * normals).sum(axis=1) > 0
    return np.where(inward[:, None], -normals, normals)


def _nearest(points: np.ndarray, core: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Smallest distance from `points` to a core's edges with the point and the edge point realizing it."""
    start, end = _edges(core)
    span = end - start
    length = (span**2).sum(axis=1)
    along = ((points[:, None] - start[None]) * span[None]).sum(axis=2) / np.where(length > 0, length, 1.0)
    closest = start[None] + np.clip(along, 0.0, 1.0)[..., None] * span[None]
    gap = np.linalg.norm(points[:, None] - closest, axis=2)
    i, j = np.unravel_index(np.argmin(gap), gap.shape)
    return float(gap[i, j]), points[i], closest[i, j]


def _inside(points: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Mask of `points` within a polygon core, all False for a segment or point."""
    if len(core) < 3:
        return np.zeros(len(points), dtype=bool)
    start, end = _edges(core)
    span = end - start
    rel = points[:, None] - start[None]
    side = span[None, :, 0] * rel[..., 1] - span[None, :, 1] * rel[..., 0]
    return (side >= 0).all(axis=1) | (side <= 0).all(axis=1)  # pyright: ignore[reportReturnType]


def _crossings(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Points where an edge of core a crosses an edge of core b."""
    start_a, end_a = _edges(a)
    start_b, end_b = _edges(b)
    span_a = (end_a - start_a)[:, None]
    span_b = (end_b - start_b)[None]
    rel = start_b[None] - start_a[:, None]
    den = span_a[..., 0] * span_b[..., 1] - span_a[..., 1] * span_b[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (rel[..., 0] * span_b[..., 1] - rel[..., 1] * span_b[..., 0]) / den
        u = (rel[..., 0] * span_a[..., 1] - rel[..., 1] * span_a[..., 0]) / den
    crossed = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return (start_a[:, None] + np.where(crossed, t, 0.0)[..., None] * span_a)[crossed]


def _touch(a: np.ndarray, radius_a: float, b: np.ndarray, radius_b: float, margin: float) -> tuple[float, Point, Point] | None:
    """Separation, contact point and unit normal from rounded core a into rounded core b, None when they are more than `margin` apart."""
    gap, on_a, on_b = _nearest(a, b)
    reverse = _nearest(b, a)
    if reverse[0] < gap:
        gap, on_b, on_a = reverse
    axes = np.concatenate([_axes(a), -_axes(b)])
    depth = (b @ axes.T).min(axis=0) - (a @ axes.T).max(axis=0)
    if len(axes) and depth.max() <= 0.0:
        shared = np.concatenate([a[_inside(a, b)], b[_inside(b, a)], _crossings(a, b)])
        point = shared.mean(axis=0) if len(shared) else (on_a + on_b) / 2.0
        normal = axes[np.argmax(depth)]
        return float(depth.max()) - radius_a - radius_b, (float(point[0]), float(point[1])), (float(normal[0]), float(normal[1]))
    separation = gap - radius_a - radius_b
    if separation > margin:
        return None
    normal = (on_b - on_a) / gap if gap > 0.0 else np.array([1.0, 0.0])
    point = (on_a + normal * radius_a + on_b - normal * radius_b) / 2.0
    return separation, (float(point[0]), float(point[1])), (float(normal[0]), float(normal[1]))


def _spaced(ring: Sequence[Point]) -> list[Point]:
    """Ring points without those within the linear slop of the last kept one, the ring still closing on its first point."""
    kept: list[Point] = []
    for point in ring:
        if not kept or math.dist(kept[-1], point) > _LINEAR_SLOP:
            kept.append(point)
    while len(kept) > 2 and math.dist(kept[-1], kept[0]) <= _LINEAR_SLOP:
        kept.pop()
    return kept


_WALL_FILTER = b2.make_filter(category_bits=_WALL, mask_bits=_WALL | _DISC | _ROBOT | _QUERY)
_DISC_FILTER = b2.make_filter(category_bits=_DISC, mask_bits=_WALL | _DISC | _QUERY)
_ROBOT_FILTER = b2.make_filter(category_bits=_ROBOT, mask_bits=_WALL | _ROBOT | _QUERY)


def _new_world() -> b2.World:
    world = b2.World(gravity=(0.0, 0.0))
    world.sleeping_enabled = False
    return world


class FlatWorld:
    """One Box2D world per level: static outlines, kinematic boxes and discs, velocity-driven robots."""

    def __init__(self, dt: float, dynamics: Dynamics = Dynamics.RIGID) -> None:
        if dt <= 0:
            raise ValueError(f'dt must be positive, got {dt}')
        self._dt = dt
        self._dynamics = dynamics
        self._worlds = [_new_world()]
        self._levels: list[float] = []
        self._entities: dict[str, _Entity] = {}
        self._robots: dict[str, _Robot] = {}
        self._owners: dict[int, str] = {}
        self._uids: dict[str, int] = {}
        self._names: dict[int, str] = {}
        self._next_uid = 0
        self._static: _Primitives | None = None
        self._resting: tuple[int, list[tuple[np.ndarray, bool, int, range]]] | None = None
        self._moving: tuple[tuple[int, int], _Primitives] | None = None
        self._steps = 0
        self._version = 0
        self.time = 0.0

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dynamics(self) -> Dynamics:
        return self._dynamics

    @property
    def names(self) -> tuple[str, ...]:
        return (*self._entities, *self._robots)

    def __contains__(self, name: str) -> bool:
        return name in self._entities or name in self._robots

    def set_levels(self, floors_z: Iterable[float]) -> None:
        """Floor heights that split z into levels, lowest first."""
        levels: list[float] = []
        for z in sorted(floors_z):
            if not levels or z - levels[-1] > _LEVEL_EPS:
                levels.append(z)
        if levels == self._levels:
            return
        self._levels = levels
        old = self._worlds
        self._worlds = [_new_world() for _ in range(max(1, len(levels)))]
        for name, entity in self._entities.items():
            self._place(name, entity)
        for name, robot in self._robots.items():
            position, velocity = robot.body.position, robot.body.linear_velocity
            moving, angle, spin = (velocity.x, velocity.y), robot.body.angle, robot.body.angular_velocity
            self._rehome(name, robot, self.level(robot.z), (position.x, position.y), angle)
            robot.body.linear_velocity = moving
            robot.body.angular_velocity = spin
        for world in old:
            world.destroy()
        self._static = None
        self._version += 1

    def level(self, z: float) -> int:
        """Index of the level whose slab contains z."""
        return max(0, bisect.bisect_right(self._levels, z + _LEVEL_EPS) - 1)

    def _span(self, z: tuple[float, float]) -> range:
        """Levels an extent rises more than GROUND_SLAB into, none for a ground slab."""
        if not self._levels:
            return range(1)
        ceilings = [*self._levels[1:], math.inf]
        inside = [level for level, (floor, ceiling) in enumerate(zip(self._levels, ceilings, strict=True)) if z[1] > floor + GROUND_SLAB and z[0] < ceiling - GROUND_SLAB]
        return range(inside[0], inside[-1] + 1) if inside else range(0)

    def _own(self, name: str) -> None:
        self._uids[name] = self._next_uid
        self._names[self._next_uid] = name
        self._next_uid += 1
        self._version += 1

    def _disown(self, name: str, shapes: Iterable[b2.Shape]) -> None:
        for shape in shapes:
            del self._owners[shape.id]
        del self._names[self._uids.pop(name)]
        self._version += 1

    def _extent(self, name: str) -> tuple[float, float]:
        entity = self._entities.get(name)
        if entity is not None:
            return entity.z
        robot = self._robots[name]
        return robot.z, robot.z + robot.height

    def _place(self, name: str, entity: _Entity) -> None:
        """Bodies of an entity in the world of every level it rises into, a shapeless anchor keeping the pose of one that rises into none."""
        position = (entity.bodies[0].position.x, entity.bodies[0].position.y) if entity.bodies else (entity.pose[0], entity.pose[1])
        rotation = b2.Rot(entity.pose[2])
        for shape in entity.shapes:
            del self._owners[shape.id]
        for body in entity.bodies:
            body.destroy()
        entity.levels = self._span(entity.z)
        entity.bodies, entity.shapes = [], []
        for level in entity.levels:
            world = self._worlds[level]
            if entity.segments is not None:
                body = world.create_static_body(position=(0.0, 0.0))
                shape_def = b2.shape_def(filter=_WALL_FILTER)
                for start, end in entity.segments:
                    segment = b2.Segment()
                    segment.point1 = (float(start[0]), float(start[1]))
                    segment.point2 = (float(end[0]), float(end[1]))
                    entity.shapes.append(body.create_segment_shape(shape_def, segment))
            else:
                body = self._kinematic(world, entity, position, rotation)
                if entity.corners is not None:
                    half_x, half_y = entity.corners[2]
                    shape_def = b2.shape_def(filter=_WALL_FILTER)
                    entity.shapes.append(body.create_polygon_shape(shape_def, b2.box(float(half_x), float(half_y))))
                else:
                    circle = b2.Circle()
                    circle.center = (0.0, 0.0)
                    circle.radius = entity.radius
                    shape_def = b2.shape_def(filter=_DISC_FILTER)
                    entity.shapes.append(body.create_circle_shape(shape_def, circle))
            entity.bodies.append(body)
        if not entity.bodies and entity.segments is None:
            entity.bodies.append(self._kinematic(self._worlds[0], entity, position, rotation))
        for shape in entity.shapes:
            self._owners[shape.id] = name

    @staticmethod
    def _kinematic(world: b2.World, entity: _Entity, position: Point, rotation: b2.Rot) -> b2.Body:
        body = world.create_kinematic_body(position=position)
        body.set_transform(b2.Vec2(*position), rotation)
        if entity.drifting:
            body.linear_velocity = entity.velocity
        return body

    def add_outline(self, name: str, rings: Sequence[Sequence[Point]], z: tuple[float, float]) -> None:
        """Static obstacle whose closed rings become two-sided segments."""
        self.remove(name)
        segments: list[tuple[Point, Point]] = []
        for ring in map(_spaced, rings):
            for i, start in enumerate(ring):
                end = ring[(i + 1) % len(ring)]
                if math.dist(start, end) <= _LINEAR_SLOP:
                    continue
                segments.append(((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))
        entity = _Entity(z, segments=np.array(segments, dtype=np.float64).reshape(-1, 2, 2))
        self._entities[name] = entity
        self._own(name)
        self._place(name, entity)
        self._static = None

    def add_box(self, name: str, size: Point, pose: Pose2D, z: tuple[float, float]) -> None:
        """Kinematic box of full extents `size`, centered on `pose`."""
        self.remove(name)
        half_x, half_y = size[0] / 2.0, size[1] / 2.0
        entity = _Entity(z, corners=np.array([(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)], dtype=np.float64), pose=(float(pose[0]), float(pose[1]), float(pose[2])))
        self._entities[name] = entity
        self._own(name)
        self._place(name, entity)

    def add_disc(self, name: str, radius: float, position: Point, z: tuple[float, float]) -> None:
        """Kinematic disc, the shape of a pedestrian, touching robots without pushing them."""
        self.remove(name)
        entity = _Entity(z, radius=float(radius), pose=(float(position[0]), float(position[1]), 0.0))
        self._entities[name] = entity
        self._own(name)
        self._place(name, entity)

    def move(self, name: str, pose: Pose2D, velocity: Point = (0.0, 0.0), z: tuple[float, float] | None = None) -> None:
        """Place a box or disc and set the velocity it keeps until the next move."""
        entity = self._entities[name]
        for body in entity.bodies:
            body.set_transform(b2.Vec2(float(pose[0]), float(pose[1])), b2.Rot(float(pose[2])))
            body.linear_velocity = (float(velocity[0]), float(velocity[1]))
        entity.pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        entity.velocity = (float(velocity[0]), float(velocity[1]))
        entity.drifting = velocity[0] != 0.0 or velocity[1] != 0.0
        self._version += 1
        if z is not None and z != entity.z:
            entity.z = z
            if self._span(z) != entity.levels:
                self._place(name, entity)

    def remove(self, name: str) -> bool:
        entity = self._entities.pop(name, None)
        if entity is not None:
            self._disown(name, entity.shapes)
            if entity.segments is not None:
                self._static = None
            for body in entity.bodies:
                body.destroy()
            return True
        robot = self._robots.pop(name, None)
        if robot is not None:
            self._disown(name, [robot.shape])
            robot.body.destroy()
            return True
        return False

    def _robot_body(self, name: str, level: int, hull: b2.Hull | None, radius: float, position: Point, angle: float) -> tuple[b2.Body, b2.Shape]:
        """Body and shape of a robot in the world of `level`, a polygon from its hull or a disc of `radius`."""
        world = self._worlds[level]
        body = world.create_dynamic_body(position=position) if self._dynamics is Dynamics.RIGID else world.create_kinematic_body(position=position)
        body.set_transform(b2.Vec2(*position), b2.Rot(angle))
        shape_def = b2.shape_def(density=1.0, filter=_ROBOT_FILTER)
        material = shape_def.material
        material.friction = 0.0
        shape_def.material = material
        if hull is not None:
            shape = body.create_polygon_shape(shape_def, b2.polygon(hull=hull))
        else:
            circle = b2.Circle()
            circle.center = (0.0, 0.0)
            circle.radius = radius
            shape = body.create_circle_shape(shape_def, circle)
        self._owners[shape.id] = name
        return body, shape

    def _rehome(self, name: str, robot: _Robot, level: int, position: Point, angle: float) -> None:
        """Rebuild a robot's body in the world of `level` at rest."""
        del self._owners[robot.shape.id]
        robot.body.destroy()
        robot.body, robot.shape = self._robot_body(name, level, robot.hull, robot.radius, position, angle)
        robot.level = level

    def add_robot(
        self,
        name: str,
        pose: Pose2D,
        z: float,
        drive: DriveModel,
        footprint: Sequence[Point] | None = None,
        radius: float = 0.3,
        height: float = ROBOT_HEIGHT,
    ) -> None:
        """Robot body from the convex hull of `footprint`, a disc of `radius` without one, `height` tall above z."""
        self.remove(name)
        hull = b2.compute_hull(np.asarray(footprint, dtype=np.float32)) if footprint is not None and len(footprint) >= 3 else None
        if hull is not None and not 3 <= hull.count <= _MAX_HULL:
            hull = None
        level = self.level(z)
        body, shape = self._robot_body(name, level, hull, float(radius), (float(pose[0]), float(pose[1])), float(pose[2]))
        corners = _ccw(np.asarray(shape.cast().polygon.vertices, dtype=np.float64).reshape(-1, 2)) if hull is not None else None
        self._robots[name] = _Robot(body=body, shape=shape, hull=hull, corners=corners, radius=float(radius), level=level, z=z, height=height, drive=drive)
        self._own(name)

    def set_robot_pose(self, name: str, pose: Pose2D, z: float) -> None:
        """Teleport a robot to rest, switching level when z crosses a floor."""
        robot = self._robots[name]
        level = self.level(z)
        if level != robot.level:
            self._rehome(name, robot, level, (float(pose[0]), float(pose[1])), float(pose[2]))
        else:
            robot.body.set_transform(b2.Vec2(float(pose[0]), float(pose[1])), b2.Rot(float(pose[2])))
            robot.body.linear_velocity = (0.0, 0.0)
            robot.body.angular_velocity = 0.0
        robot.command = (0.0, 0.0, 0.0)
        robot.setpoint = (0.0, 0.0, 0.0)
        robot.twist = (0.0, 0.0, 0.0)
        robot.acceleration = (0.0, 0.0)
        robot.z = z
        self._version += 1

    def command(self, name: str, twist: Twist2D) -> None:
        """Body-frame velocity target, held for the drive timeout of world time."""
        if not all(math.isfinite(v) for v in twist):
            return
        robot = self._robots[name]
        robot.command = (float(twist[0]), float(twist[1]), float(twist[2]))
        robot.command_time = self.time

    def robot_state(self, name: str) -> RobotState:
        robot = self._robots[name]
        position = robot.body.position
        return RobotState(position.x, position.y, robot.z, robot.body.angle, *robot.twist, *robot.acceleration)

    def advance(self, to: float) -> int:
        """Take every whole dt step that fits before world time `to`, returns the step count, a clock that ran backwards rebases."""
        start = self.time
        if to < start:
            self.time = to
            return 0
        steps = int((to - start) / self._dt + _STEP_EPS)
        for i in range(steps):
            self._step(self._dt)
            self.time = start + (i + 1) * self._dt
        if abs(to - self.time) < _STEP_EPS * self._dt:
            self.time = to
        return steps

    def _step(self, dt: float) -> None:
        before = [(robot.twist, robot.body.angle) for robot in self._robots.values()]
        for robot in self._robots.values():
            drive = robot.drive
            command = robot.command if self.time - robot.command_time <= drive.timeout else (0.0, 0.0, 0.0)
            robot.setpoint = drive.setpoint(robot.setpoint, command, dt)
            vx, vy, wz = drive.follow(robot.twist, robot.setpoint, dt)
            yaw = robot.body.angle
            cos, sin = math.cos(yaw), math.sin(yaw)
            robot.body.linear_velocity = (cos * vx - sin * vy, sin * vx + cos * vy)
            robot.body.angular_velocity = wz
            robot.twist = (vx, vy, wz)
        for world in self._worlds:
            world.step(dt, _SUBSTEPS)
        self._steps += 1
        if self._dynamics is Dynamics.RIGID:
            for robot, (_, yaw) in zip(self._robots.values(), before, strict=True):
                velocity = robot.body.linear_velocity
                cos, sin = math.cos(yaw), math.sin(yaw)
                robot.twist = (cos * velocity.x + sin * velocity.y, -sin * velocity.x + cos * velocity.y, robot.body.angular_velocity)
        for robot, ((vx, vy, _), _) in zip(self._robots.values(), before, strict=True):
            robot.acceleration = ((robot.twist[0] - vx) / dt - robot.twist[2] * robot.twist[1], (robot.twist[1] - vy) / dt + robot.twist[2] * robot.twist[0])

    def _static_primitives(self) -> _Primitives:
        """Outline segments, rebuilt only when an outline or the levels change."""
        if self._static is None:
            self._static = _primitives([(entity.segments, False, self._uids[name], entity.levels) for name, entity in self._entities.items() if entity.segments is not None], [])
        return self._static

    def _resting_boxes(self) -> list[tuple[np.ndarray, bool, int, range]]:
        """Edges of every box at rest, placed from the pose of its last move and rebuilt only on a change."""
        if self._resting is not None and self._resting[0] == self._version:
            return self._resting[1]
        boxes = [(name, entity, entity.corners) for name, entity in self._entities.items() if entity.corners is not None and not entity.drifting]
        groups: list[tuple[np.ndarray, bool, int, range]] = []
        if boxes:
            corners = np.stack([box_corners for _, _, box_corners in boxes])
            pose = np.array([entity.pose for _, entity, _ in boxes], dtype=np.float64)
            cos, sin = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
            placed = np.stack([cos * corners[..., 0] - sin * corners[..., 1] + pose[:, :1], sin * corners[..., 0] + cos * corners[..., 1] + pose[:, 1:2]], axis=-1)
            edges = np.stack([placed, placed[:, _BOX_EDGES]], axis=2)
            groups = [(edges[i], True, self._uids[name], entity.levels) for i, (name, entity, _) in enumerate(boxes)]
        self._resting = (self._version, groups)
        return groups

    def _moving_primitives(self) -> _Primitives:
        """Boxes, discs and robots where they are now, rebuilt once per step or change."""
        key = (self._steps, self._version)
        if self._moving is not None and self._moving[0] == key:
            return self._moving[1]
        segments = list(self._resting_boxes())
        circles: list[tuple[np.ndarray, float, int, range]] = []
        for name, entity in self._entities.items():
            if entity.corners is not None and entity.drifting:
                segments.append((_edges_of(_placed(entity.corners, entity.bodies[0].transform)), True, self._uids[name], entity.levels))
            elif entity.corners is None and entity.segments is None:
                position = entity.bodies[0].position
                circles.append((np.array((position.x, position.y)), entity.radius, self._uids[name], entity.levels))
        for name, robot in self._robots.items():
            levels = range(robot.level, robot.level + 1)
            if robot.corners is not None:
                segments.append((_edges_of(_placed(robot.corners, robot.body.transform)), True, self._uids[name], levels))
            else:
                position = robot.body.position
                circles.append((np.array((position.x, position.y)), robot.radius, self._uids[name], levels))
        primitives = _primitives(segments, circles)
        self._moving = (key, primitives)
        return primitives

    def cast(self, name: str, spec: ScanSpec) -> RayHits:
        """Every entity each beam of one fan from robot `name` crosses within range_max."""
        robot = self._robots[name]
        position = robot.body.position
        yaw = robot.body.angle
        mount_x, mount_y, mount_yaw = spec.mount
        cos, sin = math.cos(yaw), math.sin(yaw)
        origin = np.array((position.x + cos * mount_x - sin * mount_y, position.y + sin * mount_x + cos * mount_y))
        theta0 = yaw + mount_yaw + spec.angle_min
        angles = theta0 + spec.angle_increment * np.arange(spec.beams)
        rays = np.stack([np.cos(angles), np.sin(angles)], axis=1) * spec.range_max
        caster = self._uids[name]
        found = [
            hits
            for primitives in (self._static_primitives(), self._moving_primitives())
            for hits in (
                _segment_hits(primitives, origin, rays, theta0, spec.angle_increment, spec.range_max, robot.level, caster),
                _circle_hits(primitives, origin, rays, theta0, spec.angle_increment, spec.range_max, robot.level, caster),
            )
        ]
        beam = np.concatenate([hits[0] for hits in found])
        fraction = np.concatenate([hits[1] for hits in found])
        owner = np.concatenate([hits[2] for hits in found])

        order = np.lexsort((fraction, beam))
        row = beam[order]
        count = np.bincount(row, minlength=spec.beams)
        column = np.arange(len(row)) - (np.cumsum(count) - count)[row]
        size = (spec.beams, max(1, int(count.max(initial=0))))
        uids, index = np.unique(owner[order], return_inverse=True)
        extents = np.array([self._extent(self._names[int(uid)]) for uid in uids], dtype=np.float64).reshape(-1, 2) - (robot.z + spec.height)
        distance = np.full(size, np.inf)
        lo = np.full(size, np.inf)
        hi = np.full(size, -np.inf)
        distance[row, column] = fraction[order] * spec.range_max
        lo[row, column] = extents[index, 0]
        hi[row, column] = extents[index, 1]
        floor = self._levels[self.level(robot.z)] if self._levels else 0.0
        return RayHits(distance, lo, hi, floor - (robot.z + spec.height))

    def scan(self, name: str, spec: ScanSpec) -> np.ndarray:
        """Ranges of one fan from robot `name`, inf where no entity spans the sensor height within range_max."""
        return self.cast(name, spec).first().astype(np.float32)

    def contacts(self, name: str) -> list[Touch]:
        """Entities and peers robot `name` touches, each with one contact point and the unit normal out of the robot."""
        robot = self._robots[name]
        margin = _CONTACT_MARGIN if self._dynamics is Dynamics.RIGID else 0.0
        box = robot.body.compute_aabb()
        query = b2.QueryFilter()
        query.category_bits = _QUERY
        query.mask_bits = _WALL | _DISC | _ROBOT
        found: list[b2.Shape] = []
        self._worlds[robot.level].overlap_aabb(b2.aabb((box.lower_bound.x - margin, box.lower_bound.y - margin), (box.upper_bound.x + margin, box.upper_bound.y + margin)), found.append, query)
        core, radius = _core(robot.shape)
        nearest: dict[str, tuple[float, Point, Point]] = {}
        for shape in found:
            if shape.id == robot.shape.id:
                continue
            owner = self._owners[shape.id]
            touch = _touch(core, radius, *_core(shape), margin)
            if touch is not None and (owner not in nearest or touch[0] < nearest[owner][0]):
                nearest[owner] = touch
        return [(owner, point, normal) for owner, (_, point, normal) in sorted(nearest.items())]

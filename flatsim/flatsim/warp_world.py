"""The FlatWorld API on Warp: robot bodies, their contacts and every ray cast on one device, entities and levels on the host."""

# pyright: reportInvalidTypeForm=false

from __future__ import annotations

import bisect
import functools
import math
from collections.abc import Iterable, Sequence

import attrs
import numpy as np
import warp as wp

from flatsim.drive import DriveModel
from flatsim.world import (
    _CONTACT_MARGIN,
    _LEVEL_EPS,
    _LINEAR_SLOP,
    _STEP_EPS,
    _SUBSTEPS,
    GROUND_SLAB,
    ROBOT_HEIGHT,
    Dynamics,
    Point,
    Pose2D,
    RayHits,
    RobotState,
    ScanSpec,
    Touch,
    Twist2D,
    _ccw,
    _robot_mass,
    _RobotMass,
    _spaced,
    _touch,
)

wp.config.quiet = True
wp.set_module_options({'enable_backward': False})

MAX_VERTS = 8
MAX_MANIFOLDS = 64
MAX_HITS = 16
STATIC_CELL = 1.0
MOVING_CELL = 2.0
LINEAR_SLOP = _LINEAR_SLOP
SPECULATIVE = 4.0 * _LINEAR_SLOP
CONTACT_HERTZ = 30.0
DAMPING_RATIO = 10.0
PUSH_SPEED = 3.0
WARM_MATCH = 0.05
MAX_ROTATION = 0.25 * math.pi
MAX_LINEAR_SPEED = 400.0
MAX_GRID_SIDE = 2048
GRID_PAD = 1.0e-3
FLOAT_EPSILON = 1.1920929e-07

_TWO_SIDED, _ONE_SIDED, _CIRCLE = 0, 1, 2
_STATIC, _BOX, _PEER = 0, 1, 2


@wp.func
def _cross(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _spin(w: float, r: wp.vec2) -> wp.vec2:
    return wp.vec2(-w * r[1], w * r[0])


@wp.func
def _rotate(q: wp.vec2, v: wp.vec2) -> wp.vec2:
    return wp.vec2(q[0] * v[0] - q[1] * v[1], q[1] * v[0] + q[0] * v[1])


@wp.func
def _relative(q: wp.vec2, q0: wp.vec2) -> wp.vec2:
    return wp.vec2(q[0] * q0[0] + q[1] * q0[1], q[1] * q0[0] - q[0] * q0[1])


@wp.func
def _integrate(q: wp.vec2, angle: float) -> wp.vec2:
    c = q[0] - angle * q[1]
    s = q[1] + angle * q[0]
    mag = wp.sqrt(c * c + s * s)
    return wp.vec2(c / mag, s / mag)


@wp.func
def _max_separation(va: wp.array(dtype=wp.vec2), na: wp.array(dtype=wp.vec2), oa: int, ca: int, vb: wp.array(dtype=wp.vec2), ob: int, cb: int) -> tuple[float, int]:
    best = wp.float32(-1.0e30)
    edge = wp.int32(0)
    for i in range(ca):
        n = na[oa + i]
        v = va[oa + i]
        si = wp.float32(1.0e30)
        for j in range(cb):
            si = wp.min(si, wp.dot(n, vb[ob + j] - v))
        if si > best:
            best = si
            edge = i
    return best, edge


@wp.func
def _incident(direction: wp.vec2, normals: wp.array(dtype=wp.vec2), offset: int, count: int) -> int:
    """Edge whose normal opposes `direction` most."""
    edge = wp.int32(0)
    low = wp.float32(1.0e30)
    for k in range(count):
        d = wp.dot(direction, normals[offset + k])
        if d < low:
            low = d
            edge = k
    return edge


@wp.func
def _clip(rv: wp.array(dtype=wp.vec2), rn: wp.array(dtype=wp.vec2), ro: int, rc: int, re: int, iv: wp.array(dtype=wp.vec2), io: int, ic: int, ie: int, reach: float) -> tuple[wp.vec2, wp.vec2, wp.vec2, float, float, int]:
    """b2ClipPolygons: incident edge ie clipped to reference edge re, the reference normal, up to two midpoints with their separations."""
    normal = rn[ro + re]
    v11 = rv[ro + re]
    v12 = rv[ro + (re + 1) % rc]
    v21 = iv[io + ie]
    v22 = iv[io + (ie + 1) % ic]
    tangent = wp.vec2(-normal[1], normal[0])
    upper1 = wp.dot(v12 - v11, tangent)
    upper2 = wp.dot(v21 - v11, tangent)
    lower2 = wp.dot(v22 - v11, tangent)
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    s1 = wp.float32(0.0)
    s2 = wp.float32(0.0)
    count = wp.int32(0)
    if upper2 < 0.0 or upper1 < lower2:
        return normal, p1, p2, s1, s2, count
    span = upper2 - lower2
    lower = wp.vec2(v22)
    upper = wp.vec2(v21)
    if lower2 < 0.0 and span > FLOAT_EPSILON:
        lower = v22 + (v21 - v22) * (-lower2 / span)
    if upper2 > upper1 and span > FLOAT_EPSILON:
        upper = v22 + (v21 - v22) * ((upper1 - lower2) / span)
    d1 = wp.dot(lower - v11, normal)
    d2 = wp.dot(upper - v11, normal)
    if d1 <= reach:
        p1 = lower - normal * (0.5 * d1)
        s1 = d1
        count = 1
    if d2 <= reach:
        if count == 0:
            p1 = upper - normal * (0.5 * d2)
            s1 = d2
        else:
            p2 = upper - normal * (0.5 * d2)
            s2 = d2
        count += 1
    return normal, p1, p2, s1, s2, count


@wp.func
def _segment_distance(p1: wp.vec2, q1: wp.vec2, p2: wp.vec2, q2: wp.vec2) -> tuple[float, float, float]:
    """b2SegmentDistance: distance and the fraction along each segment."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    dd1 = wp.dot(d1, d1)
    dd2 = wp.dot(d2, d2)
    rd1 = wp.dot(r, d1)
    rd2 = wp.dot(r, d2)
    d12 = wp.dot(d1, d2)
    denominator = dd1 * dd2 - d12 * d12
    f1 = wp.float32(0.0)
    if denominator != 0.0:
        f1 = wp.clamp((d12 * rd2 - rd1 * dd2) / denominator, 0.0, 1.0)
    f2 = (d12 * f1 + rd2) / dd2
    if f2 < 0.0:
        f2 = 0.0
        f1 = wp.clamp(-rd1 / dd1, 0.0, 1.0)
    elif f2 > 1.0:
        f2 = 1.0
        f1 = wp.clamp((d12 - rd1) / dd1, 0.0, 1.0)
    return wp.length(p1 + d1 * f1 - (p2 + d2 * f2)), f1, f2


@wp.func
def _collide_polygons(va: wp.array(dtype=wp.vec2), na: wp.array(dtype=wp.vec2), oa: int, ca: int, vb: wp.array(dtype=wp.vec2), nb: wp.array(dtype=wp.vec2), ob: int, cb: int, reach: float) -> tuple[wp.vec2, wp.vec2, wp.vec2, float, float, int]:
    """b2CollidePolygons from polygon a to polygon b (two-vertex polygons are two-sided segments): normal from a to b, up to two midpoints with their separations."""
    sep_a, edge_a = _max_separation(va, na, oa, ca, vb, ob, cb)
    sep_b, edge_b = _max_separation(vb, nb, ob, cb, va, oa, ca)
    if sep_a > reach or sep_b > reach:
        return wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0), 0.0, 0.0, 0
    normal = wp.vec2(0.0, 0.0)
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    s1 = wp.float32(0.0)
    s2 = wp.float32(0.0)
    count = wp.int32(0)
    if sep_a >= sep_b:
        edge_b = _incident(na[oa + edge_a], nb, ob, cb)
        normal, p1, p2, s1, s2, count = _clip(va, na, oa, ca, edge_a, vb, ob, cb, edge_b, reach)
    else:
        edge_a = _incident(nb[ob + edge_b], na, oa, ca)
        normal, p1, p2, s1, s2, count = _clip(vb, nb, ob, cb, edge_b, va, oa, ca, edge_a, reach)
        normal = -normal
    if sep_a > 0.1 * LINEAR_SLOP or sep_b > 0.1 * LINEAR_SLOP:
        a1 = va[oa + edge_a]
        a2 = va[oa + (edge_a + 1) % ca]
        b1 = vb[ob + edge_b]
        b2 = vb[ob + (edge_b + 1) % cb]
        distance, f1, f2 = _segment_distance(a1, a2, b1, b2)
        if distance > reach:
            return wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0), 0.0, 0.0, 0
        least = wp.float32(1.0e30)
        if count > 0:
            least = s1
        if count > 1:
            least = wp.min(least, s2)
        if distance + 0.1 * LINEAR_SLOP < least and (f1 == 0.0 or f1 == 1.0) and (f2 == 0.0 or f2 == 1.0):
            pa = a1 + (a2 - a1) * f1
            pb = b1 + (b2 - b1) * f2
            middle = (pa + pb) * 0.5
            return (pb - pa) / distance, middle, middle, distance, distance, 1
    return normal, p1, p2, s1, s2, count


@wp.func
def _collide_circle(center: wp.vec2, radius: float, vb: wp.array(dtype=wp.vec2), nb: wp.array(dtype=wp.vec2), ob: int, cb: int, reach: float) -> tuple[wp.vec2, wp.vec2, float, int]:
    """Manifold from a circle to polygon b: normal from the circle to b, one midpoint and its separation."""
    best = wp.float32(-1.0e30)
    edge = wp.int32(0)
    for k in range(cb):
        d = wp.dot(nb[ob + k], center - vb[ob + k])
        if d > best:
            best = d
            edge = k
    normal = wp.vec2(0.0, 0.0)
    point = wp.vec2(0.0, 0.0)
    separation = wp.float32(0.0)
    count = wp.int32(0)
    if best <= 0.0 and cb > 2:
        normal = -nb[ob + edge]
        separation = best - radius
    else:
        nearest = wp.vec2(0.0, 0.0)
        gap = wp.float32(1.0e30)
        for k in range(cb):
            a = vb[ob + k]
            e = vb[ob + (k + 1) % cb] - a
            along = wp.clamp(wp.dot(center - a, e) / wp.max(wp.dot(e, e), 1.0e-12), 0.0, 1.0)
            q = a + e * along
            d = wp.length(q - center)
            if d < gap:
                gap = d
                nearest = q
        if gap > 1.0e-9:
            normal = (nearest - center) / gap
        else:
            normal = -nb[ob + edge]
        separation = gap - radius
    if separation <= reach:
        point = center + normal * (radius + 0.5 * separation)
        count = 1
    return normal, point, separation, count


@wp.func
def _soft(hertz: float, h: float) -> tuple[float, float, float]:
    omega = 2.0 * wp.pi * hertz
    a1 = 2.0 * DAMPING_RATIO + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return omega / a1, a2 * a3, a3


@wp.kernel
def _place_robots(com: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2), center: wp.array(dtype=wp.vec2), local_v: wp.array(dtype=wp.vec2), local_n: wp.array(dtype=wp.vec2), count: wp.array(dtype=int), world_v: wp.array(dtype=wp.vec2), world_n: wp.array(dtype=wp.vec2), points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q = rot[i]
    origin = com[i] - _rotate(q, center[i])
    for k in range(count[i]):
        world_v[i * MAX_VERTS + k] = origin + _rotate(q, local_v[i * MAX_VERTS + k])
        world_n[i * MAX_VERTS + k] = _rotate(q, local_n[i * MAX_VERTS + k])
    points[i] = wp.vec3(com[i][0], com[i][1], 0.0)


@wp.kernel
def _place_boxes(center: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2), half: wp.array(dtype=wp.vec2), world_v: wp.array(dtype=wp.vec2), world_n: wp.array(dtype=wp.vec2)):
    j = wp.tid()
    q = rot[j]
    h = half[j]
    c = center[j]
    world_v[j * 4 + 0] = c + _rotate(q, wp.vec2(-h[0], -h[1]))
    world_v[j * 4 + 1] = c + _rotate(q, wp.vec2(h[0], -h[1]))
    world_v[j * 4 + 2] = c + _rotate(q, wp.vec2(h[0], h[1]))
    world_v[j * 4 + 3] = c + _rotate(q, wp.vec2(-h[0], h[1]))
    world_n[j * 4 + 0] = _rotate(q, wp.vec2(0.0, -1.0))
    world_n[j * 4 + 1] = _rotate(q, wp.vec2(1.0, 0.0))
    world_n[j * 4 + 2] = _rotate(q, wp.vec2(0.0, 1.0))
    world_n[j * 4 + 3] = _rotate(q, wp.vec2(-1.0, 0.0))


@wp.func
def _record(
    i: int,
    slot: int,
    kind: int,
    other: int,
    normal: wp.vec2,
    p1: wp.vec2,
    p2: wp.vec2,
    s1: float,
    s2: float,
    points: int,
    com_a: wp.vec2,
    com_b: wp.vec2,
    inv_mass_a: float,
    inv_inertia_a: float,
    inv_mass_b: float,
    inv_inertia_b: float,
    c_kind: wp.array2d(dtype=int),
    c_other: wp.array2d(dtype=int),
    c_normal: wp.array2d(dtype=wp.vec2),
    c_points: wp.array2d(dtype=int),
    c_ra: wp.array2d(dtype=wp.vec2),
    c_rb: wp.array2d(dtype=wp.vec2),
    c_sep: wp.array2d(dtype=float),
    c_mass: wp.array2d(dtype=float),
    c_impulse: wp.array2d(dtype=float),
):
    c_kind[i, slot] = kind
    c_other[i, slot] = other
    c_normal[i, slot] = normal
    c_points[i, slot] = points
    for k in range(points):
        p = p1
        s = s1
        if k == 1:
            p = p2
            s = s2
        ra = p - com_a
        rb = p - com_b
        rna = _cross(ra, normal)
        rnb = _cross(rb, normal)
        mass = inv_mass_a + inv_inertia_a * rna * rna + inv_mass_b + inv_inertia_b * rnb * rnb
        c_ra[i, slot * 2 + k] = ra
        c_rb[i, slot * 2 + k] = rb
        c_sep[i, slot * 2 + k] = s
        c_mass[i, slot * 2 + k] = 1.0 / mass
        c_impulse[i, slot * 2 + k] = 0.0


@wp.kernel
def _collide(
    dt: float,
    com: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    level: wp.array(dtype=int),
    kind: wp.array(dtype=int),
    count: wp.array(dtype=int),
    radius: wp.array(dtype=float),
    extent: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    inv_inertia: wp.array(dtype=float),
    robot_v: wp.array(dtype=wp.vec2),
    robot_n: wp.array(dtype=wp.vec2),
    grid: wp.uint64,
    peer_reach: float,
    seg_v: wp.array(dtype=wp.vec2),
    seg_n: wp.array(dtype=wp.vec2),
    seg_first: wp.array(dtype=int),
    seg_last: wp.array(dtype=int),
    seg_cell_x: wp.array(dtype=int),
    seg_cell_y: wp.array(dtype=int),
    cell_start: wp.array(dtype=int),
    cell_items: wp.array(dtype=int),
    grid_origin: wp.vec2,
    grid_cell: float,
    grid_nx: int,
    grid_ny: int,
    box_center: wp.array(dtype=wp.vec2),
    box_extent: wp.array(dtype=float),
    box_first: wp.array(dtype=int),
    box_last: wp.array(dtype=int),
    box_v: wp.array(dtype=wp.vec2),
    box_n: wp.array(dtype=wp.vec2),
    c_count: wp.array(dtype=int),
    c_kind: wp.array2d(dtype=int),
    c_other: wp.array2d(dtype=int),
    c_normal: wp.array2d(dtype=wp.vec2),
    c_points: wp.array2d(dtype=int),
    c_ra: wp.array2d(dtype=wp.vec2),
    c_rb: wp.array2d(dtype=wp.vec2),
    c_sep: wp.array2d(dtype=float),
    c_mass: wp.array2d(dtype=float),
    c_impulse: wp.array2d(dtype=float),
):
    i = wp.tid()
    center = com[i]
    lv = level[i]
    reach = SPECULATIVE + wp.length(vel[i]) * dt
    span = extent[i] + reach
    slots = wp.int32(0)
    is_circle = kind[i] == 1
    lo_x = wp.max(int(wp.floor((center[0] - span - grid_origin[0]) / grid_cell)), 0)
    lo_y = wp.max(int(wp.floor((center[1] - span - grid_origin[1]) / grid_cell)), 0)
    hi_x = wp.min(int(wp.floor((center[0] + span - grid_origin[0]) / grid_cell)), grid_nx - 1)
    hi_y = wp.min(int(wp.floor((center[1] + span - grid_origin[1]) / grid_cell)), grid_ny - 1)
    for cy in range(lo_y, hi_y + 1):
        for cx in range(lo_x, hi_x + 1):
            cell = cy * grid_nx + cx
            for item in range(cell_start[cell], cell_start[cell + 1]):
                s = cell_items[item]
                if seg_first[s] > lv or seg_last[s] < lv:
                    continue
                sx = wp.max(seg_cell_x[s], lo_x)
                sy = wp.max(seg_cell_y[s], lo_y)
                if sx != cx or sy != cy or slots >= MAX_MANIFOLDS:
                    continue
                if is_circle:
                    normal, point, separation, points = _collide_circle(center, radius[i], seg_v, seg_n, s * 2, 2, reach)
                    if points > 0:
                        _record(i, slots, _STATIC, s, normal, point, point, separation, separation, 1, center, point, inv_mass[i], inv_inertia[i], 0.0, 0.0, c_kind, c_other, c_normal, c_points, c_ra, c_rb, c_sep, c_mass, c_impulse)
                        slots += 1
                else:
                    normal, p1, p2, s1, s2, points = _collide_polygons(robot_v, robot_n, i * MAX_VERTS, count[i], seg_v, seg_n, s * 2, 2, reach)
                    if points > 0:
                        _record(i, slots, _STATIC, s, normal, p1, p2, s1, s2, points, center, p1, inv_mass[i], inv_inertia[i], 0.0, 0.0, c_kind, c_other, c_normal, c_points, c_ra, c_rb, c_sep, c_mass, c_impulse)
                        slots += 1
    for j in range(box_center.shape[0]):
        if box_first[j] > lv or box_last[j] < lv or slots >= MAX_MANIFOLDS:
            continue
        if wp.length(box_center[j] - center) > span + box_extent[j]:
            continue
        if is_circle:
            normal, point, separation, points = _collide_circle(center, radius[i], box_v, box_n, j * 4, 4, reach)
            if points > 0:
                _record(i, slots, _BOX, j, normal, point, point, separation, separation, 1, center, box_center[j], inv_mass[i], inv_inertia[i], 0.0, 0.0, c_kind, c_other, c_normal, c_points, c_ra, c_rb, c_sep, c_mass, c_impulse)
                slots += 1
        else:
            normal, p1, p2, s1, s2, points = _collide_polygons(robot_v, robot_n, i * MAX_VERTS, count[i], box_v, box_n, j * 4, 4, reach)
            if points > 0:
                _record(i, slots, _BOX, j, normal, p1, p2, s1, s2, points, center, box_center[j], inv_mass[i], inv_inertia[i], 0.0, 0.0, c_kind, c_other, c_normal, c_points, c_ra, c_rb, c_sep, c_mass, c_impulse)
                slots += 1
    if peer_reach > 0.0:
        query = wp.hash_grid_query(grid, wp.vec3(center[0], center[1], 0.0), peer_reach)
        j = wp.int32(0)
        while wp.hash_grid_query_next(query, j):
            if j == i or level[j] != lv or slots >= MAX_MANIFOLDS:
                continue
            pair = SPECULATIVE + (wp.length(vel[i]) + wp.length(vel[j])) * dt
            if wp.length(com[j] - center) > extent[i] + extent[j] + pair:
                continue
            first = wp.min(i, j)
            second = wp.max(i, j)
            normal = wp.vec2(0.0, 0.0)
            p1 = wp.vec2(0.0, 0.0)
            p2 = wp.vec2(0.0, 0.0)
            s1 = wp.float32(0.0)
            s2 = wp.float32(0.0)
            points = wp.int32(0)
            if kind[first] == 1 and kind[second] == 1:
                d = com[second] - com[first]
                gap = wp.length(d)
                if gap > 1.0e-9:
                    normal = d / gap
                    s1 = gap - radius[first] - radius[second]
                    if s1 <= pair:
                        p1 = com[first] + normal * (radius[first] + 0.5 * s1)
                        points = 1
            elif kind[first] == 1:
                normal, p1, s1, points = _collide_circle(com[first], radius[first], robot_v, robot_n, second * MAX_VERTS, count[second], pair)
            elif kind[second] == 1:
                normal, p1, s1, points = _collide_circle(com[second], radius[second], robot_v, robot_n, first * MAX_VERTS, count[first], pair)
                normal = -normal
            else:
                normal, p1, p2, s1, s2, points = _collide_polygons(robot_v, robot_n, first * MAX_VERTS, count[first], robot_v, robot_n, second * MAX_VERTS, count[second], pair)
            if points > 0:
                if i == second:
                    normal = -normal
                _record(i, slots, _PEER, j, normal, p1, p2, s1, s2, points, center, com[j], inv_mass[i], inv_inertia[i], inv_mass[j], inv_inertia[j], c_kind, c_other, c_normal, c_points, c_ra, c_rb, c_sep, c_mass, c_impulse)
                slots += 1
    c_count[i] = slots


@wp.kernel
def _warm(
    c_count: wp.array(dtype=int),
    c_kind: wp.array2d(dtype=int),
    c_other: wp.array2d(dtype=int),
    c_points: wp.array2d(dtype=int),
    c_ra: wp.array2d(dtype=wp.vec2),
    c_impulse: wp.array2d(dtype=float),
    p_count: wp.array(dtype=int),
    p_kind: wp.array2d(dtype=int),
    p_other: wp.array2d(dtype=int),
    p_points: wp.array2d(dtype=int),
    p_ra: wp.array2d(dtype=wp.vec2),
    p_impulse: wp.array2d(dtype=float),
):
    i = wp.tid()
    for m in range(c_count[i]):
        for o in range(p_count[i]):
            if p_kind[i, o] == c_kind[i, m] and p_other[i, o] == c_other[i, m]:
                for k in range(c_points[i, m]):
                    best = float(WARM_MATCH)
                    for q in range(p_points[i, o]):
                        d = wp.length(p_ra[i, o * 2 + q] - c_ra[i, m * 2 + k])
                        if d < best:
                            best = d
                            c_impulse[i, m * 2 + k] = p_impulse[i, o * 2 + q]


@wp.kernel
def _warm_start(
    vel_in: wp.array(dtype=wp.vec2),
    spin_in: wp.array(dtype=float),
    vel_out: wp.array(dtype=wp.vec2),
    spin_out: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    inv_inertia: wp.array(dtype=float),
    c_count: wp.array(dtype=int),
    c_normal: wp.array2d(dtype=wp.vec2),
    c_points: wp.array2d(dtype=int),
    c_ra: wp.array2d(dtype=wp.vec2),
    c_impulse: wp.array2d(dtype=float),
):
    i = wp.tid()
    v = vel_in[i]
    w = spin_in[i]
    for m in range(c_count[i]):
        n = c_normal[i, m]
        for k in range(c_points[i, m]):
            p = n * c_impulse[i, m * 2 + k]
            v = v - p * inv_mass[i]
            w = w - inv_inertia[i] * _cross(c_ra[i, m * 2 + k], p)
    vel_out[i] = v
    spin_out[i] = w


@wp.kernel
def _solve(
    h: float,
    hertz: float,
    elapsed: float,
    bias: bool,
    vel_in: wp.array(dtype=wp.vec2),
    spin_in: wp.array(dtype=float),
    vel_out: wp.array(dtype=wp.vec2),
    spin_out: wp.array(dtype=float),
    com: wp.array(dtype=wp.vec2),
    rot: wp.array(dtype=wp.vec2),
    com0: wp.array(dtype=wp.vec2),
    rot0: wp.array(dtype=wp.vec2),
    inv_mass: wp.array(dtype=float),
    inv_inertia: wp.array(dtype=float),
    box_velocity: wp.array(dtype=wp.vec2),
    c_count: wp.array(dtype=int),
    c_kind: wp.array2d(dtype=int),
    c_other: wp.array2d(dtype=int),
    c_normal: wp.array2d(dtype=wp.vec2),
    c_points: wp.array2d(dtype=int),
    c_ra: wp.array2d(dtype=wp.vec2),
    c_rb: wp.array2d(dtype=wp.vec2),
    c_sep: wp.array2d(dtype=float),
    c_mass: wp.array2d(dtype=float),
    c_impulse: wp.array2d(dtype=float),
):
    i = wp.tid()
    static_rate, static_scale, static_impulse = _soft(2.0 * hertz, h)
    peer_rate, peer_scale, peer_impulse = _soft(hertz, h)
    lin = vel_in[i]
    ang = spin_in[i]
    turn = _relative(rot[i], rot0[i])
    moved = com[i] - com0[i]
    for m in range(c_count[i]):
        n = c_normal[i, m]
        kind = c_kind[i, m]
        other = c_other[i, m]
        vb = wp.vec2(0.0, 0.0)
        wb = wp.float32(0.0)
        turn_b = wp.vec2(1.0, 0.0)
        shift_b = wp.vec2(0.0, 0.0)
        rate = static_rate
        scale = static_scale
        keep = static_impulse
        if kind == _BOX:
            vb = box_velocity[other]
            shift_b = vb * elapsed
        elif kind == _PEER:
            vb = vel_in[other]
            wb = spin_in[other]
            turn_b = _relative(rot[other], rot0[other])
            shift_b = com[other] - com0[other]
        if kind != _STATIC:
            rate = peer_rate
            scale = peer_scale
            keep = peer_impulse
        for k in range(c_points[i, m]):
            ra = c_ra[i, m * 2 + k]
            rb = c_rb[i, m * 2 + k]
            drift = shift_b + _rotate(turn_b, rb) - rb - moved - _rotate(turn, ra) + ra
            s = c_sep[i, m * 2 + k] + wp.dot(drift, n)
            mass = c_mass[i, m * 2 + k]
            total = c_impulse[i, m * 2 + k]
            push = wp.float32(0.0)
            mass_scale = wp.float32(1.0)
            impulse_scale = wp.float32(0.0)
            if s > 0.0:
                push = s / h
            elif bias:
                push = wp.max(rate * s, -PUSH_SPEED)
                mass_scale = scale
                impulse_scale = keep
            dv = vb + _spin(wb, rb) - lin - _spin(ang, ra)
            vn = wp.dot(dv, n)
            impulse = -mass * mass_scale * (vn + push) - impulse_scale * total
            updated = wp.max(total + impulse, 0.0)
            impulse = updated - total
            c_impulse[i, m * 2 + k] = updated
            p = n * impulse
            lin = lin - p * inv_mass[i]
            ang = ang - inv_inertia[i] * _cross(ra, p)
    vel_out[i] = lin
    spin_out[i] = ang


@wp.kernel
def _integrate_bodies(h: float, vel: wp.array(dtype=wp.vec2), spin: wp.array(dtype=float), com: wp.array(dtype=wp.vec2), rot: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    com[i] = com[i] + vel[i] * h
    rot[i] = _integrate(rot[i], h * spin[i])


@wp.func
def _hit(kind: int, a: wp.vec2, b: wp.vec2, o: wp.vec2, ray: wp.vec2) -> float:
    """Fraction along `ray` from o where it enters primitive (a, b), -1 for none."""
    if kind == _CIRCLE:
        rel = a - o
        r = b[0]
        gap2 = wp.dot(rel, rel)
        if gap2 <= r * r:
            return -1.0
        qa = wp.dot(ray, ray)
        qb = -wp.dot(rel, ray)
        disc = qb * qb - qa * (gap2 - r * r)
        if disc < 0.0:
            return -1.0
        t = (-qb - wp.sqrt(disc)) / qa
        if t > 0.0 and t <= 1.0:
            return t
        return -1.0
    edge = b - a
    rel = a - o
    den = ray[0] * edge[1] - ray[1] * edge[0]
    if den == 0.0 or (kind == _ONE_SIDED and den > 0.0):
        return -1.0
    t = (rel[0] * edge[1] - rel[1] * edge[0]) / den
    u = (rel[0] * ray[1] - rel[1] * ray[0]) / den
    if t > 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
        return t
    return -1.0


@wp.func
def _walk(
    b: int,
    o: wp.vec2,
    direction: wp.vec2,
    reach: float,
    lv: int,
    caster: int,
    base: int,
    kind: wp.array(dtype=int),
    pa: wp.array(dtype=wp.vec2),
    pb: wp.array(dtype=wp.vec2),
    first: wp.array(dtype=int),
    last: wp.array(dtype=int),
    owner: wp.array(dtype=int),
    start: wp.array(dtype=int),
    items: wp.array(dtype=int),
    origin: wp.vec2,
    cell: float,
    nx: int,
    ny: int,
    out_distance: wp.array2d(dtype=float),
    out_owner: wp.array2d(dtype=int),
    out_prim: wp.array2d(dtype=int),
    out_count: wp.array(dtype=int),
):
    if nx == 0:
        return
    ray = direction * reach
    low = origin
    high = origin + wp.vec2(float(nx) * cell, float(ny) * cell)
    enter = wp.float32(0.0)
    leave = float(reach)
    for axis in range(2):
        if wp.abs(direction[axis]) < 1.0e-12:
            if o[axis] < low[axis] or o[axis] > high[axis]:
                return
        else:
            t1 = (low[axis] - o[axis]) / direction[axis]
            t2 = (high[axis] - o[axis]) / direction[axis]
            enter = wp.max(enter, wp.min(t1, t2))
            leave = wp.min(leave, wp.max(t1, t2))
    if enter > leave:
        return
    p = o + direction * enter
    cx = wp.clamp(int(wp.floor((p[0] - origin[0]) / cell)), 0, nx - 1)
    cy = wp.clamp(int(wp.floor((p[1] - origin[1]) / cell)), 0, ny - 1)
    step_x = wp.int32(1)
    step_y = wp.int32(1)
    if direction[0] < 0.0:
        step_x = -1
    if direction[1] < 0.0:
        step_y = -1
    next_x = wp.float32(1.0e30)
    next_y = wp.float32(1.0e30)
    delta_x = wp.float32(1.0e30)
    delta_y = wp.float32(1.0e30)
    if wp.abs(direction[0]) >= 1.0e-12:
        bound = origin[0] + float(cx + wp.max(step_x, 0)) * cell
        next_x = (bound - o[0]) / direction[0]
        delta_x = cell / wp.abs(direction[0])
    if wp.abs(direction[1]) >= 1.0e-12:
        bound = origin[1] + float(cy + wp.max(step_y, 0)) * cell
        next_y = (bound - o[1]) / direction[1]
        delta_y = cell / wp.abs(direction[1])
    t = float(enter)
    while t <= leave and cx >= 0 and cx < nx and cy >= 0 and cy < ny:
        if out_count[b] == MAX_HITS and t > out_distance[b, MAX_HITS - 1]:
            return
        c = cy * nx + cx
        for item in range(start[c], start[c + 1]):
            q = items[item]
            if first[q] > lv or last[q] < lv or owner[q] == caster:
                continue
            f = _hit(kind[q], pa[q], pb[q], o, ray)
            if f < 0.0:
                continue
            pid = base + q
            seen = wp.int32(0)
            n = out_count[b]
            for k in range(n):
                if out_prim[b, k] == pid:
                    seen = 1
            if seen == 1:
                continue
            d = f * reach
            if n == MAX_HITS and d >= out_distance[b, MAX_HITS - 1]:
                continue
            slot = wp.min(n, MAX_HITS - 1)
            while slot > 0:
                if out_distance[b, slot - 1] <= d:
                    break
                out_distance[b, slot] = out_distance[b, slot - 1]
                out_owner[b, slot] = out_owner[b, slot - 1]
                out_prim[b, slot] = out_prim[b, slot - 1]
                slot -= 1
            out_distance[b, slot] = d
            out_owner[b, slot] = owner[q]
            out_prim[b, slot] = pid
            out_count[b] = wp.min(n + 1, MAX_HITS)
        if next_x < next_y:
            t = next_x
            next_x = next_x + delta_x
            cx = cx + step_x
        else:
            t = next_y
            next_y = next_y + delta_y
            cy = cy + step_y


@wp.func
def _walk_first(
    o: wp.vec2,
    direction: wp.vec2,
    reach: float,
    limit: float,
    slope: float,
    base: float,
    lv: int,
    caster: int,
    kind: wp.array(dtype=int),
    pa: wp.array(dtype=wp.vec2),
    pb: wp.array(dtype=wp.vec2),
    first: wp.array(dtype=int),
    last: wp.array(dtype=int),
    owner: wp.array(dtype=int),
    start: wp.array(dtype=int),
    items: wp.array(dtype=int),
    origin: wp.vec2,
    cell: float,
    nx: int,
    ny: int,
    extent_low: wp.array(dtype=float),
    extent_high: wp.array(dtype=float),
) -> float:
    """Nearest distance below `limit` where the ray meets a primitive whose owner spans its height."""
    best = float(limit)
    if nx == 0:
        return best
    ray = direction * reach
    low = origin
    high = origin + wp.vec2(float(nx) * cell, float(ny) * cell)
    enter = wp.float32(0.0)
    leave = float(limit)
    for axis in range(2):
        if wp.abs(direction[axis]) < 1.0e-12:
            if o[axis] < low[axis] or o[axis] > high[axis]:
                return best
        else:
            t1 = (low[axis] - o[axis]) / direction[axis]
            t2 = (high[axis] - o[axis]) / direction[axis]
            enter = wp.max(enter, wp.min(t1, t2))
            leave = wp.min(leave, wp.max(t1, t2))
    if enter > leave:
        return best
    p = o + direction * enter
    cx = wp.clamp(int(wp.floor((p[0] - origin[0]) / cell)), 0, nx - 1)
    cy = wp.clamp(int(wp.floor((p[1] - origin[1]) / cell)), 0, ny - 1)
    step_x = wp.int32(1)
    step_y = wp.int32(1)
    if direction[0] < 0.0:
        step_x = -1
    if direction[1] < 0.0:
        step_y = -1
    next_x = wp.float32(1.0e30)
    next_y = wp.float32(1.0e30)
    delta_x = wp.float32(1.0e30)
    delta_y = wp.float32(1.0e30)
    if wp.abs(direction[0]) >= 1.0e-12:
        bound = origin[0] + float(cx + wp.max(step_x, 0)) * cell
        next_x = (bound - o[0]) / direction[0]
        delta_x = cell / wp.abs(direction[0])
    if wp.abs(direction[1]) >= 1.0e-12:
        bound = origin[1] + float(cy + wp.max(step_y, 0)) * cell
        next_y = (bound - o[1]) / direction[1]
        delta_y = cell / wp.abs(direction[1])
    t = float(enter)
    while t <= leave and t < best and cx >= 0 and cx < nx and cy >= 0 and cy < ny:
        c = cy * nx + cx
        for item in range(start[c], start[c + 1]):
            q = items[item]
            if first[q] > lv or last[q] < lv or owner[q] == caster:
                continue
            f = _hit(kind[q], pa[q], pb[q], o, ray)
            if f < 0.0:
                continue
            d = f * reach
            rise = d * slope + base
            if d < best and extent_low[owner[q]] <= rise and rise <= extent_high[owner[q]]:
                best = d
        if next_x < next_y:
            t = next_x
            next_x = next_x + delta_x
            cx = cx + step_x
        else:
            t = next_y
            next_y = next_y + delta_y
            cy = cy + step_y
    return best


@wp.kernel
def _first(
    ray_fan: wp.array(dtype=int),
    ray_beam: wp.array(dtype=int),
    ray_slope: wp.array(dtype=float),
    fan_origin: wp.array(dtype=wp.vec2),
    fan_theta: wp.array(dtype=wp.float64),
    fan_increment: wp.array(dtype=wp.float64),
    fan_reach: wp.array(dtype=float),
    fan_level: wp.array(dtype=int),
    fan_caster: wp.array(dtype=int),
    fan_base: wp.array(dtype=float),
    fan_floor: wp.array(dtype=float),
    extent_low: wp.array(dtype=float),
    extent_high: wp.array(dtype=float),
    s_kind: wp.array(dtype=int),
    s_a: wp.array(dtype=wp.vec2),
    s_b: wp.array(dtype=wp.vec2),
    s_first: wp.array(dtype=int),
    s_last: wp.array(dtype=int),
    s_owner: wp.array(dtype=int),
    s_start: wp.array(dtype=int),
    s_items: wp.array(dtype=int),
    s_origin: wp.vec2,
    s_cell: float,
    s_nx: int,
    s_ny: int,
    m_kind: wp.array(dtype=int),
    m_a: wp.array(dtype=wp.vec2),
    m_b: wp.array(dtype=wp.vec2),
    m_first: wp.array(dtype=int),
    m_last: wp.array(dtype=int),
    m_owner: wp.array(dtype=int),
    m_start: wp.array(dtype=int),
    m_items: wp.array(dtype=int),
    m_origin: wp.vec2,
    m_cell: float,
    m_nx: int,
    m_ny: int,
    out: wp.array(dtype=float),
):
    r = wp.tid()
    f = ray_fan[r]
    angle = fan_theta[f] + fan_increment[f] * wp.float64(ray_beam[r])
    direction = wp.vec2(float(wp.cos(angle)), float(wp.sin(angle)))
    slope = ray_slope[r]
    reach = fan_reach[f]
    o = fan_origin[f]
    floor = fan_floor[f]
    ground = wp.float32(1.0e30)
    if slope < 0.0 and floor < 0.0:
        ground = floor / slope
    limit = wp.min(ground, reach * (1.0 + 1.0e-6))
    best = _walk_first(o, direction, reach, limit, slope, fan_base[f], fan_level[f], fan_caster[f], s_kind, s_a, s_b, s_first, s_last, s_owner, s_start, s_items, s_origin, s_cell, s_nx, s_ny, extent_low, extent_high)
    best = _walk_first(o, direction, reach, best, slope, fan_base[f], fan_level[f], fan_caster[f], m_kind, m_a, m_b, m_first, m_last, m_owner, m_start, m_items, m_origin, m_cell, m_nx, m_ny, extent_low, extent_high)
    if best >= limit:
        best = ground
    if best > 1.0e29:
        best = wp.inf
    out[r] = best


@wp.kernel
def _cast(
    beam_fan: wp.array(dtype=int),
    fan_first_beam: wp.array(dtype=int),
    fan_origin: wp.array(dtype=wp.vec2),
    fan_theta: wp.array(dtype=wp.float64),
    fan_increment: wp.array(dtype=wp.float64),
    fan_reach: wp.array(dtype=float),
    fan_level: wp.array(dtype=int),
    fan_caster: wp.array(dtype=int),
    s_kind: wp.array(dtype=int),
    s_a: wp.array(dtype=wp.vec2),
    s_b: wp.array(dtype=wp.vec2),
    s_first: wp.array(dtype=int),
    s_last: wp.array(dtype=int),
    s_owner: wp.array(dtype=int),
    s_start: wp.array(dtype=int),
    s_items: wp.array(dtype=int),
    s_origin: wp.vec2,
    s_cell: float,
    s_nx: int,
    s_ny: int,
    m_kind: wp.array(dtype=int),
    m_a: wp.array(dtype=wp.vec2),
    m_b: wp.array(dtype=wp.vec2),
    m_first: wp.array(dtype=int),
    m_last: wp.array(dtype=int),
    m_owner: wp.array(dtype=int),
    m_start: wp.array(dtype=int),
    m_items: wp.array(dtype=int),
    m_origin: wp.vec2,
    m_cell: float,
    m_nx: int,
    m_ny: int,
    static_count: int,
    out_distance: wp.array2d(dtype=float),
    out_owner: wp.array2d(dtype=int),
    out_prim: wp.array2d(dtype=int),
    out_count: wp.array(dtype=int),
):
    b = wp.tid()
    f = beam_fan[b]
    k = b - fan_first_beam[f]
    angle = fan_theta[f] + fan_increment[f] * wp.float64(k)
    direction = wp.vec2(float(wp.cos(angle)), float(wp.sin(angle)))
    out_count[b] = 0
    o = fan_origin[f]
    _walk(b, o, direction, fan_reach[f], fan_level[f], fan_caster[f], 0, s_kind, s_a, s_b, s_first, s_last, s_owner, s_start, s_items, s_origin, s_cell, s_nx, s_ny, out_distance, out_owner, out_prim, out_count)
    _walk(b, o, direction, fan_reach[f], fan_level[f], fan_caster[f], static_count, m_kind, m_a, m_b, m_first, m_last, m_owner, m_start, m_items, m_origin, m_cell, m_nx, m_ny, out_distance, out_owner, out_prim, out_count)


@attrs.frozen
class _Grid:
    """Primitives (two-sided or one-sided segments a to b, circles at a of radius b.x) bucketed by cell, each with its levels and owner."""

    kind: np.ndarray
    a: np.ndarray
    b: np.ndarray
    first: np.ndarray
    last: np.ndarray
    owner: np.ndarray
    origin: np.ndarray
    cell: float
    nx: int
    ny: int
    start: np.ndarray
    items: np.ndarray
    first_cell: np.ndarray


def _grid(kind: np.ndarray, a: np.ndarray, b: np.ndarray, first: np.ndarray, last: np.ndarray, owner: np.ndarray, cell: float) -> _Grid:
    """Bucket primitives by cell, cells widening until the longer side has MAX_GRID_SIDE of them."""
    circle = kind == _CIRCLE
    low = np.where(circle[:, None], a - b[:, :1], np.minimum(a, b)) - GRID_PAD
    high = np.where(circle[:, None], a + b[:, :1], np.maximum(a, b)) + GRID_PAD
    if not len(kind):
        return _Grid(kind, a, b, first, last, owner, np.zeros(2), cell, 0, 0, np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros((0, 2), dtype=np.int32))
    cell = max(cell, float((high.max(axis=0) - low.min(axis=0)).max()) / MAX_GRID_SIDE)
    origin = low.min(axis=0) - cell
    lo = np.floor((low - origin) / cell).astype(np.int64)
    hi = np.floor((high - origin) / cell).astype(np.int64)
    nx, ny = (hi.max(axis=0) + 2).tolist()
    width = hi[:, 0] - lo[:, 0] + 1
    counts = width * (hi[:, 1] - lo[:, 1] + 1)
    prim = np.repeat(np.arange(len(kind)), counts)
    k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    cells = (np.repeat(lo[:, 1], counts) + k // np.repeat(width, counts)) * nx + np.repeat(lo[:, 0], counts) + k % np.repeat(width, counts)
    order = np.argsort(cells, kind='stable')
    start = np.zeros(nx * ny + 1, dtype=np.int32)
    start[1:] = np.cumsum(np.bincount(cells, minlength=nx * ny))
    return _Grid(kind, a, b, first, last, owner, origin, cell, int(nx), int(ny), start, prim[order].astype(np.int32), lo.astype(np.int32))


def _edges(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end points of the edges of (n, k, 2) rings."""
    return corners.reshape(-1, 2), np.roll(corners, -1, axis=1).reshape(-1, 2)


def _joined(rows: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(rows).astype(dtype) if rows else np.zeros(0, dtype=dtype)


def _rotation(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)


def _turn(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectors `v` (..., 2) rotated by unit complex numbers `q` (..., 2)."""
    return np.stack([q[..., 0] * v[..., 0] - q[..., 1] * v[..., 1], q[..., 1] * v[..., 0] + q[..., 0] * v[..., 1]], axis=-1)


@attrs.define
class _Entity:
    z: tuple[float, float]
    levels: range
    segments: np.ndarray | None = None
    half: Point | None = None
    radius: float = 0.0
    pose: Pose2D = (0.0, 0.0, 0.0)
    velocity: Point = (0.0, 0.0)


@attrs.define
class _Robot:
    level: int
    z: float
    height: float
    drive: DriveModel
    mass: _RobotMass
    corners: np.ndarray | None
    radius: float


_ROW_WIDTHS = {'com': 2, 'rot': 2, 'vel': 2, 'spin': 1, 'center': 2, 'command': 3, 'command_time': 1, 'setpoint': 3, 'twist': 3, 'acceleration': 2}
_FLOAT = wp.float32


def _slew(current: np.ndarray, target: np.ndarray, step: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(step), current + np.clip(target - current, -step, step), target)


def _follow(twist: np.ndarray, setpoint: np.ndarray, gain: np.ndarray, lag: np.ndarray, traction: np.ndarray, pivot: np.ndarray, dt: float) -> np.ndarray:
    """`DriveModel.follow` over rows of robots."""
    turn = setpoint[:, 2] * gain[:, 2]
    target = np.stack([setpoint[:, 0] * gain[:, 0], setpoint[:, 1] * gain[:, 1] - pivot * turn, turn], axis=1)
    lagging = lag > 0.0
    blend = 1.0 - np.exp(-dt / np.where(lagging, lag, 1.0))
    target = np.where(lagging[:, None], twist + (target - twist) * blend[:, None], target)
    delta = target[:, :2] - twist[:, :2]
    change = np.hypot(delta[:, 0], delta[:, 1])
    moving = change > 0.0
    scale = np.where(moving, np.minimum(1.0, traction * dt / np.where(moving, change, 1.0)), 0.0)
    return np.column_stack([twist[:, :2] + delta * scale[:, None], target[:, 2]])


class WarpWorld:
    """Static outlines, kinematic boxes and discs, velocity-driven robots, stepped and cast on a Warp device."""

    def __init__(self, dt: float, dynamics: Dynamics = Dynamics.RIGID, device: str | None = None) -> None:
        if dt <= 0:
            raise ValueError(f'dt must be positive, got {dt}')
        wp.init()
        self._device = wp.get_device(device) if device is not None else wp.get_preferred_device()
        self._dt = dt
        self._dynamics = dynamics
        self._levels: list[float] = []
        self._entities: dict[str, _Entity] = {}
        self._robots: dict[str, _Robot] = {}
        self._uids: dict[str, int] = {}
        self._names: dict[int, str] = {}
        self._next_uid = 0
        self._version = 0
        self._steps = 0
        self.time = 0.0
        self._index: dict[str, int] = {}
        self._state = {key: np.zeros((0, width)) for key, width in _ROW_WIDTHS.items()}
        self._shapes_dirty = True
        self._poses_dirty = True
        self._drive: dict[str, np.ndarray] = {}
        self._bodies: dict[str, wp.array] = {}
        self._hash = wp.HashGrid(128, 128, 1, device=self._device)
        self._peer_reach = 0.0
        self._static: _Grid | None = None
        self._static_arrays: dict[str, wp.array] = {}
        self._moving: tuple[tuple[int, int], _Grid, dict[str, wp.array]] | None = None
        self._movers_dirty = True
        self._mover_index: dict[str, int] = {}
        self._movers: dict[str, np.ndarray] = {}
        self._boxes: dict[str, wp.array] = {}
        self._out: dict[str, wp.array] = {}
        self._scratch: dict[str, wp.array] = {}
        self._corners: dict[str, np.ndarray] = {}
        self._extents: tuple[np.ndarray, np.ndarray] | None = None
        self._rays: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dynamics(self) -> Dynamics:
        return self._dynamics

    @property
    def device(self) -> str:
        return str(self._device)

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
        for entity in self._entities.values():
            entity.levels = self._span(entity.z)
        for robot in self._robots.values():
            robot.level = self.level(robot.z)
        self._static = None
        self._movers_dirty = True
        self._poses_dirty = True
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
        self._extents = None

    def _disown(self, name: str) -> None:
        del self._names[self._uids.pop(name)]
        self._version += 1
        self._extents = None

    def _extent(self, name: str) -> tuple[float, float]:
        entity = self._entities.get(name)
        if entity is not None:
            return entity.z
        robot = self._robots[name]
        return robot.z, robot.z + robot.height

    def _extent_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Lowest and highest z of every owner by uid."""
        if self._extents is None:
            low, high = np.zeros(self._next_uid), np.zeros(self._next_uid)
            for uid, name in self._names.items():
                low[uid], high[uid] = self._extent(name)
            self._extents = low, high
        return self._extents

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
        self._entities[name] = _Entity(z, self._span(z), segments=np.array(segments, dtype=np.float64).reshape(-1, 2, 2))
        self._own(name)
        self._static = None

    def add_box(self, name: str, size: Point, pose: Pose2D, z: tuple[float, float]) -> None:
        """Kinematic box of full extents `size`, centered on `pose`."""
        self.remove(name)
        self._entities[name] = _Entity(z, self._span(z), half=(size[0] / 2.0, size[1] / 2.0), pose=(float(pose[0]), float(pose[1]), float(pose[2])))
        self._own(name)
        self._movers_dirty = True

    def add_disc(self, name: str, radius: float, position: Point, z: tuple[float, float]) -> None:
        """Kinematic disc, the shape of a pedestrian, touching robots without pushing them."""
        self.remove(name)
        self._entities[name] = _Entity(z, self._span(z), radius=float(radius), pose=(float(position[0]), float(position[1]), 0.0))
        self._own(name)
        self._movers_dirty = True

    def move(self, name: str, pose: Pose2D, velocity: Point = (0.0, 0.0), z: tuple[float, float] | None = None) -> None:
        """Place a box or disc and set the velocity it keeps until the next move."""
        entity = self._entities[name]
        entity.pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        entity.velocity = (float(velocity[0]), float(velocity[1]))
        row = self._mover_index.get(name)
        if row is not None:
            self._movers['position'][row] = entity.pose[:2]
            self._movers['yaw'][row] = entity.pose[2]
            self._movers['velocity'][row] = entity.velocity
        self._version += 1
        if z is not None and z != entity.z:
            entity.z = z
            entity.levels = self._span(z)
            self._movers_dirty = True
            self._extents = None
            if entity.segments is not None:
                self._static = None

    def remove(self, name: str) -> bool:
        entity = self._entities.pop(name, None)
        if entity is not None:
            self._disown(name)
            if entity.segments is not None:
                self._static = None
            else:
                self._mover_index.pop(name, None)
                self._movers_dirty = True
            return True
        if name in self._robots:
            i = self._index[name]
            del self._robots[name]
            self._state = {key: np.delete(rows, i, axis=0) for key, rows in self._state.items()}
            self._index = {robot: row for row, robot in enumerate(self._robots)}
            self._disown(name)
            self._shapes_dirty = True
            return True
        return False

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
        mass = _robot_mass(footprint, float(radius))
        self._robots[name] = _Robot(level=self.level(z), z=z, height=height, drive=drive, mass=mass, corners=_ccw(mass.vertices) if mass.vertices is not None else None, radius=float(radius))
        self._index[name] = len(self._index)
        q = _rotation(np.array(float(pose[2])))
        center = np.array(mass.center) if self._dynamics is Dynamics.RIGID else np.zeros(2)
        row = {'com': np.array(pose[:2]) + _turn(q, center), 'rot': q, 'center': center, 'command_time': np.array([-math.inf])}
        self._state = {key: np.vstack([rows, row.get(key, np.zeros(_ROW_WIDTHS[key]))]) for key, rows in self._state.items()}
        self._own(name)
        self._shapes_dirty = True

    def set_robot_pose(self, name: str, pose: Pose2D, z: float) -> None:
        """Teleport a robot to rest, switching level when z crosses a floor."""
        robot = self._robots[name]
        i = self._index[name]
        state = self._state
        q = _rotation(np.array(float(pose[2])))
        state['rot'][i] = q
        state['com'][i] = np.array(pose[:2]) + _turn(q, state['center'][i])
        for key in ('vel', 'spin', 'command', 'setpoint', 'twist', 'acceleration'):
            state[key][i] = 0.0
        robot.z = z
        robot.level = self.level(z)
        self._poses_dirty = True
        self._extents = None
        self._version += 1

    def command(self, name: str, twist: Twist2D) -> None:
        """Body-frame velocity target, held for the drive timeout of world time."""
        if not all(math.isfinite(v) for v in twist):
            return
        i = self._index[name]
        self._state['command'][i] = twist
        self._state['command_time'][i] = self.time

    def robot_state(self, name: str) -> RobotState:
        robot = self._robots[name]
        i = self._index[name]
        state = self._state
        q = state['rot'][i]
        origin = state['com'][i] - _turn(q, state['center'][i])
        vx, vy, wz = state['twist'][i].tolist()
        ax, ay = state['acceleration'][i].tolist()
        return RobotState(float(origin[0]), float(origin[1]), robot.z, math.atan2(q[1], q[0]), vx, vy, wz, ax, ay)

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

    def _array(self, values: np.ndarray, dtype: type) -> wp.array:
        return wp.array(values, dtype=dtype, device=self._device)

    def _zeros(self, count: int, dtype: type, shape: int | tuple[int, int] | None = None) -> wp.array:
        return wp.zeros(count if shape is None else shape, dtype=dtype, device=self._device)

    def _reuse(self, key: str, values: np.ndarray, dtype: type) -> wp.array:
        """Device array `key` holding `values`, reallocated only when their length changes."""
        array = self._scratch.get(key)
        if array is None or array.shape[0] != len(values):
            array = self._scratch[key] = self._array(values, dtype)
        elif len(values):
            array.assign(values)
        return array

    def _pack_movers(self) -> None:
        """Host rows of every box and disc, the positions in them carried over."""
        if not self._movers_dirty:
            return
        for name, row in self._mover_index.items():
            entity = self._entities.get(name)
            if entity is not None:
                x, y = self._movers['position'][row].tolist()
                entity.pose = (x, y, entity.pose[2])
        rows = [(name, entity) for name, entity in self._entities.items() if entity.segments is None]
        self._mover_index = {name: row for row, (name, _) in enumerate(rows)}
        self._movers = {
            'position': np.array([entity.pose[:2] for _, entity in rows], dtype=np.float64).reshape(-1, 2),
            'yaw': np.array([entity.pose[2] for _, entity in rows], dtype=np.float64),
            'velocity': np.array([entity.velocity for _, entity in rows], dtype=np.float64).reshape(-1, 2),
            'half': np.array([entity.half or (0.0, 0.0) for _, entity in rows], dtype=np.float64).reshape(-1, 2),
            'radius': np.array([entity.radius for _, entity in rows], dtype=np.float64),
            'first': np.array([entity.levels.start for _, entity in rows], dtype=np.int32),
            'last': np.array([entity.levels.stop - 1 for _, entity in rows], dtype=np.int32),
            'uid': np.array([self._uids[name] for name, _ in rows], dtype=np.int32),
            'box': np.array([entity.half is not None for _, entity in rows], dtype=bool),
        }
        boxes = self._movers['box']
        half = self._movers['half'][boxes]
        self._boxes = {
            'center': wp.zeros(len(half), dtype=wp.vec2, device=self._device),
            'rot': wp.zeros(len(half), dtype=wp.vec2, device=self._device),
            'velocity': wp.zeros(len(half), dtype=wp.vec2, device=self._device),
            'half': self._array(half.astype(np.float32), wp.vec2),
            'extent': self._array(np.hypot(half[:, 0], half[:, 1]).astype(np.float32), wp.float32),
            'first': self._array(self._movers['first'][boxes], wp.int32),
            'last': self._array(self._movers['last'][boxes], wp.int32),
            'world_v': wp.zeros(len(half) * 4, dtype=wp.vec2, device=self._device),
            'world_n': wp.zeros(len(half) * 4, dtype=wp.vec2, device=self._device),
        }
        self._movers_dirty = False

    def _static_grid(self) -> _Grid:
        """Outline segments, rebuilt only when an outline or the levels change."""
        if self._static is not None:
            return self._static
        rows = [(walls, entity.levels, self._uids[name]) for name, entity in self._entities.items() if (walls := entity.segments) is not None and len(walls)]
        segments = np.concatenate([walls for walls, _, _ in rows]) if rows else np.zeros((0, 2, 2))
        counts = [len(walls) for walls, _, _ in rows]
        first = np.repeat(np.array([levels.start for _, levels, _ in rows], dtype=np.int32), counts)
        last = np.repeat(np.array([levels.stop - 1 for _, levels, _ in rows], dtype=np.int32), counts)
        owner = np.repeat(np.array([uid for _, _, uid in rows], dtype=np.int32), counts)
        a, b = segments[:, 0], segments[:, 1]
        grid = _grid(np.full(len(a), _TWO_SIDED, dtype=np.int32), a, b, first, last, owner, STATIC_CELL)
        edge = b - a
        normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / np.maximum(np.hypot(edge[:, 0], edge[:, 1]), 1e-12)[:, None]
        self._static_arrays = {
            **self._grid_arrays(grid),
            'cell_x': self._array(grid.first_cell[:, 0], wp.int32),
            'cell_y': self._array(grid.first_cell[:, 1], wp.int32),
            'seg_v': self._array(segments.reshape(-1, 2).astype(np.float32), wp.vec2),
            'seg_n': self._array(np.stack([normal, -normal], axis=1).reshape(-1, 2).astype(np.float32), wp.vec2),
        }
        self._static = grid
        return grid

    def _grid_arrays(self, grid: _Grid) -> dict[str, wp.array]:
        return {
            'kind': self._array(grid.kind, wp.int32),
            'a': self._array(grid.a.astype(np.float32), wp.vec2),
            'b': self._array(grid.b.astype(np.float32), wp.vec2),
            'first': self._array(grid.first, wp.int32),
            'last': self._array(grid.last, wp.int32),
            'owner': self._array(grid.owner, wp.int32),
            'start': self._array(grid.start, wp.int32),
            'items': self._array(grid.items, wp.int32),
        }

    def _build_bodies(self) -> None:
        """Device copies of the robots: shapes when they changed, poses when the host moved them."""
        count = len(self._robots)
        if self._shapes_dirty:
            robots = list(self._robots.values())
            local_v = np.zeros((count, MAX_VERTS, 2))
            local_n = np.zeros((count, MAX_VERTS, 2))
            for row, robot in enumerate(robots):
                vertices, normals = robot.mass.vertices, robot.mass.normals
                if vertices is not None and normals is not None:
                    local_v[row, : len(vertices)] = vertices
                    local_n[row, : len(normals)] = normals
            verts = np.array([0 if robot.mass.vertices is None else len(robot.mass.vertices) for robot in robots], dtype=np.int32)
            center = self._state['center']
            extent = np.array([robot.radius if robot.mass.vertices is None else float(np.hypot(*(robot.mass.vertices - center[row]).T).max()) for row, robot in enumerate(robots)])
            self._peer_reach = 2.0 * float(extent.max(initial=0.0)) + 0.5
            zeros = functools.partial(self._zeros, count)
            self._bodies = {
                'center': self._array(center.astype(np.float32), wp.vec2),
                'local_v': self._array(local_v.reshape(-1, 2).astype(np.float32), wp.vec2),
                'local_n': self._array(local_n.reshape(-1, 2).astype(np.float32), wp.vec2),
                'count': self._array(verts, wp.int32),
                'kind': self._array((verts == 0).astype(np.int32), wp.int32),
                'radius': self._array(np.array([robot.radius for robot in robots], dtype=np.float32), wp.float32),
                'extent': self._array(extent.astype(np.float32), wp.float32),
                'inv_mass': self._array(np.array([1.0 / robot.mass.mass if robot.mass.mass > 0.0 else 0.0 for robot in robots], dtype=np.float32), wp.float32),
                'inv_inertia': self._array(np.array([1.0 / robot.mass.inertia if robot.mass.inertia > 0.0 else 0.0 for robot in robots], dtype=np.float32), wp.float32),
                'level': zeros(wp.int32),
                'com': zeros(wp.vec2),
                'rot': zeros(wp.vec2),
                'vel': zeros(wp.vec2),
                'spin': zeros(wp.float32),
                'vel_b': zeros(wp.vec2),
                'spin_b': zeros(wp.float32),
                'com0': zeros(wp.vec2),
                'rot0': zeros(wp.vec2),
                'world_v': zeros(wp.vec2, count * MAX_VERTS),
                'world_n': zeros(wp.vec2, count * MAX_VERTS),
                'points': zeros(wp.vec3),
                'c_count': zeros(wp.int32),
                'c_kind': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'c_other': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'c_normal': zeros(wp.vec2, (count, MAX_MANIFOLDS)),
                'c_points': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'c_ra': zeros(wp.vec2, (count, MAX_MANIFOLDS * 2)),
                'c_rb': zeros(wp.vec2, (count, MAX_MANIFOLDS * 2)),
                'c_sep': zeros(wp.float32, (count, MAX_MANIFOLDS * 2)),
                'c_mass': zeros(wp.float32, (count, MAX_MANIFOLDS * 2)),
                'c_impulse': zeros(wp.float32, (count, MAX_MANIFOLDS * 2)),
                'p_count': zeros(wp.int32),
                'p_kind': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'p_other': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'p_points': zeros(wp.int32, (count, MAX_MANIFOLDS)),
                'p_ra': zeros(wp.vec2, (count, MAX_MANIFOLDS * 2)),
                'p_impulse': zeros(wp.float32, (count, MAX_MANIFOLDS * 2)),
            }
            drives = [robot.drive for robot in robots]
            axes = [(drive.linear, drive.lateral, drive.angular) for drive in drives]
            self._drive = {
                'low': np.array([[axis.low for axis in row] for row in axes]).reshape(-1, 3),
                'high': np.array([[axis.high for axis in row] for row in axes]).reshape(-1, 3),
                'accel': np.array([[axis.accel for axis in row] for row in axes]).reshape(-1, 3),
                'gain': np.array([drive.gain for drive in drives]).reshape(-1, 3),
                'lag': np.array([drive.lag for drive in drives]),
                'traction': np.array([drive.traction for drive in drives]),
                'pivot': np.array([drive.pivot for drive in drives]),
                'timeout': np.array([drive.timeout for drive in drives]),
            }
            pad = np.zeros((count, MAX_VERTS, 2))
            sides = np.zeros(count, dtype=np.int64)
            for row, robot in enumerate(robots):
                if robot.corners is not None:
                    sides[row] = len(robot.corners)
                    pad[row, : sides[row]] = robot.corners
            slots = np.arange(MAX_VERTS)
            self._corners = {
                'pad': pad,
                'used': slots < sides[:, None],
                'next': (slots + 1) % np.maximum(sides[:, None], 1),
                'disc': sides == 0,
                'radius': np.array([robot.radius for robot in robots]),
                'uid': np.array([self._uids[name] for name in self._robots], dtype=np.int32),
            }
            self._shapes_dirty = False
            self._poses_dirty = True
        if self._poses_dirty:
            self._bodies['com'].assign(self._state['com'].astype(np.float32))
            self._bodies['rot'].assign(self._state['rot'].astype(np.float32))
            self._bodies['level'].assign(np.array([robot.level for robot in self._robots.values()], dtype=np.int32))
            self._poses_dirty = False

    def _step(self, dt: float) -> None:
        self._pack_movers()
        if self._robots:
            self._step_robots(dt)
        if len(self._movers['position']):
            self._movers['position'] += self._movers['velocity'] * dt
        self._steps += 1

    def _step_robots(self, dt: float) -> None:
        self._build_bodies()
        state, drive, bodies = self._state, self._drive, self._bodies
        before = state['twist'].copy()
        rot = state['rot']
        cos, sin = rot[:, 0], rot[:, 1]
        command = np.where((self.time - state['command_time'] <= drive['timeout'][:, None]), state['command'], 0.0)
        state['setpoint'] = _slew(state['setpoint'], np.clip(command, drive['low'], drive['high']), drive['accel'] * dt)
        twist = _follow(before, state['setpoint'], drive['gain'], drive['lag'], drive['traction'], drive['pivot'], dt)
        state['vel'] = np.stack([cos * twist[:, 0] - sin * twist[:, 1], sin * twist[:, 0] + cos * twist[:, 1]], axis=1)
        limit = MAX_ROTATION / dt
        state['spin'] = np.clip(twist[:, 2:], -limit, limit)
        speed = np.hypot(state['vel'][:, 0], state['vel'][:, 1])
        state['vel'] = state['vel'] * (MAX_LINEAR_SPEED / np.maximum(speed, MAX_LINEAR_SPEED))[:, None]
        state['twist'] = twist
        bodies['vel'].assign(state['vel'].astype(np.float32))
        bodies['spin'].assign(state['spin'].ravel().astype(np.float32))
        rigid = self._dynamics is Dynamics.RIGID
        count = len(self._robots)
        device = self._device
        contacts = [bodies[key] for key in ('c_count', 'c_kind', 'c_other', 'c_normal', 'c_points', 'c_ra', 'c_rb', 'c_sep', 'c_mass', 'c_impulse')]
        boxes = self._boxes
        if rigid:
            grid = self._static_grid()
            statics = self._static_arrays
            boxes_rows = self._movers['box']
            boxes['center'].assign(self._movers['position'][boxes_rows].astype(np.float32))
            boxes['rot'].assign(_rotation(self._movers['yaw'][boxes_rows]).astype(np.float32))
            boxes['velocity'].assign(self._movers['velocity'][boxes_rows].astype(np.float32))
            wp.launch(_place_boxes, dim=len(boxes['half']), inputs=[boxes['center'], boxes['rot'], boxes['half'], boxes['world_v'], boxes['world_n']], device=device)
            wp.launch(_place_robots, dim=count, inputs=[bodies['com'], bodies['rot'], bodies['center'], bodies['local_v'], bodies['local_n'], bodies['count'], bodies['world_v'], bodies['world_n'], bodies['points']], device=device)
            self._hash.build(bodies['points'], self._peer_reach)
            for key in ('count', 'kind', 'other', 'points', 'ra', 'impulse'):
                wp.copy(bodies[f'p_{key}'], bodies[f'c_{key}'])
            wp.launch(
                _collide,
                dim=count,
                inputs=[
                    dt,
                    bodies['com'],
                    bodies['vel'],
                    bodies['level'],
                    bodies['kind'],
                    bodies['count'],
                    bodies['radius'],
                    bodies['extent'],
                    bodies['inv_mass'],
                    bodies['inv_inertia'],
                    bodies['world_v'],
                    bodies['world_n'],
                    self._hash.id,
                    self._peer_reach if count > 1 else 0.0,
                    statics['seg_v'],
                    statics['seg_n'],
                    statics['first'],
                    statics['last'],
                    statics['cell_x'],
                    statics['cell_y'],
                    statics['start'],
                    statics['items'],
                    wp.vec2(*grid.origin.astype(np.float32).tolist()),
                    grid.cell,
                    grid.nx,
                    grid.ny,
                    boxes['center'],
                    boxes['extent'],
                    boxes['first'],
                    boxes['last'],
                    boxes['world_v'],
                    boxes['world_n'],
                    *contacts,
                ],
                device=device,
            )
            wp.launch(_warm, dim=count, inputs=[*(bodies[f'c_{key}'] for key in ('count', 'kind', 'other', 'points', 'ra', 'impulse')), *(bodies[f'p_{key}'] for key in ('count', 'kind', 'other', 'points', 'ra', 'impulse'))], device=device)
        wp.copy(bodies['com0'], bodies['com'])
        wp.copy(bodies['rot0'], bodies['rot'])
        h = dt / _SUBSTEPS
        hertz = min(CONTACT_HERTZ, 0.25 / h)
        now = (bodies['vel'], bodies['spin'])
        then = (bodies['vel_b'], bodies['spin_b'])
        mass = (bodies['inv_mass'], bodies['inv_inertia'])
        pose = (bodies['com'], bodies['rot'], bodies['com0'], bodies['rot0'])
        warm = [bodies[key] for key in ('c_count', 'c_normal', 'c_points', 'c_ra', 'c_impulse')]
        for k in range(_SUBSTEPS):
            if rigid:
                wp.launch(_warm_start, dim=count, inputs=[*now, *then, *mass, *warm], device=device)
                now, then = then, now
                wp.launch(_solve, dim=count, inputs=[h, hertz, k * h, True, *now, *then, *pose, *mass, boxes['velocity'], *contacts], device=device)
                now, then = then, now
            wp.launch(_integrate_bodies, dim=count, inputs=[h, *now, bodies['com'], bodies['rot']], device=device)
            if rigid:
                wp.launch(_solve, dim=count, inputs=[h, hertz, (k + 1) * h, False, *now, *then, *pose, *mass, boxes['velocity'], *contacts], device=device)
                now, then = then, now
        state['com'] = bodies['com'].numpy().astype(np.float64)
        state['rot'] = bodies['rot'].numpy().astype(np.float64)
        state['vel'] = now[0].numpy().astype(np.float64)
        state['spin'] = now[1].numpy().astype(np.float64)[:, None]
        if rigid:
            velocity = state['vel']
            twist = np.column_stack([cos * velocity[:, 0] + sin * velocity[:, 1], -sin * velocity[:, 0] + cos * velocity[:, 1], state['spin']])
            state['twist'] = twist
        state['acceleration'] = np.stack([(twist[:, 0] - before[:, 0]) / dt - twist[:, 2] * twist[:, 1], (twist[:, 1] - before[:, 1]) / dt + twist[:, 2] * twist[:, 0]], axis=1)

    def _moving_grid(self) -> tuple[_Grid, dict[str, wp.array]]:
        """Boxes, discs and robots where they are now, rebuilt once per step or change."""
        key = (self._steps, self._version)
        if self._moving is not None and self._moving[0] == key:
            return self._moving[1], self._moving[2]
        self._pack_movers()
        movers, state = self._movers, self._state
        kinds, starts, ends, firsts, lasts, owners = [], [], [], [], [], []
        boxes = movers['box']
        half = movers['half'][boxes]
        if len(half):
            corners = half[:, None] * np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
            placed = _turn(_rotation(movers['yaw'][boxes])[:, None], corners) + movers['position'][boxes][:, None]
            a, b = _edges(placed)
            starts.append(a)
            ends.append(b)
            kinds.append(np.full(len(a), _ONE_SIDED))
            firsts.append(np.repeat(movers['first'][boxes], 4))
            lasts.append(np.repeat(movers['last'][boxes], 4))
            owners.append(np.repeat(movers['uid'][boxes], 4))
        discs = ~boxes
        if discs.any():
            starts.append(movers['position'][discs])
            ends.append(np.column_stack([movers['radius'][discs], np.zeros(int(discs.sum()))]))
            kinds.append(np.full(int(discs.sum()), _CIRCLE))
            firsts.append(movers['first'][discs])
            lasts.append(movers['last'][discs])
            owners.append(movers['uid'][discs])
        if self._robots:
            self._build_bodies()
            origin = state['com'] - _turn(state['rot'], state['center'])
            placed = _turn(state['rot'][:, None], self._corners['pad']) + origin[:, None]
            rows, slots = np.nonzero(self._corners['used'])
            starts.append(placed[rows, slots])
            ends.append(placed[rows, self._corners['next'][rows, slots]])
            kinds.append(np.full(len(rows), _ONE_SIDED))
            levels = np.array([robot.level for robot in self._robots.values()], dtype=np.int32)
            uids = self._corners['uid']
            firsts.append(levels[rows])
            lasts.append(levels[rows])
            owners.append(uids[rows])
            disc = self._corners['disc']
            starts.append(origin[disc])
            ends.append(np.column_stack([self._corners['radius'][disc], np.zeros(int(disc.sum()))]))
            kinds.append(np.full(int(disc.sum()), _CIRCLE))
            firsts.append(levels[disc])
            lasts.append(levels[disc])
            owners.append(uids[disc])
        grid = _grid(_joined(kinds, np.int32), _joined(starts, np.float64).reshape(-1, 2), _joined(ends, np.float64).reshape(-1, 2), _joined(firsts, np.int32), _joined(lasts, np.int32), _joined(owners, np.int32), MOVING_CELL)
        arrays = {
            'kind': self._reuse('m_kind', grid.kind, wp.int32),
            'a': self._reuse('m_a', grid.a.astype(np.float32), wp.vec2),
            'b': self._reuse('m_b', grid.b.astype(np.float32), wp.vec2),
            'first': self._reuse('m_first', grid.first, wp.int32),
            'last': self._reuse('m_last', grid.last, wp.int32),
            'owner': self._reuse('m_owner', grid.owner, wp.int32),
            'start': self._reuse('m_start', grid.start, wp.int32),
            'items': self._reuse('m_items', grid.items, wp.int32),
        }
        self._moving = (key, grid, arrays)
        return grid, arrays

    def first_many(self, requests: Sequence[tuple[str, ScanSpec, np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Horizontal distance to the first hit of each ray of every (robot, fan, beam per ray, rise per meter per ray), in one launch."""
        if not requests:
            return []
        static = self._static_grid()
        moving, moving_arrays = self._moving_grid()
        statics = self._static_arrays
        state = self._state
        rows = np.array([self._index[request[0]] for request in requests])
        specs = [request[1] for request in requests]
        robots = [self._robots[request[0]] for request in requests]
        mounts = np.array([spec.mount for spec in specs]).reshape(-1, 3)
        q = state['rot'][rows]
        origin = state['com'][rows] + _turn(q, mounts[:, :2] - state['center'][rows])
        theta = np.arctan2(q[:, 1], q[:, 0]) + mounts[:, 2] + np.array([spec.angle_min for spec in specs])
        base = np.array([robot.z + spec.height for robot, spec in zip(robots, specs, strict=True)], dtype=np.float32)
        floor = np.array([self._levels[robot.level] if self._levels else 0.0 for robot in robots], dtype=np.float32) - base
        shapes = [request[2].shape for request in requests]
        sizes = np.array([request[2].size for request in requests], dtype=np.int64)
        total = int(sizes.sum())
        rays = [(request[2], request[3]) for request in requests]
        if len(rays) != len(self._rays) or any(new[0] is not old[0] or new[1] is not old[1] for new, old in zip(rays, self._rays, strict=True)):
            self._reuse('ray_fan', np.repeat(np.arange(len(requests)), sizes).astype(np.int32), wp.int32)
            self._reuse('ray_beam', np.concatenate([request[2].ravel() for request in requests]).astype(np.int32), wp.int32)
            self._reuse('ray_slope', np.concatenate([np.broadcast_to(request[3], request[2].shape).ravel() for request in requests]).astype(np.float32), wp.float32)
            self._reuse('ray_out', np.zeros(total, dtype=np.float32), wp.float32)
            self._rays = rays
        low, high = self._extent_table()
        scratch = self._scratch
        wp.launch(
            _first,
            dim=total,
            inputs=[
                scratch['ray_fan'],
                scratch['ray_beam'],
                scratch['ray_slope'],
                self._reuse('fan_origin', origin.astype(np.float32), wp.vec2),
                self._reuse('fan_theta', theta, wp.float64),
                self._reuse('fan_increment', np.array([spec.angle_increment for spec in specs], dtype=np.float64), wp.float64),
                self._reuse('fan_reach', np.array([spec.range_max for spec in specs], dtype=np.float32), wp.float32),
                self._reuse('fan_level', np.array([robot.level for robot in robots], dtype=np.int32), wp.int32),
                self._reuse('fan_caster', np.array([self._uids[request[0]] for request in requests], dtype=np.int32), wp.int32),
                self._reuse('fan_base', base, wp.float32),
                self._reuse('fan_floor', floor, wp.float32),
                self._reuse('extent_low', low.astype(np.float32), wp.float32),
                self._reuse('extent_high', high.astype(np.float32), wp.float32),
                statics['kind'],
                statics['a'],
                statics['b'],
                statics['first'],
                statics['last'],
                statics['owner'],
                statics['start'],
                statics['items'],
                wp.vec2(*static.origin.astype(np.float32).tolist()),
                static.cell,
                static.nx,
                static.ny,
                moving_arrays['kind'],
                moving_arrays['a'],
                moving_arrays['b'],
                moving_arrays['first'],
                moving_arrays['last'],
                moving_arrays['owner'],
                moving_arrays['start'],
                moving_arrays['items'],
                wp.vec2(*moving.origin.astype(np.float32).tolist()),
                moving.cell,
                moving.nx,
                moving.ny,
                scratch['ray_out'],
            ],
            device=self._device,
        )
        out = scratch['ray_out'].numpy().copy()
        ends = np.cumsum(sizes)
        return [out[end - size : end].reshape(shape) for end, size, shape in zip(ends.tolist(), sizes.tolist(), shapes, strict=True)]

    def _cast_fans(self, requests: Sequence[tuple[str, ScanSpec]]) -> list[RayHits]:
        static = self._static_grid()
        moving, moving_arrays = self._moving_grid()
        statics = self._static_arrays
        state = self._state
        fans = len(requests)
        rows = np.array([self._index[name] for name, _ in requests])
        specs = [spec for _, spec in requests]
        robots = [self._robots[name] for name, _ in requests]
        mounts = np.array([spec.mount for spec in specs]).reshape(-1, 3)
        q = state['rot'][rows]
        origin = state['com'][rows] + _turn(q, mounts[:, :2] - state['center'][rows])
        theta = np.arctan2(q[:, 1], q[:, 0]) + mounts[:, 2] + np.array([spec.angle_min for spec in specs])
        level = np.array([robot.level for robot in robots], dtype=np.int32)
        caster = np.array([self._uids[name] for name, _ in requests], dtype=np.int32)
        beams = np.array([spec.beams for spec in specs], dtype=np.int64)
        total = int(beams.sum())
        offset = np.cumsum(beams) - beams
        increment = np.array([spec.angle_increment for _, spec in requests], dtype=np.float64)
        reach = np.array([spec.range_max for _, spec in requests], dtype=np.float32)
        if self._out.get('count') is None or self._out['count'].shape[0] < total:
            size = max(total, 1024)
            self._out = {key: wp.zeros(shape, dtype=dtype, device=self._device) for key, shape, dtype in (('distance', (size, MAX_HITS), wp.float32), ('owner', (size, MAX_HITS), wp.int32), ('prim', (size, MAX_HITS), wp.int32), ('count', size, wp.int32))}
        out = self._out
        wp.launch(
            _cast,
            dim=total,
            inputs=[
                self._reuse('beam_fan', np.repeat(np.arange(fans), beams).astype(np.int32), wp.int32),
                self._reuse('fan_first', offset.astype(np.int32), wp.int32),
                self._reuse('fan_origin', origin.astype(np.float32), wp.vec2),
                self._reuse('fan_theta', theta, wp.float64),
                self._reuse('fan_increment', increment, wp.float64),
                self._reuse('fan_reach', reach, wp.float32),
                self._reuse('fan_level', level, wp.int32),
                self._reuse('fan_caster', caster, wp.int32),
                statics['kind'],
                statics['a'],
                statics['b'],
                statics['first'],
                statics['last'],
                statics['owner'],
                statics['start'],
                statics['items'],
                wp.vec2(*static.origin.astype(np.float32).tolist()),
                static.cell,
                static.nx,
                static.ny,
                moving_arrays['kind'],
                moving_arrays['a'],
                moving_arrays['b'],
                moving_arrays['first'],
                moving_arrays['last'],
                moving_arrays['owner'],
                moving_arrays['start'],
                moving_arrays['items'],
                wp.vec2(*moving.origin.astype(np.float32).tolist()),
                moving.cell,
                moving.nx,
                moving.ny,
                len(static.kind),
                out['distance'],
                out['owner'],
                out['prim'],
                out['count'],
            ],
            device=self._device,
        )
        hits = out['count'].numpy()[:total]
        width = max(1, int(hits.max(initial=0)))
        valid = np.arange(width) < hits[:, None]
        owner = out['owner'].numpy()[:total, :width]
        distance = np.where(valid, out['distance'].numpy()[:total, :width].astype(np.float64), np.inf)
        low, high = self._extent_table()
        base = np.repeat(np.array([robot.z + spec.height for robot, spec in zip(robots, specs, strict=True)]), beams)[:, None]
        lo = np.where(valid, low[owner] - base, np.inf)
        hi = np.where(valid, high[owner] - base, -np.inf)
        results = []
        for f, robot in enumerate(robots):
            beam = slice(int(offset[f]), int(offset[f] + beams[f]))
            floor = self._levels[robot.level] if self._levels else 0.0
            results.append(RayHits(distance[beam], lo[beam], hi[beam], floor - (robot.z + specs[f].height)))
        return results

    def cast(self, name: str, spec: ScanSpec) -> RayHits:
        """Every entity each beam of one fan from robot `name` crosses within range_max."""
        return self._cast_fans([(name, spec)])[0]

    def scan(self, name: str, spec: ScanSpec) -> np.ndarray:
        """Ranges of one fan from robot `name`, inf where no entity spans the sensor height within range_max."""
        return self.cast(name, spec).first().astype(np.float32)

    def contacts(self, name: str) -> list[Touch]:
        """Entities and peers robot `name` touches, each with one contact point and the unit normal out of the robot."""
        robot = self._robots[name]
        i = self._index[name]
        state = self._state
        margin = _CONTACT_MARGIN if self._dynamics is Dynamics.RIGID else 0.0
        q = state['rot'][i]
        origin = state['com'][i] - _turn(q, state['center'][i])
        core, radius = (_turn(q, robot.corners) + origin, 0.0) if robot.corners is not None else (origin[None], robot.radius)
        reach = margin + radius
        low, high = core.min(axis=0) - reach, core.max(axis=0) + reach
        nearest: dict[str, tuple[float, Point, Point]] = {}

        def consider(owner: str, other: np.ndarray, other_radius: float) -> None:
            touch = _touch(core, radius, other, other_radius, margin)
            if touch is not None and (owner not in nearest or touch[0] < nearest[owner][0]):
                nearest[owner] = touch

        grid = self._static_grid()
        if grid.nx:
            lo = np.clip(np.floor((low - grid.origin) / grid.cell).astype(np.int64), 0, (grid.nx - 1, grid.ny - 1))
            hi = np.clip(np.floor((high - grid.origin) / grid.cell).astype(np.int64), 0, (grid.nx - 1, grid.ny - 1))
            cells = np.concatenate([grid.items[grid.start[y * grid.nx + lo[0]] : grid.start[y * grid.nx + hi[0] + 1]] for y in range(lo[1], hi[1] + 1)])
            for s in np.unique(cells):
                if grid.first[s] <= robot.level <= grid.last[s]:
                    consider(self._names[int(grid.owner[s])], np.stack([grid.a[s], grid.b[s]]), 0.0)
        self._pack_movers()
        movers = self._movers
        names = list(self._mover_index)
        if names:
            near = (movers['first'] <= robot.level) & (robot.level <= movers['last'])
            span = np.hypot(movers['half'][:, 0], movers['half'][:, 1]) + movers['radius']
            near &= np.all((movers['position'] + span[:, None] >= low) & (movers['position'] - span[:, None] <= high), axis=1)
            for row in np.flatnonzero(near):
                if movers['box'][row]:
                    half = movers['half'][row]
                    corners = half * np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
                    consider(names[row], _turn(_rotation(movers['yaw'][row]), corners) + movers['position'][row], 0.0)
                else:
                    consider(names[row], movers['position'][row][None], float(movers['radius'][row]))
        for other, peer in self._robots.items():
            if other == name or peer.level != robot.level:
                continue
            j = self._index[other]
            peer_q = state['rot'][j]
            peer_origin = state['com'][j] - _turn(peer_q, state['center'][j])
            peer_core, peer_radius = (_turn(peer_q, peer.corners) + peer_origin, 0.0) if peer.corners is not None else (peer_origin[None], peer.radius)
            if np.all(peer_core.max(axis=0) + peer_radius >= low) and np.all(peer_core.min(axis=0) - peer_radius <= high):
                consider(other, peer_core, peer_radius)
        return [(owner, point, normal) for owner, (_, point, normal) in sorted(nearest.items())]

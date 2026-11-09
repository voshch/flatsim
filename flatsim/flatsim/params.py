"""Node parameters of the flatsim server and the enums they take."""

from __future__ import annotations

import enum

from arena_rclpy_mixins.params import Param, Range


class Dynamics(enum.Enum):
    RIGID = 'rigid'
    KINEMATIC = 'kinematic'


class Physics(enum.Enum):
    AUTO = 'auto'
    BOX2D = 'box2d'
    WARP = 'warp'


PHYSICS_DT = Param[float]('physics_dt', 0.0333, 'Physics step in sim seconds.', constraint=Range(lo=0.0, lo_open=True), read_only=True)
DYNAMICS = Param[Dynamics]('dynamics', Dynamics.RIGID, 'How walls and obstacles act on a robot.', read_only=True)
PHYSICS = Param[Physics]('physics', Physics.AUTO, 'Physics engine of the server.', read_only=True)
PAUSED = Param[bool]('paused', False, 'Start paused, stepping only on request.', read_only=True)
REAL_TIME_FACTOR = Param[float]('real_time_factor', 1.0, 'Sim seconds per wall second while unpaused (0 = as fast as it can).', constraint=Range(lo=0.0))
SERVER_PARAMS = (PHYSICS_DT, DYNAMICS, PHYSICS, PAUSED, REAL_TIME_FACTOR)

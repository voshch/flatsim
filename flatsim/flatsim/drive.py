"""Drive response of one planar robot body: command limiter, lag, traction and turning pivot."""

from __future__ import annotations

import math

import attrs

Twist2D = tuple[float, float, float]

DEFAULT_TIMEOUT = 0.5


@attrs.frozen
class Axis:
    """Velocity bounds of one body axis with its acceleration cap."""

    low: float = -math.inf
    high: float = math.inf
    accel: float = math.inf

    def clip(self, value: float) -> float:
        return min(max(value, self.low), self.high)

    def slew(self, current: float, target: float, dt: float) -> float:
        return _toward(current, target, self.accel * dt)


@attrs.frozen
class DriveModel:
    """Command limiter followed by the body response, per body axis."""

    linear: Axis = Axis()
    lateral: Axis = Axis()
    angular: Axis = Axis()
    gain: Twist2D = (1.0, 1.0, 1.0)
    lag: float = 0.0
    traction: float = math.inf
    pivot: float = 0.0
    timeout: float = DEFAULT_TIMEOUT

    def setpoint(self, current: Twist2D, command: Twist2D, dt: float) -> Twist2D:
        """Command after velocity clipping and acceleration limiting."""
        return (
            self.linear.slew(current[0], self.linear.clip(command[0]), dt),
            self.lateral.slew(current[1], self.lateral.clip(command[1]), dt),
            self.angular.slew(current[2], self.angular.clip(command[2]), dt),
        )

    def follow(self, twist: Twist2D, setpoint: Twist2D, dt: float) -> Twist2D:
        """Body twist at the base origin one step later, turning about the pivot, lagging by `lag`, the planar change capped by traction."""
        turn = setpoint[2] * self.gain[2]
        target = (setpoint[0] * self.gain[0], setpoint[1] * self.gain[1] - self.pivot * turn, turn)
        if self.lag > 0.0:
            blend = 1.0 - math.exp(-dt / self.lag)
            target = (*(now + (goal - now) * blend for now, goal in zip(twist, target, strict=True)),)
        dvx, dvy = target[0] - twist[0], target[1] - twist[1]
        change = math.hypot(dvx, dvy)
        scale = min(1.0, self.traction * dt / change) if change > 0.0 else 0.0
        return (twist[0] + dvx * scale, twist[1] + dvy * scale, target[2])

    def encoded(self, twist: Twist2D) -> Twist2D:
        """Setpoint a steady body twist answers to, the twist wheel odometry reads."""
        return (_over(twist[0], self.gain[0]), _over(twist[1] + self.pivot * twist[2], self.gain[1]), _over(twist[2], self.gain[2]))


def _over(value: float, gain: float) -> float:
    return value / gain if gain else 0.0


def _toward(current: float, target: float, step: float) -> float:
    if not math.isfinite(step):
        return target
    return current + min(max(target - current, -step), step)

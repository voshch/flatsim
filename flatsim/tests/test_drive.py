"""DriveModel: command limits, traction, gain, pivot and lag."""

import math

import pytest
from flatsim.drive import Axis, DriveModel


def test_setpoint_clips_then_slews_each_axis():
    model = DriveModel(linear=Axis(-1.0, 1.0, 2.0), lateral=Axis(0.0, 0.0), angular=Axis(-2.0, 2.0))
    assert model.setpoint((0.0, 0.0, 0.0), (5.0, 0.7, 9.0), 0.1) == pytest.approx((0.2, 0.0, 2.0))
    assert model.setpoint((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1) == pytest.approx((0.8, 0.0, 0.0))


def test_follow_ramps_at_the_traction_limit():
    model = DriveModel(traction=7.85)
    twist, steps = (0.0, 0.0, 0.0), 0
    while twist[0] < 2.0 - 1e-9:
        twist = model.follow(twist, (2.0, 0.0, 0.0), 1.0 / 30.0)
        steps += 1
    assert steps == math.ceil(2.0 / (7.85 / 30.0))
    assert twist[0] == pytest.approx(2.0)


def test_follow_caps_the_planar_change_as_a_vector_and_applies_the_gain():
    model = DriveModel(traction=1.0, gain=(1.0, 1.0, 1.5))
    vx, vy, wz = model.follow((0.0, 0.0, 0.0), (3.0, 4.0, 1.0), 0.5)
    assert math.hypot(vx, vy) == pytest.approx(0.5)
    assert vy / vx == pytest.approx(4.0 / 3.0)
    assert wz == pytest.approx(1.5)


def test_follow_turns_about_the_pivot():
    model = DriveModel(pivot=0.13, gain=(1.0, 1.0, 1.2))
    assert model.follow((0.0, 0.0, 0.0), (0.5, 0.0, 1.0), 0.1) == pytest.approx((0.5, -0.13 * 1.2, 1.2))


def test_encoded_reads_a_steady_body_twist_as_its_setpoint():
    model = DriveModel(pivot=0.13, gain=(0.9, 1.1, 1.2))
    assert model.encoded(model.follow((0.0, 0.0, 0.0), (0.5, 0.2, 1.0), 0.1)) == pytest.approx((0.5, 0.2, 1.0))
    assert DriveModel(gain=(1.0, 0.0, 1.0)).encoded((0.5, 0.3, 1.0)) == (0.5, 0.0, 1.0)


def test_follow_with_a_lag_reaches_the_time_constant_point_after_one_lag():
    model = DriveModel(lag=0.25)
    twist, dt = (0.0, 0.0, 0.0), 1.0 / 400.0
    for _ in range(100):
        twist = model.follow(twist, (0.5, 0.2, 1.0), dt)
    assert twist == pytest.approx((0.5 * (1 - math.exp(-1)), 0.2 * (1 - math.exp(-1)), 1 - math.exp(-1)))

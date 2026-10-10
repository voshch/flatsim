"""Sensor schedule and drive model conversion of the robot topics."""

from __future__ import annotations

import math

import pytest


def test_a_ten_hertz_lidar_publishes_on_every_three_tick_step() -> None:
    pytest.importorskip('flatsim_msgs')
    from flatsim.robot_io import Rate, period

    dt = 0.0333
    rate = Rate(period=period(10.0, dt))
    start = 1791463462.0
    missed = 0
    for step in range(1, 5001):
        now = start + 3 * step * dt
        if rate.due(now):
            rate.last_stamp = now
        else:
            missed += 1
    assert missed == 0
    assert period(None, dt) == dt
    assert period(1000.0, dt) == dt


def test_drive_model_message_keeps_every_field() -> None:
    pytest.importorskip('flatsim_msgs')
    from flatsim.drive import Axis, DriveModel
    from flatsim.robot_io import drive_model
    from flatsim_msgs.msg import Axis as AxisMsg
    from flatsim_msgs.msg import DriveModel as DriveModelMsg

    msg = DriveModelMsg(
        linear=AxisMsg(low=-2.0, high=2.0, accel=20.0),
        lateral=AxisMsg(low=0.0, high=0.0, accel=math.inf),
        angular=AxisMsg(low=-4.0, high=4.0, accel=25.0),
        gain=[1.0, 1.0, 1.36],
        lag=0.1,
        traction=7.85,
        pivot=0.131,
        timeout=0.25,
    )
    assert drive_model(msg) == DriveModel(Axis(-2.0, 2.0, 20.0), Axis(0.0, 0.0), Axis(-4.0, 4.0, 25.0), (1.0, 1.0, 1.36), 0.1, 7.85, 0.131, 0.25)

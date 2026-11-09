"""Physics engine of a world: Box2D on the host or Warp on a device, `auto` taking Warp when a CUDA device is visible."""

from __future__ import annotations

import typing

from flatsim.params import Dynamics, Physics
from flatsim.world import FlatWorld

if typing.TYPE_CHECKING:
    from flatsim.warp_world import WarpWorld

    World = FlatWorld | WarpWorld


def resolve(physics: Physics) -> Physics:
    """The engine `physics` stands for."""
    if physics is not Physics.AUTO:
        return physics
    import warp as wp

    wp.config.quiet = True
    wp.init()
    return Physics.WARP if wp.is_cuda_available() else Physics.BOX2D


def make_world(dt: float, dynamics: Dynamics = Dynamics.RIGID, physics: Physics = Physics.BOX2D) -> World:
    if resolve(physics) is Physics.WARP:
        from flatsim.warp_world import WarpWorld

        return WarpWorld(dt, dynamics)
    return FlatWorld(dt, dynamics)

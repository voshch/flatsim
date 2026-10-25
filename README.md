# flatsim

A 2D physics simulator for mobile robots on ROS 2: planar robots, walls,
obstacles and pedestrians on Box2D 3 (`pyb2d3`), with ray-cast sensors and no
renderer. One server process owns sim time and steps one world. It is driven
entirely through ROS services and topics.

## Layout

```
flatsim/                     this repository
|-- flatsim/                 ament_python package
|   |-- flatsim/
|   |   |-- world.py         FlatWorld: bodies, outlines, boxes, discs, levels, stepping, batched ray casts, contacts
|   |   |-- drive.py         DriveModel: per-axis velocity and acceleration limits, gain, lag, traction, pivot
|   |   |-- sensors.py       beam fans, lidar and depth camera point clouds, IMU acceleration
|   |   |-- urdf.py          link offsets, movable joints and gazebo sensor blocks read from a URDF string
|   |   |-- robot_io.py      topics of one robot: odom, tf, joint states, sensors and their schedule
|   |   |-- scene.py         the world of the server with its robots and floors
|   |   `-- server.py        flatsim_server: clock, pause and step, services
|   |-- launch/flatsim.launch.py
|   `-- tests/
`-- flatsim_msgs/            messages and services
```

## Quick start

```
mkdir -p ws/src && cd ws/src && git clone https://github.com/voshch/flatsim.git && cd ..
rosdep install --from-paths src --ignore-src -y
python3 -m pip install --break-system-packages attrs pyb2d3
colcon build && source install/setup.bash
ros2 launch flatsim flatsim.launch.py
```

`pyb2d3` has no rosdep key, so pip installs it next to the system packages
(or into the virtual environment your ROS Python uses).

In a second shell, put a wall and a robot into the world and drive it:

```
ros2 service call /flatsim/spawn_boxes flatsim_msgs/srv/SpawnBoxes \
  "{boxes: [{name: wall, size_x: 0.1, size_y: 6.0, x: 3.0, z_max: 2.0}]}"
ros2 service call /flatsim/spawn_robot flatsim_msgs/srv/SpawnRobot "{
  name: bot, robot_namespace: /bot, odom_frame: odom, base_frame: base_link, radius: 0.2,
  drive: {linear: {low: -1.0, high: 1.0, accel: 2.0}, angular: {low: -2.0, high: 2.0, accel: 6.0},
          gain: [1.0, 1.0, 1.0], traction: .inf, timeout: 0.5},
  urdf: '<robot name=\"bot\"><link name=\"base_link\"/></robot>',
  sensors: [{name: scan, type: laserscan, topic: /bot/scan, frame: base_link,
             fan: [{angle_min: -3.14159, angle_increment: 0.0174533, beams: 361, range_max: 10.0}]}]}"
ros2 topic pub -r 10 /bot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
ros2 topic echo --once /bot/odom
```

## Time

The server publishes `/clock` and nothing else moves time.

- Unpaused, it steps the world by `physics_dt` and paces itself to
  `real_time_factor` sim seconds per wall second, `0` being as fast as it can.
- Paused, it advances only on `control_world` with `multi_step`, takes exactly
  that many steps before it answers, and returns the sim time it reached. The
  frozen time keeps going out on `/clock` every 10 ms.

Parameters of the `flatsim` node (launch arguments of the same names):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `physics_dt` | `0.0333` | Physics step in sim seconds. Fixed at start. |
| `real_time_factor` | `1.0` | Pace while unpaused, `0` = unbounded. Can be set while running. |
| `paused` | `false` | Start paused. Fixed at start, changed through `control_world`. |
| `dynamics` | `rigid` | `rigid`: walls and obstacles stop or push a robot, pedestrian discs touch it without pushing. `kinematic`: it passes through, contacts are still published. Fixed at start. |

## World

The server keeps one world. Names are unique across
its entities, floors and robots, and a caller that runs several setups side by
side keeps them apart in the plane and gives their names a common prefix, so
that `delete_prefix` removes one setup whole.

Entities, each with a z extent `[z_min, z_max]`:

- outlines: static closed rings, each edge a two-sided wall segment
- boxes: rectangles placed by messages on `/flatsim/poses`
- discs: circles with a velocity, the shape of a pedestrian

Floors are named heights. Together they split z into levels, and a robot
collides with and sees only the entities whose extent reaches into the level it
stands on.

The world holds at most 32 robots and 32 levels.

## Services

All under `/flatsim`, types in [flatsim_msgs/srv](flatsim_msgs/srv). Each
service queues 100 requests, so give a client that sends many at once the same
depth.

| Service | Type | Purpose |
| --- | --- | --- |
| `control_world` | `ControlWorld` | Pause, unpause, or take `multi_step` physics steps while paused |
| `add_floors` | `AddFloors` | Add or replace floors |
| `add_outlines` | `AddOutlines` | Add or replace static outlines |
| `spawn_boxes` | `SpawnBoxes` | Add or replace boxes |
| `delete_entities` | `DeleteEntities` | Remove outlines, boxes, discs and floors by name |
| `delete_prefix` | `DeletePrefix` | Remove every entity, floor and robot whose name starts with a prefix, returns how many |
| `spawn_robot` | `SpawnRobot` | Add or replace a robot and open its topics |
| `set_robot_pose` | `SetRobotPose` | Teleport a robot to rest and publish its transform |
| `delete_robot` | `DeleteRobot` | Remove a robot and close its topics |
| `republish` | `Republish` | Publish the state and every sensor of the robots whose name starts with a prefix, at the current time |

## Topics

| Topic | Type | Direction |
| --- | --- | --- |
| `/clock` | `rosgraph_msgs/Clock` | out, every physics step and every 10 ms while paused |
| `/flatsim/discs` | `flatsim_msgs/Discs` | in, discs to create or move and discs to remove, applied in order of arrival |
| `/flatsim/poses` | `flatsim_msgs/EntityPoses` | in, new poses of boxes and discs, unknown names ignored |
| `<robot_namespace>/cmd_vel` | `geometry_msgs/Twist` | in, the body twist command |
| `<robot_namespace>/odom` | `nav_msgs/Odometry` | out, every step |
| `<robot_namespace>/joint_states` | `sensor_msgs/JointState` | out, every step, the movable joints of the URDF at zero |
| `tf_topic` of the spawn request | `tf2_msgs/TFMessage` | out, every step: `world_frame` to odom (identity) and odom to base |
| `topic` of each sensor | see below | out, at the sensor's update rate |

## Robots

`spawn_robot` takes everything a robot needs, so no robot description lives in
this repository:

- the body: a convex `footprint` in the base frame, or a disc of `radius`
- the [`DriveModel`](flatsim_msgs/msg/DriveModel.msg): per axis the velocity
  bounds and acceleration limit of the command, then the body response (gain,
  first-order lag, traction limit, turn pivot) and the command timeout. An
  axis with `low = high = 0` does not move, a zero gain does not either.
- the URDF, read for its movable joints, the pose of every sensor frame in the
  base frame, and the `<gazebo reference="..."><sensor>` blocks
- the sensors, each a [`SensorSpec`](flatsim_msgs/msg/SensorSpec.msg):

| `type` | Message | Source |
| --- | --- | --- |
| `laserscan` | `sensor_msgs/LaserScan` | the `<sensor>` named by `sensor` (lidar or ray block), or the spec's `fan` when the URDF has none |
| `pointcloud` | `sensor_msgs/PointCloud2` | the same lidar block with its vertical rings, or a depth camera block |
| `imu` | `sensor_msgs/Imu` | body rate, yaw and specific force in the sensor frame |
| `contact` | `ros_gz_interfaces/Contacts` | entities and robots the body touches |

A laserscan and a pointcloud that name the same `<sensor>` share one ray cast.
Sensors publish on whole multiples of their update period, every step without
one. Odometry reads the wheel twist, the body twist divided by the gain and
taken at the pivot, while the IMU reports the body rate. Camera images are not
simulated.

## Library

The world is usable without ROS: `flatsim.world`, `flatsim.drive`,
`flatsim.sensors` and `flatsim.urdf` import only `attrs`, `numpy` and
`pyb2d3`.

- `FlatWorld(dt, dynamics)` steps in whole `dt` steps. `advance(to)` takes
  every step that fits before `to` and returns how many it took.
- `add_outline`, `add_box`, `add_disc`, `move`, `remove`, `set_levels`,
  `add_robot`, `command`, `set_robot_pose` mirror the services above.
- `cast(name, spec)` returns every hit of a `ScanSpec` fan with its z extent,
  so one cast yields a planar scan and every ring of a 3D lidar. The fan is
  intersected with the world's segments and circles in one batch, each beam
  tested only against the shapes its direction can reach.
  `RayHits.first(slope)` is the nearest hit per beam at that elevation, with
  downward beams returning the level floor. `scan` is the planar shorthand.
- `robot_state` gives pose, body twist and body acceleration, `contacts` the
  entities a robot's shape touches. `DriveModel.encoded` turns a body twist
  into the setpoint it answers to, the twist wheel odometry reads.

The same inputs give byte-identical states and scans.

## Test

```
colcon build && source install/setup.bash
cd src/flatsim/flatsim && python3 -m pytest tests
```

The tests of the library run without ROS. The server tests start a
`flatsim_server` and a probe node in the test process and skip when
`flatsim_msgs` is not built. They use `/clock` and `/flatsim`, so give them a
`ROS_DOMAIN_ID` of their own on a shared network.

## License

MIT, see [LICENSE.md](LICENSE.md).

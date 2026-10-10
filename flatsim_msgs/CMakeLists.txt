cmake_minimum_required(VERSION 3.8)
project(flatsim_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Axis.msg"
  "msg/Box.msg"
  "msg/Disc.msg"
  "msg/Discs.msg"
  "msg/DriveModel.msg"
  "msg/EntityPose.msg"
  "msg/EntityPoses.msg"
  "msg/Fan.msg"
  "msg/Floor.msg"
  "msg/Outline.msg"
  "msg/Ring.msg"
  "msg/SensorSpec.msg"
  "srv/AddFloors.srv"
  "srv/AddOutlines.srv"
  "srv/ControlWorld.srv"
  "srv/DeleteEntities.srv"
  "srv/DeletePrefix.srv"
  "srv/DeleteRobot.srv"
  "srv/Republish.srv"
  "srv/SetRobotPose.srv"
  "srv/SpawnBoxes.srv"
  "srv/SpawnRobot.srv"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

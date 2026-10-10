"""URDF link offsets, joint names and sensor blocks behind the flat sensor mounts and joint states."""

from __future__ import annotations

import math

import pytest

from flatsim.urdf import CameraBlock, LidarBlock, frame_inverted, frame_offset, movable_joints, sensor_block

URDF = """<?xml version="1.0"?>
<robot name="probe">
  <link name="base_footprint"/>
  <link name="base_link"/>
  <link name="chassis"/>
  <link name="lidar_mount"/>
  <link name="lidar_link"/>
  <link name="wheel"/>
  <link name="mirror_wheel"/>
  <link name="rear_laser"/>
  <link name="imu_link"/>
  <link name="camera_link"/>
  <joint name="footprint_joint" type="fixed">
    <parent link="base_footprint"/><child link="base_link"/>
    <origin xyz="0.3 0 0.05" rpy="0 0 0"/>
  </joint>
  <joint name="chassis_joint" type="fixed">
    <parent link="base_link"/><child link="chassis"/>
    <origin xyz="0 0 0.1"/>
  </joint>
  <joint name="mount_joint" type="fixed">
    <parent link="chassis"/><child link="lidar_mount"/>
    <origin xyz="0.12 0 0.2" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="spin_joint" type="revolute">
    <parent link="lidar_mount"/><child link="lidar_link"/>
    <origin xyz="0.01 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/><child link="wheel"/>
  </joint>
  <joint name="mirror_joint" type="continuous">
    <parent link="base_link"/><child link="mirror_wheel"/>
    <mimic joint="wheel_joint"/>
  </joint>
  <joint name="rear_laser_joint" type="fixed">
    <parent link="base_link"/><child link="rear_laser"/>
    <origin xyz="-0.3 0 0.25" rpy="3.141592653589793 0 2.356194490192345"/>
  </joint>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/><child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 -3.141592653589793 0.7853981633974483"/>
  </joint>
  <joint name="camera_joint" type="fixed">
    <parent link="chassis"/><child link="camera_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0.1 0"/>
  </joint>
  <gazebo reference="lidar_link">
    <sensor name="gpu_lidar" type="gpu_lidar">
      <gz_frame_id>probe/lidar_link</gz_frame_id>
      <pose>0 0 0.142 0 0 0</pose>
      <update_rate>10</update_rate>
      <lidar>
        <scan>
          <horizontal>
            <samples>640</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
          <vertical>
            <samples>16</samples>
            <resolution>1</resolution>
            <min_angle>-0.261799</min_angle>
            <max_angle>0.261799</max_angle>
          </vertical>
        </scan>
        <range>
          <min>0.08</min>
          <max>12.0</max>
          <resolution>0.01</resolution>
        </range>
      </lidar>
      <always_on>1</always_on>
    </sensor>
  </gazebo>
  <gazebo reference="rear_laser">
    <sensor type="ray" name="rear_laser">
      <visualize>false</visualize>
      <update_rate> 12.5 </update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>541</samples>
            <resolution>1.0</resolution>
            <min_angle>-2.3562</min_angle>
            <max_angle>2.3562</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.05</min>
          <max>30.0</max>
        </range>
      </ray>
      <plugin name="rear_laser" filename="libgazebo_ros_ray_sensor.so">
        <update_rate>40</update_rate>
      </plugin>
    </sensor>
  </gazebo>
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <plugin filename="libgz-sim-imu-system" name="gz::sim::systems::Imu">
        <topic>/imu/data</topic>
      </plugin>
      <always_on>true</always_on>
      <update_rate>50</update_rate>
      <imu>
        <angular_velocity><x><noise type="gaussian"><mean>0.0</mean></noise></x></angular_velocity>
      </imu>
    </sensor>
    <material>Gazebo/Grey</material>
  </gazebo>
  <gazebo reference="camera_link">
    <sensor name="front_depth" type="depth_camera">
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <topic>camera/depth_image</topic>
      <camera>
        <camera_info_topic>camera/camera_info</camera_info_topic>
        <horizontal_fov>1.5184351666666667</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R_FLOAT32</format>
        </image>
        <clip>
          <near>0.105</near>
          <far>8.0</far>
        </clip>
      </camera>
    </sensor>
    <sensor name="bumper" type="contact">
      <contact><collision>base_link_collision</collision></contact>
    </sensor>
    <sensor name="bare" type="lidar"/>
  </gazebo>
  <gazebo reference="chassis">
    <sensor name="imu_sensor" type="imu">
      <update_rate>200</update_rate>
    </sensor>
  </gazebo>
  <gazebo>
    <sensor name="unbound" type="imu"/>
  </gazebo>
</robot>
"""


def test_offset_composes_the_joint_chain_below_the_parent_link():
    x, y, z, yaw = frame_offset(URDF, 'base_link', 'lidar_link')
    assert (x, y, z) == pytest.approx((0.12, 0.01, 0.3))
    assert yaw == pytest.approx(math.pi / 2)


def test_offset_between_siblings_and_upward_goes_through_the_root():
    x, y, z, yaw = frame_offset(URDF, 'lidar_link', 'base_footprint')
    assert (x, y, z) == pytest.approx((-0.01, 0.42, -0.35))
    assert yaw == pytest.approx(-math.pi / 2)
    assert frame_offset(URDF, 'wheel', 'chassis') == pytest.approx((0.0, 0.0, 0.1, 0.0))


def test_unknown_link_names_the_known_ones():
    with pytest.raises(ValueError, match='lidar_link'):
        frame_offset(URDF, 'base_link', 'laser')


def test_movable_joints_skip_fixed_and_mimic():
    assert movable_joints(URDF) == ['spin_joint', 'wheel_joint']


def test_inverted_frame_is_told_by_its_z_axis_whatever_the_yaw():
    assert frame_inverted(URDF, 'base_link', 'rear_laser')
    assert frame_inverted(URDF, 'base_link', 'imu_link')
    assert not frame_inverted(URDF, 'base_link', 'lidar_link')
    assert not frame_inverted(URDF, 'base_link', 'camera_link')
    assert frame_offset(URDF, 'base_link', 'rear_laser') == pytest.approx((-0.3, 0.0, 0.25, 2.356194490192345))


def test_sixteen_ring_lidar_block_keeps_both_scan_axes_and_the_range():
    block = sensor_block(URDF, 'gpu_lidar')
    assert (block.name, block.type, block.link) == ('gpu_lidar', 'gpu_lidar', 'lidar_link')
    assert block.update_rate == 10.0
    assert block.camera is None
    assert block.lidar == LidarBlock(samples=640, min_angle=-3.14159, max_angle=3.14159, vertical_samples=16, vertical_min_angle=-0.261799, vertical_max_angle=0.261799, range_min=0.08, range_max=12.0)


def test_planar_ray_block_has_one_vertical_sample_and_its_own_rate():
    block = sensor_block(URDF, 'rear_laser')
    assert (block.type, block.link, block.update_rate) == ('ray', 'rear_laser', 12.5)
    assert block.lidar == LidarBlock(samples=541, min_angle=-2.3562, max_angle=2.3562, vertical_samples=1, vertical_min_angle=0.0, vertical_max_angle=0.0, range_min=0.05, range_max=30.0)


def test_imu_block_is_picked_by_link_when_two_share_a_name():
    first = sensor_block(URDF, 'imu_sensor')
    assert (first.type, first.link, first.update_rate) == ('imu', 'imu_link', 50.0)
    assert first.lidar is None and first.camera is None
    assert sensor_block(URDF, 'imu_sensor', link='chassis').update_rate == 200.0
    assert sensor_block(URDF, 'imu_sensor', link='wheel').link == 'imu_link'


def test_depth_camera_block_reads_the_pinhole_geometry():
    block = sensor_block(URDF, 'front_depth')
    assert (block.type, block.link, block.update_rate) == ('depth_camera', 'camera_link', 30.0)
    assert block.lidar is None
    assert block.camera == CameraBlock(horizontal_fov=1.5184351666666667, width=640, height=480, near=0.105, far=8.0)


def test_contact_block_and_missing_optional_fields():
    bumper = sensor_block(URDF, 'bumper')
    assert (bumper.type, bumper.update_rate, bumper.lidar, bumper.camera) == ('contact', None, None, None)
    bare = sensor_block(URDF, 'bare')
    assert bare.update_rate is None
    assert bare.lidar == LidarBlock(samples=1, min_angle=-math.pi, max_angle=math.pi, vertical_samples=1, vertical_min_angle=0.0, vertical_max_angle=0.0, range_min=0.0, range_max=100.0)


def test_sensor_without_a_referenced_block_is_missing():
    assert sensor_block(URDF, 'front_laser') is None
    assert sensor_block(URDF, 'unbound') is None

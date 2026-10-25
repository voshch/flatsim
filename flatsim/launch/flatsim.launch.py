import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
import launch_ros.parameter_descriptions


def generate_launch_description() -> launch.LaunchDescription:
    arguments = [
        launch.actions.DeclareLaunchArgument('physics_dt', default_value='0.0333', description='Physics step in sim seconds.'),
        launch.actions.DeclareLaunchArgument('real_time_factor', default_value='1.0', description='Sim seconds per wall second while unpaused, 0 = as fast as it can.'),
        launch.actions.DeclareLaunchArgument('paused', default_value='false', choices=['true', 'false'], description='Start paused, stepping only on request.'),
        launch.actions.DeclareLaunchArgument('dynamics', default_value='rigid', choices=['rigid', 'kinematic'], description='rigid = walls and obstacles stop or push a robot and pedestrian discs touch it without pushing, kinematic = it passes through.'),
    ]

    def parameter(name: str, value_type: type) -> launch_ros.parameter_descriptions.ParameterValue:
        return launch_ros.parameter_descriptions.ParameterValue(launch.substitutions.LaunchConfiguration(name), value_type=value_type)

    server = launch_ros.actions.Node(
        package='flatsim',
        executable='flatsim_server',
        name='flatsim',
        output='screen',
        parameters=[
            {
                'physics_dt': parameter('physics_dt', float),
                'real_time_factor': parameter('real_time_factor', float),
                'paused': parameter('paused', bool),
                'dynamics': parameter('dynamics', str),
            }
        ],
    )
    return launch.LaunchDescription([*arguments, server])

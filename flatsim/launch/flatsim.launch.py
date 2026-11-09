import launch
import launch_ros.actions
from arena_rclpy_mixins.launch_argument import LaunchArgument, forwarded

from flatsim.params import SERVER_PARAMS


def _server(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    return [launch_ros.actions.Node(package='flatsim', executable='flatsim_server', name='flatsim', output='screen', parameters=[forwarded(context, SERVER_PARAMS)])]


def generate_launch_description() -> launch.LaunchDescription:
    return launch.LaunchDescription([*(LaunchArgument.from_param(param) for param in SERVER_PARAMS), launch.actions.OpaqueFunction(function=_server)])

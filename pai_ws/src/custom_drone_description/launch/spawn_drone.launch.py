import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('custom_drone_description')
    xacro_file = os.path.join(pkg_share, 'urdf', 'drone.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'drone_world.world')

    robot_description_raw = subprocess.check_output(['xacro', xacro_file]).decode('utf-8')
    robot_description = {'robot_description': robot_description_raw}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # ワールドファイルを指定してGazeboを起動
    launch_gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen'
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'custom_drone', '-z', '0.5'],
        output='screen'
    )

    node_drone_controller = Node(
        package='custom_drone_description',
        executable='drone_3d_controller.py',
        output='screen'
    )

    return LaunchDescription([
        node_robot_state_publisher,
        launch_gazebo,
        spawn_entity,
        node_drone_controller
    ])
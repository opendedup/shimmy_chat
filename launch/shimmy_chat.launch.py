import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'shimmy_chat'
    share_dir = get_package_share_directory(pkg_name)
    
    # Construct the path to the parameters file
    params_file = os.path.join(share_dir, 'config', 'shimmy_chat.yaml')

    audio_interface = Node(
        package=pkg_name,
        executable='audio_interface_node',
        name='audio_interface_node',
        output='screen',
        parameters=[params_file]
    )

    speech_processor = Node(
        package=pkg_name,
        executable='speech_processor_node',
        name='speech_processor_node',
        output='screen',
        parameters=[params_file]
    )

    chat_logic = Node(
        package=pkg_name,
        executable='chat_logic_node',
        name='chat_logic_node',
        output='screen',
        parameters=[params_file]
    )

    speech_synthesis = Node(
        package=pkg_name,
        executable='speech_synthesis_node',
        name='speech_synthesis_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        audio_interface,
        speech_processor,
        chat_logic,
        speech_synthesis
    ]) 
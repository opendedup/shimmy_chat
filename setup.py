import os # Added for path joining
from glob import glob # Added for finding files
from setuptools import find_packages, setup

package_name = 'shimmy_chat'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
    ],
    install_requires=[
        'setuptools',
        'google-cloud-speech',
        'google-generativeai', # Added for Gemini
        # 'pyaudio', # Potentially needed if doing local audio
        'pydub' # For audio chunking/vad
        ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS 2 package for Shimmy robot chat interaction',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add executables here later
            # Example: 'my_node = shimmy_chat.my_node:main',
            'audio_interface_node = shimmy_chat.audio_interface_node:main',
            'speech_processor_node = shimmy_chat.speech_processor_node:main',
            'chat_logic_node = shimmy_chat.chat_logic_node:main',
            'speech_synthesis_node = shimmy_chat.speech_synthesis_node:main',
            'audio_capture_node = shimmy_chat.audio_capture_node:main',
            'google_asr_subscriber = shimmy_chat.google_asr_subscriber:main',
            'tts_node = shimmy_chat.tts_node:main',
            'audio_vad_node = shimmy_chat.audio_vad_node:main',
        ],
    },
) 
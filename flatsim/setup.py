from glob import glob

from setuptools import find_packages, setup

package_name = 'flatsim'

setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='2D Box2D simulator of planar robots, walls, obstacles and pedestrians with ray-cast sensors',
    license='MIT',
    entry_points={
        'console_scripts': [
            'flatsim_server = flatsim.server:main',
        ],
    },
)

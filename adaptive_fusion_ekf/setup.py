from setuptools import find_packages, setup

package_name = 'adaptive_fusion_ekf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ekf.launch.py']),
        ('share/' + package_name + '/config', ['config/ekf_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xuan',
    maintainer_email='wx3515753265@gmail.com',
    description='Adaptive UWB and SLAM fusion EKF nodes for ROS 2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ekf_fusion_node = adaptive_fusion_ekf.ekf_fusion_node:main',
        ],
    },
)

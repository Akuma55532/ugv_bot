from setuptools import setup

package_name = "adaptive_fusion_navigation"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/nav2_navigation.launch.py",
                "launch/nav2_bringup_map.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            ["config/nav2_fusion_params.yaml"],
        ),
        (
            "share/" + package_name + "/maps",
            [
                "maps/turtlebot3_world.yaml",
                "maps/turtlebot3_world.pgm",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xuan",
    maintainer_email="wx3515753265@gmail.com",
    description="Navigation2 bringup and configuration for adaptive fusion experiments",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
)

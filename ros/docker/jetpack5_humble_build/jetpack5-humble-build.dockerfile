# Get Jetpack 6, Update to latest image version and build ROS2 from source
# Jetpack 5 version that we have = 35.3.1, latest image available 35.4.1
# Based from https://github.com/dusty-nv/jetson-containers/blob/master/packages/ros/Dockerfile.ros2
ARG L4T_RELEASE_MAJOR=35.3
ARG L4T_RELEASE_MINOR=1
ARG ROS_PACKAGE=ros_base
ARG ROS_VERSION=humble

# Generate API Key
# docker login nvcr.io
# login: $oauthtoken, password: API Key
# Ubuntu 20.04 same version as current Jetpack 5, includes CUDA 35.x.x
FROM nvcr.io/nvidia/l4t-jetpack:r$L4T_RELEASE_MAJOR.$L4T_RELEASE_MINOR

ENV ROS_DISTRO=${ROS_VERSION}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

# change the locale from POSIX to UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# build ROS from source
COPY ros2_build.sh ros2_build.sh
RUN ./ros2_build.sh

# Set the default DDS middleware to cyclonedds
# https://github.com/ros2/rclcpp/issues/1335
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# commands will be appended/run by the entrypoint which sources the ROS environment
COPY ros_entrypoint.sh /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]

WORKDIR /

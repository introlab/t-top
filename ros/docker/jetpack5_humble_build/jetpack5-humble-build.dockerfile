# Get Jetpack 6, Update to latest image version and build ROS2 from source
# Jetpack 5 version that we have = 35.3.1, latest image available 35.4.1
# Based from https://github.com/dusty-nv/jetson-containers/blob/master/packages/ros/Dockerfile.ros2

# Generate API Key
# docker login nvcr.io
# login: $oauthtoken, password: API Key
# Ubuntu 20.04 same version as current Jetpack 5, includes CUDA 35.x.x
ARG L4T_RELEASE_MAJOR=35.3
ARG L4T_RELEASE_MINOR=1
FROM nvcr.io/nvidia/l4t-jetpack:r$L4T_RELEASE_MAJOR.$L4T_RELEASE_MINOR

ARG ROS_PACKAGE=ros_base
ARG ROS_VERSION=humble
ENV ROS_DISTRO=${ROS_VERSION}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

# change the locale from POSIX to UTF-8
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends locales \
#    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1


# install deps and build librealsense with CUDA support
RUN apt-get install -y --no-install-recommends \
  git \
  libgtk-3-dev \
  libglfw3-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  build-essential \
  cmake \
  cmake-curses-gui \
  libssl-dev \
  libusb-1.0-0-dev \
  libudev-dev \
  pkg-config

  RUN git clone https://github.com/IntelRealSense/librealsense.git -b v2.55.1 --depth 1  --recurse-submodules

  # Make build directory
  RUN mkdir -p /tmp/librealsense/build
  WORKDIR /tmp/librealsense/build
  RUN cmake ../ -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DBUILD_EXAMPLES=false -DBUILD_WITH_CUDA=true
  RUN make -j 12
  RUN make install

# Useful ?
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
# build ROS from source

COPY ros2_build.sh ros2_build.sh
RUN ./ros2_build.sh

# Set the default DDS middleware to cyclonedds
# https://github.com/ros2/rclcpp/issues/1335
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp


#
# Additional system deps
#
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  libasound2-dev \
  libpulse-dev \
  libconfig-dev \
  alsa-utils \
  gfortran \
  'libgfortran-*-dev' \
  texinfo \
  libfftw3-dev \
  libsqlite3-dev \
  portaudio19-dev \
  libgecode-dev \
  qtbase5-dev \
  qt5-qmake \
  v4l-utils \
  ffmpeg \
  chromium-browser \
  libqt5websockets5-dev \
  libqt5charts5-dev \
  libqt5serialport5-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev \
  libgstreamer-plugins-bad1.0-dev \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-tools \
  nodejs \
  npm

# Make ROS2 workspace
RUN mkdir -p /root/ros2/workspace/src
WORKDIR /root/ros2/workspace/src
# Get CV Camera
RUN git clone -b master https://github.com/Kapernikov/cv_camera.git --depth 1 --recurse-submodules
# Diagnostics Updater
RUN git clone -b ros2-${ROS_VERSION} https://github.com/ros/diagnostics.git --depth 1 --recurse-submodules
# Realsense ROS2 from source and compile
RUN git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.55.1 --depth 1 --recurse-submodules
# Go back to workspace and compile both packages
WORKDIR /root/ros2/workspace
RUN source ${ROS_ROOT}/install/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release


# Add T-TOP to the container
RUN touch "LATEST_BUILD_JUNE_5_2024_12_00AM"
WORKDIR /root/ros2/workspace/src
RUN git clone -b ros2-migration https://github.com/introlab/t-top.git --depth 1 --recurse-submodules
WORKDIR /root/ros2/workspace
RUN source ${ROS_ROOT}/install/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release


# commands will be appended/run by the entrypoint which sources the ROS environment
COPY ros_entrypoint.sh /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]

WORKDIR /

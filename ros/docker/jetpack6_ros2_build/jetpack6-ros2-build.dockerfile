# Get Jetpack 6, Update to latest image version and build ROS2 from source
# Jetpack 5 version that we have = 35.3.1, latest image available 35.4.1
ARG L4T_RELEASE_MAJOR=36.3
ARG L4T_RELEASE_MINOR=0

# Generate API Key
# docker login nvcr.io
# login: $oauthtoken, password: API Key

# Ubuntu 22.04 same version as current Jetpack 6, includes CUDA 36.x.x
# Ubuntu 20.04 samve version as current Jetpack 5, includes CUDA 35.x.x
FROM nvcr.io/nvidia/l4t-jetpack:r$L4T_RELEASE_MAJOR.$L4T_RELEASE_MINOR

# Change default shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# Set debian frontend non interactive for apt
ENV DEBIAN_FRONTEND=noninteractive

# Set system locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Update APT
RUN apt-get update
RUN apt-get upgrade -y

# Procedure for ROS2 Install from :  https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html

RUN apt-get install software-properties-common -y
RUN add-apt-repository universe

RUN apt-get update && apt-get install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Base ROS2 deps
RUN apt-get update && apt-get install -y \
  python3-flake8-docstrings \
  python3-pip \
  python3-pytest-cov \
  python3-flake8-blind-except \
  python3-flake8-builtins \
  python3-flake8-class-newline \
  python3-flake8-comprehensions \
  python3-flake8-deprecated \
  python3-flake8-import-order \
  python3-flake8-quotes \
  python3-pytest-repeat \
  python3-pytest-rerunfailures \
  ros-dev-tools

RUN mkdir -p /ros2_humble/src
WORKDIR /ros2_humble
RUN vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src

RUN apt-get upgrade -y
RUN rosdep init
RUN rosdep update
RUN rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"

# Build ROS2
RUN colcon build --symlink-install

RUN echo "source /ros2_humble/install/local_setup.bash" >> ~/.bashrc
#Reload SHELL with updated bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Download and compile librealsense with CUDA support
RUN apt-get install \
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
  pkg-config -y

WORKDIR /root
RUN git clone https://github.com/IntelRealSense/librealsense.git -b v2.55.1 --depth 1  --recurse-submodules

# Make build directory
RUN mkdir -p /root/librealsense/build
WORKDIR /root/librealsense/build
RUN cmake ../ -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DBUILD_EXAMPLES=false -DBUILD_WITH_CUDA=true
RUN make -j 12
RUN make install

# Make ROS2 workspace
RUN mkdir -p /root/ros2/workspace/src

#
# Additional system deps
#
RUN apt-get install \
  libboost-python-dev \
  libboost-all-dev \
  'libpcl-*-dev' \
  libeigen3-dev \
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
  python3-all-dev \
  libgecode-dev \
  qtbase5-dev \
  qt5-qmake \
  v4l-utils \
  libopenblas-dev \
  libpython3-dev \
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
  libspdlog-dev  \
  libpdal-dev \
  libangles-dev \
  libzmq3-dev  \
  librsl-dev \
  python3-jinja2 \
  python3-typeguard \
  nodejs \
  npm -y

WORKDIR /root/ros2/workspace/src
# Get CV Bridge from source and compile
RUN git clone -b humble https://github.com/ros-perception/vision_opencv.git --depth 1 --recurse-submodules
# Get CV Camera
RUN git clone -b master https://github.com/Kapernikov/cv_camera.git --depth 1 --recurse-submodules
# Diagnostics Updater
RUN git clone -b ros2-humble https://github.com/ros/diagnostics.git --depth 1 --recurse-submodules
# Realsense ROS2 from source and compile
RUN git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.55.1 --depth 1 --recurse-submodules

# Go back to workspace and compile both packages
WORKDIR /root/ros2/workspace
RUN source /ros2_humble/install/local_setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

#
# RTABMAP
#
WORKDIR /root
RUN git clone -b master https://github.com/introlab/rtabmap.git --recurse-submodules
RUN mkdir -p /root/rtabmap/build
WORKDIR /root/rtabmap/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

#
# BEHAVIOR TREE
#
WORKDIR /root
# RUN git clone -b 4.6.1 https://github.com/BehaviorTree/BehaviorTree.CPP.git --depth 1 --recurse-submodules
RUN git clone -b release/humble/behaviortree_cpp_v3 https://github.com/BehaviorTree/behaviortree_cpp_v3-release.git
RUN mkdir -p /root/behaviortree_cpp_v3-release/build
WORKDIR /root/behaviortree_cpp_v3-release/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

#
# XTENSOR
#
RUN apt-get install 'libgraphicsmagick*-dev' -y

WORKDIR /root
RUN git clone -b 0.7.0 https://github.com/xtensor-stack/xtl.git --depth 1 --recurse-submodules
RUN mkdir -p /root/xtl/build
WORKDIR /root/xtl/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

WORKDIR /root
RUN git clone -b 7.4.8 https://github.com/xtensor-stack/xsimd.git --depth 1 --recurse-submodules
RUN mkdir -p /root/xsimd/build
WORKDIR /root/xsimd/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

WORKDIR /root
RUN git clone -b 0.23.10 https://github.com/xtensor-stack/xtensor --depth 1 --recurse-submodules
RUN mkdir -p /root/xtensor/build
WORKDIR /root/xtensor/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

#
# Octomap
#
WORKDIR /root
RUN git clone -b v1.10.0 https://github.com/OctoMap/octomap.git --depth 1 --recurse-submodules
RUN mkdir -p /root/octomap/build
WORKDIR /root/octomap/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

WORKDIR /root
RUN git clone -b v3.11.3 https://github.com/nlohmann/json.git --depth 1 --recurse-submodules
RUN mkdir -p /root/json/build
WORKDIR /root/json/build
RUN cmake ../
RUN cmake --build . --parallel 12
RUN cmake --install .

#
# RTABMAP ROS & DEPENDENCIES
#


RUN apt-get install libceres-dev libompl-dev -y

WORKDIR /root/ros2/workspace/src
# NAV2
RUN git clone -b humble https://github.com/ros-navigation/navigation2 --depth 1 --recurse-submodules && \
  git clone -b ros2 https://github.com/ros-geographic-info/geographic_info.git --depth 1 --recurse-submodules && \
  git clone -b ros2 https://github.com/ros/bond_core.git --depth 1 --recurse-submodules && \
  git clone -b 1.1.0 https://github.com/PickNikRobotics/RSL.git --depth 1 --recurse-submodules && \
  git clone -b 1.0.2 https://github.com/PickNikRobotics/cpp_polyfills.git --depth 1 --recurse-submodules && \
  git clone -b 0.3.8 https://github.com/PickNikRobotics/generate_parameter_library.git && \
  # PCL - ROS
  git clone -b ros2 https://github.com/ros-perception/pcl_msgs.git --depth 1 --recurse-submodules && \
  git clone -b ros2 https://github.com/ros-perception/perception_pcl.git --depth 1 --recurse-submodules && \
  git clone -b humble https://github.com/ros-perception/image_pipeline.git --depth 1 --recurse-submodules && \
  # OCTOMAP
  git clone -b ros2 https://github.com/OctoMap/octomap_msgs.git --depth 1 --recurse-submodules && \
  git clone -b humble https://github.com/ANYbotics/grid_map.git --depth 1 --recurse-submodules && \
  git clone -b ros2 https://github.com/ros/filters.git --depth 1 --recurse-submodules && \
  # RUN git clone -b humble https://github.com/BehaviorTree/BehaviorTree.ROS2.git --depth 1 --recurse-submodules
  # RUN git clone -b ros2 https://github.com/OctoMap/octomap_ros.git --depth 1 --recurse-submodules
  # RTABMAP-ROS
  git clone -b ros2 https://github.com/introlab/rtabmap_ros.git --depth 1 --recurse-submodules

WORKDIR /root/ros2/workspace
RUN source /ros2_humble/install/local_setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release


#
# T-TOP ROS2
#

RUN touch "LATEST_BUILD_JUNE_4_2024_9_00AM"
WORKDIR /root/ros2/workspace/src
RUN git clone -b ros2-migration https://github.com/introlab/t-top.git --depth 1 --recurse-submodules

#
# WORKDIR /root/ros2/workspace
# RUN source /ros2_humble/install/local_setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Run CMD to start bash
CMD ["/bin/bash"]

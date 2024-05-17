ARG L4T_RELEASE_MAJOR=35.3
ARG L4T_RELEASE_MINOR=1

# Generate API Key
# docker login nvcr.io
# login: $oauthtoken, password: API Key

# Ubuntu 20.04 same version as current Jetpack
FROM nvcr.io/nvidia/l4t-base:$L4T_RELEASE_MAJOR.$L4T_RELEASE_MINOR
FROM nvcr.io/nvidia/l4t-cuda:11.4.19-runtime

# Set system locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Setup ROS2 GPG
RUN apt-get update && apt-get install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update APT
RUN apt-get update
RUN apt-get upgrade -y

# Install ROS2 Foxy
RUN apt-get install ros-foxy-desktop python3-argcomplete -y

# Install ROS1 Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install ros-noetic-desktop-full -y

# Download and compile librealsense with CUDA support
RUN apt-get install git libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev build-essential cmake cmake-curses-gui libssl-dev libusb-1.0-0-dev pkg-config -y
RUN apt-get install cuda-nvcc-11-4 -y
WORKDIR /root
RUN git clone https://github.com/IntelRealSense/librealsense.git -b v2.55.1 --depth 1  --recurse-submodules

# Make build directory
RUN mkdir -p /root/librealsense/build
WORKDIR /root/librealsense/build
RUN cmake ../ -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DBUILD_EXAMPLES=false -DBUILD_WITH_CUDA=true
# -DFORCE_RSUSB_BACKEND=true
RUN make -j8
RUN make install

# Install ROS1 Bridge
RUN apt-get install ros-foxy-ros1-bridge ros-foxy-rosbridge-msgs -y

SHELL ["/bin/bash", "-c"]

# Get ROS2 Realsense from source and compile
RUN apt-get install python3-colcon-common-extensions ros-foxy-ament-cmake-python ros-foxy-diagnostic-updater -y
RUN mkdir -p /root/ros2/workspace/src

WORKDIR /root/ros2/workspace/src
# Get Realsense ROS2 from source and compile
RUN git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.54.1 --depth 1 --recurse-submodules
# Got back to workspace and compile both packages
WORKDIR /root/ros2/workspace
RUN source /opt/ros/foxy/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# From now on, use bash shell with ROS1 & ROS2 + realsense-ros available
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc
RUN echo "source /root/ros2/workspace/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && \
                         source /opt/ros/foxy/setup.bash && \
                         source /root/ros2/workspace/install/setup.bash && \
                         ros2 launch /root/ros2/workspace/launch/realsense_bridge.launch.xml"]

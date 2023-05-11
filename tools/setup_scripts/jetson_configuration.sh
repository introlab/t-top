set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ -t 1 ]; then
    ECHO_IN_BLUE () {
        echo -e ${BLUE}${1}${NC}
    }
    ECHO_IN_GREEN () {
        echo -e ${GREEN}${1}${NC}
    }
else
    ECHO_IN_BLUE () {
        echo ${1}
    }
    ECHO_IN_GREEN () {
        echo ${1}
    }
fi

cmake_build_install_native () {
    # arg 1 [optional]: number of threads to use (-j)
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
    if [ $# -lt 1 ] ; then
        cmake --build .
    else
        cmake --build -j$1 .
    fi
    cmake --install .
}

add_to_bashrc () {
    # arg 1: line to add to bashrc
    local BASHRC_FILE=~/.bashrc

    grep --quiet --no-messages --fixed-regexp --line-regexp -- "$1" "$BASHRC_FILE" || echo "$1" >> "$BASHRC_FILE"
    source $BASHRC_FILE
}

add_to_root_bashrc () {
    # arg 1: line to add to root bashrc
    local BASHRC_FILE=/root/.bashrc

    sudo bash -c "grep --quiet --no-messages --fixed-regexp --line-regexp -- '$1' '$BASHRC_FILE' || echo '$1' >> '$BASHRC_FILE'"
}

ECHO_IN_GREEN "###############################################################"
ECHO_IN_GREEN "T-Top Setup Script"
ECHO_IN_GREEN "###############################################################"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Cloning the T-Top repo"
ECHO_IN_BLUE "###############################################################"
mkdir -p ~/t-top_ws/src
cd ~/t-top_ws/src
git clone --recurse-submodules git@github.com:introlab/t-top.git

TTOP_REPO_PATH='~/t-top_ws/src/t-top'
PATCH_FILES_DIR="$TTOP_REPO_PATH/tools/setup_scripts/patch"
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setting up udev rules"
ECHO_IN_BLUE "###############################################################"
sudo cp $TTOP_REPO_PATH/tools/udev_rules/99-opencr-custom.rules /etc/udev/rules.d/
sudo cp $TTOP_REPO_PATH/tools/udev_rules/99-teensy.rules /etc/udev/rules.d/
sudo cp $TTOP_REPO_PATH/tools/udev_rules/99-camera-2d-wide.rules /etc/udev/rules.d/
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Adding user to dialout group"
ECHO_IN_BLUE "###############################################################"
sudo usermod -a -G dialout $USER
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Installing NPM and Node"
ECHO_IN_BLUE "###############################################################"
sudo apt install -y curl software-properties-common
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt install -y nodejs
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Installing tools"
ECHO_IN_BLUE "###############################################################"
sudo apt install -y htop python3-pip perl
sudo -H pip3 install -U jetson-stats
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Updating CMake"
ECHO_IN_BLUE "###############################################################"
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install cmake
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Cloning Librealsense 2"
ECHO_IN_BLUE "###############################################################"
mkdir -p ~/deps
cd ~/deps

git clone https://github.com/jetsonhacks/buildLibrealsense2Xavier
cd buildLibrealsense2Xavier
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Patching Librealsense 2"
ECHO_IN_BLUE "###############################################################"
patch -u installLibrealsense.sh $PATCH_FILES_DIR/installLibrealsense.patch
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Building and installing Librealsense 2"
ECHO_IN_BLUE "###############################################################"
./installLibrealsense.sh
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ROS build dependencies"
ECHO_IN_BLUE "###############################################################"
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update

sudo apt install -y python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-catkin-tools \
    build-essential \
    libboost-all-dev \
    libpoco-dev python3-empy \
    libtinyxml-dev \
    libtinyxml2-dev \
    qt5-default \
    sip-dev \
    python3-sip \
    python3-sip-dbg \
    python3-sip-dev \
    python3-pyqt5 \
    python3-nose \
    python3-twisted \
    python3-serial \
    python3-autobahn \
    python3-tornado \
    python3-bson \
    python3-qt-binding \
    libcurl4-gnutls-dev \
    libgtest-dev \
    liblz4-dev \
    libfltk1.1-dev \
    liburdfdom-headers-dev \
    liburdfdom-dev \
    liburdfdom-tools \
    libgpgme-dev \
    libyaml-cpp-dev \
    libpcl-dev \
    libgtk-3-dev \
    libassimp-dev \
    libogre-1.9-dev \
    libconfig-dev \
    liblog4cplus-dev \
    alsa-utils \
    liblog4cpp5-dev \
    liblog4cxx-dev \
    libbz2-dev \
    libbullet-dev \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    libapriltag-dev
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ROS system dependencies"
ECHO_IN_BLUE "###############################################################"
cd ~/deps
git clone https://github.com/ros/console_bridge
cd console_bridge
cmake_build_install_native &
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
# make -j
# sudo make install

cd ~/deps
git clone https://github.com/ethz-asl/libnabo.git
cd libnabo
cmake_build_install_native &
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
# make -j
# sudo make install

cd ~/deps
git clone https://github.com/ethz-asl/libpointmatcher.git
cd libpointmatcher
cmake_build_install_native &
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
# make -j
# sudo make install

wait

cd ~/deps
git clone -b 0.20.18-noetic https://github.com/introlab/rtabmap.git
cmake_build_install_native 4
# cd rtabmap/build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
# make -j4
# sudo make install
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Generate ROS build workspace and install dependencies"
ECHO_IN_BLUE "###############################################################"
sudo rosdep init
rosdep update

mkdir -p ~/ros_catkin_ws/src
cd ~/ros_catkin_ws

rosinstall_generator desktop_full --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
vcs import --input noetic-desktop.rosinstall ./src
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Remove useless ROS packages"
ECHO_IN_BLUE "###############################################################"
rm -rf ~/ros_catkin_ws/src/gazebo_ros_pkgs/
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Add missing ROS packages"
ECHO_IN_BLUE "###############################################################"
cd ~/ros_catkin_ws/src
git clone -b 1.0.1 https://github.com/GT-RAIL/rosauth.git
git clone -b noetic-devel https://github.com/ros-drivers/rosserial.git
git clone -b ros1 https://github.com/RobotWebTools/rosbridge_suite.git
git clone -b noetic https://github.com/ccny-ros-pkg/imu_tools.git
git clone --recursive https://github.com/orocos/orocos_kinematics_dynamics.git

git clone -b 0.20.18-noetic https://github.com/introlab/rtabmap_ros.git
git clone -b 1.7.1 https://github.com/ros-perception/perception_pcl.git
git clone -b noetic-devel https://github.com/ros-perception/pcl_msgs.git
git clone -b noetic-devel https://github.com/ros-planning/navigation.git
git clone -b noetic-devel https://github.com/ros-perception/image_transport_plugins

git clone -b kinetic-devel https://github.com/pal-robotics/ddynamic_reconfigure.git
git clone -b 2.3.2 https://github.com/IntelRealSense/realsense-ros.git
git clone https://github.com/OTL/cv_camera.git
git clone -b 0.6.4-noetic https://github.com/introlab/find-object.git
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Replace incomplete ROS packages"
ECHO_IN_BLUE "###############################################################"
rm -rf geometry2 navigation_msgs vision_opencv image_common

git clone -b noetic-devel https://github.com/ros/geometry2.git
git clone -b ros1 https://github.com/ros-planning/navigation_msgs.git
git clone -b noetic https://github.com/ros-perception/vision_opencv.git
git clone -b noetic-devel https://github.com/ros-perception/image_common.git
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Uninstall python3.9"
ECHO_IN_BLUE "###############################################################"
sudo apt autoremove python3.9
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build ROS workspace"
ECHO_IN_BLUE "###############################################################"
cd ~/ros_catkin_ws
catkin config --init --install --space-suffix _isolated --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCATKIN_ENABLE_TESTING=0
catkin build
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Add ROS setup source to .bashrc"
ECHO_IN_BLUE "###############################################################"
SOURCE_LINE='source ~/ros_catkin_ws/install_isolated/setup.bash'
add_to_bashrc $SOURCE_LINE
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install system dependencies"
ECHO_IN_BLUE "###############################################################"
sudo apt install -y libasound2-dev \
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
    qt5-default \
    v4l-utils \
    libopenblas-dev \
    libpython3-dev \
    ffmpeg \
    chromium-browser
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install general Python dependencies"
ECHO_IN_BLUE "###############################################################"
sudo apt install -y 'libprotobuf*' protobuf-compiler ninja-build
sudo -H pip3 install numpy scipy numba cupy matplotlib google-cloud-texttospeech google-cloud-speech libconf tqdm sounddevice librosa audioread requests ipinfo pybind11-stubgen sphinx build
sudo -H pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install PyTorch for Jetson"
ECHO_IN_BLUE "###############################################################"
cd ~/deps
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
sudo -H pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

cd ~/deps
git clone --depth 1 -b v0.13.0 https://github.com/pytorch/vision.git
cd vision
sudo -H python3 setup.py install

cd ~/deps
git clone --depth 1 -b v0.12.0 https://github.com/pytorch/audio.git --recurse-submodule
cd audio
add_to_root_bashrc 'export PATH=/usr/local/cuda-11.4/bin:$PATH'
add_to_root_bashrc 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH'
sudo -H pip3 install -r requirements.txt
sudo -H bash -c 'TORCH_CUDA_ARCH_LIST="7.2;8.7" CUDACXX=/usr/local/cuda/bin/nvcc python3 setup.py install'

cd ~/deps
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo -H python3 setup.py install --plugins
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build the T-Top workspace"
ECHO_IN_BLUE "###############################################################"
cd $TTOP_REPO_PATH/../..
catkin config --init --cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
catkin config -profile release --init --space-suffix _release --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

catkin build
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setup user autologin and disable utomatic sleep and screen lock"
ECHO_IN_BLUE "###############################################################"
perl -pi -e "s/\@\@USER\@\@/$USER/" $PATCH_FILES_DIR/gdm3_config.patch
sudo patch -u /etc/gdm3/custom.conf $PATCH_FILES_DIR/gdm3_config.patch

gsettings set org.gnome.desktop.screensaver ubuntu-lock-on-suspend 'false'
gsettings set org.gnome.desktop.screensaver lock-delay 0
gsettings set org.gnome.session idle-delay 0
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Configure screen orientation and touchscreen calibration"
ECHO_IN_BLUE "###############################################################"
xrandr --output $(xrandr | grep HDMI | cut -d" " -f1) --rotate right
patch -u /usr/share/X11/xorg.conf.d/40-libinput.conf $PATCH_FILES_DIR/40-libinput.patch

#! /usr/bin/bash

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

if [ -t 1 ]; then
    ECHO_IN_BLUE () {
        echo -e ${BLUE}${1}${NC}
    }
    ECHO_IN_GREEN () {
        echo -e ${GREEN}${1}${NC}
    }
    ECHO_IN_ORANGE () {
        echo -e ${ORANGE}${1}${NC}
    }
else
    ECHO_IN_BLUE () {
        echo ${1}
    }
    ECHO_IN_GREEN () {
        echo ${1}
    }
    ECHO_IN_ORANGE () {
        echo ${1}
    }
fi

sudo_stay_validated () {
    while true; do
        sudo -v
        sleep 60
    done
}

cmake_build_install_native () {
    # arg 1 [optional]: number of threads to use (-j)
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
    if [ $# -lt 1 ] ; then
        cmake --build .
    else
        cmake --build . -j$1
    fi
    sudo cmake --install .
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

is_xavier () {
    cat /proc/device-tree/compatible | grep -q "jetson-xavier"
    if [ $? -eq 0 ] ; then
        echo true
    else
        echo false
    fi
}

clone_git () {
    # arg 1: git clone command
    local FOLDER=$(echo $@ | perl -pe 's|.*/(.*)\.git.*|$1|')
    if [ ! -d "$FOLDER/.git" ] ; then
        git clone $@
    fi
}

apply_patch () {
    patch --dry-run -uN $@ | grep --quiet --no-messages "previously applied.*Skipping patch" || patch -u $@
}

sudo_apply_patch () {
    sudo patch --dry-run -uN $@ | grep --quiet --no-messages "previously applied.*Skipping patch" || sudo patch -u $@
}

_STAMPNUM=0

checkstamp () {
    # arg 1: file to check
    if [ -f ~/.ttop/install_stamps/*$1 ] ; then
        echo true
    else
        echo false
    fi
}

makestamp () {
    # arg 1: file to create
    mkdir -p ~/.ttop/install_stamps
    filename=$(printf "%02d" $_STAMPNUM)_$1
    touch ~/.ttop/install_stamps/$filename
    export _STAMPNUM=$(($_STAMPNUM + 1))
}

stepstamp () {
    export _STAMPNUM=$(($_STAMPNUM + 1))
}

SKIP_SECTION () {
    # arg 1: message
    ECHO_IN_ORANGE "$1"
    stepstamp
}

ECHO_IN_GREEN "###############################################################"
ECHO_IN_GREEN "T-Top Setup Script"
ECHO_IN_GREEN "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Enter sudo password for the whole script"
ECHO_IN_BLUE "###############################################################"
sudo -v && sudo_stay_validated &
SUDO_KEEPALIVE_PID=$!
trap "kill ${SUDO_KEEPALIVE_PID}" EXIT
trap "exit" INT TERM QUIT KILL
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Update and upgrade system"
ECHO_IN_BLUE "###############################################################"
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Cloning the T-Top repo"
ECHO_IN_BLUE "###############################################################"
TTOP_REPO_PATH=~/t-top_ws/src/t-top
SETUP_SCRIPTS_DIR=$TTOP_REPO_PATH/tools/setup_scripts
PATCH_FILES_DIR=$SETUP_SCRIPTS_DIR/patch

if [ $(checkstamp ttop_repo) = "false" ] ; then
    mkdir -p ~/t-top_ws/src
    cd ~/t-top_ws/src
    # TODO remove -b t-top-v4
    clone_git --recurse-submodules https://github.com/introlab/t-top.git -b t-top-v4
    makestamp ttop_repo
else
    SKIP_SECTION "T-Top repo already cloned, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setting Jetson power mode"
ECHO_IN_BLUE "###############################################################"
if [ is_xavier = "true" ] ; then
    sudo nvpmodel -m 0
else
    grep --quiet --no-messages --fixed-regexp -- "MODE_38_8_W" /etc/nvpmodel.conf || sudo cp /etc/nvpmodel.conf /etc/nvpmodel/nvpmodel.conf.backup
    grep --quiet --no-messages --fixed-regexp -- "MODE_38_8_W" /etc/nvpmodel.conf || sudo cp $SETUP_SCRIPTS_DIR/files/jetson_orin_nvpmodel.conf /etc/nvpmodel.conf
    # Make sure the change to zero is applied by going to 1
    sudo nvpmodel -m 1 > /dev/null
    sudo nvpmodel -m 0
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Disabling sudo password for shutdown and nvpmodel"
ECHO_IN_BLUE "###############################################################"
sudo cp $SETUP_SCRIPTS_DIR/files/sudoers_ttop /etc/sudoers.d/ttop
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
ECHO_IN_BLUE ">> Setup user autologin and disable automatic sleep and screen lock"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp autologin) = "false" ] ; then
    perl -pe "s/\@\@USER\@\@/$USER/" $PATCH_FILES_DIR/gdm3_config.patch > $PATCH_FILES_DIR/gdm3_config.patch.tmp
    sudo_apply_patch /etc/gdm3/custom.conf $PATCH_FILES_DIR/gdm3_config.patch.tmp
    rm $PATCH_FILES_DIR/gdm3_config.patch.tmp

    gsettings set org.gnome.desktop.screensaver ubuntu-lock-on-suspend 'false'
    gsettings set org.gnome.desktop.screensaver lock-delay 0
    gsettings set org.gnome.desktop.session idle-delay 0

    makestamp autologin
else
    SKIP_SECTION "Autologin and automatic sleep and screen lock already setup, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Configure screen orientation and touchscreen calibration"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp screen_config) = "false" ] ; then

    sudo_apply_patch /usr/share/X11/xorg.conf.d/40-libinput.conf $PATCH_FILES_DIR/40-libinput.patch

    if [ $(xrandr | grep 'HDMI.* connected' | cut -d" " -f1 | wc -l) -eq 1 ] ; then
        xrandr --output $(xrandr | grep 'HDMI.* connected' | cut -d" " -f1) --rotate right
    elif [ $(xrandr | grep 'DP.* connected' | cut -d" " -f1 | wc -l) -eq 1 ] ; then
        xrandr --output $(xrandr | grep 'DP.* connected' | cut -d" " -f1) --rotate right
    else
        echo "ERROR: No external display detected"
        # Will fail the script
        return 1
    fi

    makestamp screen_config
else
    SKIP_SECTION "Screen already configured, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Installing NPM and Node"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp node) = "false" ] ; then
    sudo apt install -y curl software-properties-common
    curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
    sudo apt install -y nodejs
    makestamp node
else
    SKIP_SECTION "Node already installed, skipping"
fi
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
if [ $(checkstamp cmake) = "false" ] ; then
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt update
    sudo apt install -y cmake
    makestamp cmake
else
    SKIP_SECTION "CMake already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Cloning Librealsense 2"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp librealsense_clone) = "false" ] ; then
    mkdir -p ~/deps
    cd ~/deps
    clone_git https://github.com/jetsonhacks/buildLibrealsense2Xavier.git
    makestamp librealsense_clone
else
    SKIP_SECTION "Librealsense 2 already cloned, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Patching Librealsense 2"
ECHO_IN_BLUE "###############################################################"
cd ~/deps/buildLibrealsense2Xavier
apply_patch installLibrealsense.sh $PATCH_FILES_DIR/installLibrealsense.patch
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Building and installing Librealsense 2"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp librealsense_build) = "false" ] ; then
    cd ~/deps/buildLibrealsense2Xavier
    # Installed later on, but will cause a failure if installed now
    sudo apt autoremove -y libapriltag-dev
    yes | ./installLibrealsense.sh
    makestamp librealsense_build
else
    SKIP_SECTION "Librealsense 2 already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ROS build dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_build_deps) = "false" ] ; then

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
        libfltk1.3-dev \
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
        libapriltag-dev \
        libdc1394-22-dev
    makestamp ros_build_deps
else
    SKIP_SECTION "ROS build dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ROS system dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_system_deps) = "false" ] ; then

    cd ~/deps
    clone_git https://github.com/ros/console_bridge.git
    cd console_bridge
    cmake_build_install_native

    cd ~/deps
    clone_git https://github.com/ethz-asl/libnabo.git
    cd libnabo
    cmake_build_install_native

    cd ~/deps
    clone_git https://github.com/ethz-asl/libpointmatcher.git
    cd libpointmatcher
    cmake_build_install_native

    cd ~/deps
    clone_git -b 0.20.18-noetic https://github.com/introlab/rtabmap.git
    cd rtabmap
    cmake_build_install_native 4

    makestamp ros_system_deps
else
    SKIP_SECTION "ROS system dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Generate ROS build workspace and install dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_ws_deps) = "false" ] ; then

    if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ] ; then
        sudo rosdep init
    fi
    rosdep update

    mkdir -p ~/ros_catkin_ws/src
    cd ~/ros_catkin_ws

    rosinstall_generator desktop_full --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
    vcs import --input noetic-desktop.rosinstall ./src
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y

    makestamp ros_ws_deps
else
    SKIP_SECTION "ROS workspace dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Remove useless ROS packages"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_deps_rm) = "false" ] ; then
    rm -rf ~/ros_catkin_ws/src/gazebo_ros_pkgs/
    makestamp ros_deps_rm
else
    SKIP_SECTION "Useless ROS packages already removed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Add missing ROS packages"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_deps_add) = "false" ] ; then

    cd ~/ros_catkin_ws/src
    clone_git -b 1.0.1 https://github.com/GT-RAIL/rosauth.git
    clone_git -b noetic-devel https://github.com/ros-drivers/rosserial.git
    clone_git -b ros1 https://github.com/RobotWebTools/rosbridge_suite.git
    clone_git -b noetic https://github.com/ccny-ros-pkg/imu_tools.git
    clone_git --recursive https://github.com/orocos/orocos_kinematics_dynamics.git

    clone_git -b 0.20.18-noetic https://github.com/introlab/rtabmap_ros.git
    clone_git -b noetic-devel https://github.com/ros-planning/navigation.git

    clone_git -b kinetic-devel https://github.com/pal-robotics/ddynamic_reconfigure.git
    clone_git -b 2.3.2 https://github.com/IntelRealSense/realsense-ros.git
    clone_git https://github.com/OTL/cv_camera.git
    clone_git -b 0.6.4-noetic https://github.com/introlab/find-object.git

    makestamp ros_deps_add
else
    SKIP_SECTION "Missing ROS packages already added, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Replace incomplete ROS packages"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_deps_replace) = "false" ] ; then

    cd ~/ros_catkin_ws/src
    rm -rf geometry2 navigation_msgs vision_opencv image_common perception_pcl pcl_msgs image_transport_plugins

    clone_git -b noetic-devel https://github.com/ros/geometry2.git
    clone_git -b ros1 https://github.com/ros-planning/navigation_msgs.git
    clone_git -b noetic https://github.com/ros-perception/vision_opencv.git
    clone_git -b noetic-devel https://github.com/ros-perception/image_common.git

    clone_git -b 1.7.1 https://github.com/ros-perception/perception_pcl.git
    clone_git -b noetic-devel https://github.com/ros-perception/pcl_msgs.git
    clone_git -b noetic-devel https://github.com/ros-perception/image_transport_plugins.git

    rosdep install --from-paths ./image_transport_plugins --ignore-packages-from-source --rosdistro noetic -y

    makestamp ros_deps_replace
else
    SKIP_SECTION "Incomplete ROS packages already replaced, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build ROS workspace"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_ws_build) = "false" ] ; then
    cd ~/ros_catkin_ws
    apt list --installed | grep --quiet --no-messages --fixed-regexp "python3.9/" && sudo apt autoremove -y python3.9 || true
    catkin config --init --install --space-suffix _isolated --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCATKIN_ENABLE_TESTING=0 -Wno-dev
    catkin build

    makestamp ros_ws_build
else
    SKIP_SECTION "ROS workspace already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Add ROS setup source to .bashrc"
ECHO_IN_BLUE "###############################################################"
add_to_bashrc 'source ~/ros_catkin_ws/install_isolated/setup.bash'
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install system dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_system_deps) = "false" ] ; then
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
        chromium-browser \
        libqt5websockets5-dev \
        libqt5charts5-dev

    makestamp ttop_system_deps
else
    SKIP_SECTION "T-Top system dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install general Python dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_python_deps) = "false" ] ; then
    sudo apt install -y 'libprotobuf*' protobuf-compiler ninja-build
    sudo -H pip3 install numpy scipy numba cupy matplotlib google-cloud-texttospeech google-cloud-speech libconf tqdm sounddevice librosa audioread requests ipinfo pybind11-stubgen sphinx build
    sudo -H pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

    makestamp ttop_python_deps
else
    SKIP_SECTION "T-Top Python dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install PyTorch for Jetson"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp pytorch) = "false" ] ; then

    cd ~/deps
    wget -N https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
    sudo -H pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

    cd ~/deps
    clone_git --depth 1 -b v0.13.0 https://github.com/pytorch/vision.git
    cd vision
    sudo -H python3 setup.py install

    cd ~/deps
    clone_git --depth 1 -b v0.12.0 https://github.com/pytorch/audio.git --recurse-submodule
    cd audio
    add_to_root_bashrc 'export PATH=/usr/local/cuda-11.4/bin:$PATH'
    add_to_root_bashrc 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH'
    sudo -H pip3 install -r requirements.txt
    sudo -H bash -c 'TORCH_CUDA_ARCH_LIST="7.2;8.7" CUDACXX=/usr/local/cuda/bin/nvcc python3 setup.py install'

    cd ~/deps
    clone_git https://github.com/NVIDIA-AI-IOT/torch2trt.git
    cd torch2trt
    sudo -H python3 setup.py install --plugins

    makestamp pytorch
else
    SKIP_SECTION "PyTorch already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install OpenTera-WebRTC ROS Python dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp opentera_deps) = "false" ] ; then
    cd $TTOP_REPO_PATH/ros/opentera-webrtc-ros/opentera_client_ros
    sudo -H pip3 install -r requirements.txt
    cd $TTOP_REPO_PATH/ros/opentera-webrtc-ros/opentera_webrtc_ros
    sudo -H pip3 install -r requirements.txt
    cd $TTOP_REPO_PATH/ros/opentera-webrtc-ros/opentera_webrtc_ros/opentera-webrtc
    sudo -H pip3 install -r requirements.txt

    makestamp opentera_deps
else
    SKIP_SECTION "OpenTera-WebRTC ROS Python dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build the T-Top workspace"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_ws_build) = "false" ] ; then
    cd $TTOP_REPO_PATH/../..
    catkin config --init --cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    catkin config --profile release --init --space-suffix _release --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    catkin build

    makestamp ttop_ws_build
else
    SKIP_SECTION "T-Top workspace already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

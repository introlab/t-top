#! /usr/bin/bash

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

TTOP_REPO_PATH=~/t-top_ws/src/t-top
SETUP_SCRIPTS_DIR=$TTOP_REPO_PATH/tools/setup_scripts
PATCH_FILES_DIR=$SETUP_SCRIPTS_DIR/patch

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
    ECHO_IN_RED () {
        echo -e ${RED}${1}${NC}
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
    ECHO_IN_RED () {
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

get_jetson_model () {
    $TTOP_REPO_PATH/tools/jetson_model/get_jetson_model.py
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
sudo -v
sudo_stay_validated &
SUDO_KEEPALIVE_PID=$!
trap "kill ${SUDO_KEEPALIVE_PID}" EXIT
trap "exit" INT TERM KILL
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
if [ $(checkstamp ttop_repo) = "false" ] ; then
    mkdir -p ~/t-top_ws/src
    cd ~/t-top_ws/src
    clone_git --recurse-submodules https://github.com/introlab/t-top.git
    makestamp ttop_repo
else
    SKIP_SECTION "T-Top repo already cloned, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

JETSON_MODEL=$(get_jetson_model)

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setting Jetson power mode"
ECHO_IN_BLUE "###############################################################"
if [ $JETSON_MODEL = "xavier" ] ; then
    sudo nvpmodel -m 0
elif [ $JETSON_MODEL = "orin" ] ; then
    grep --quiet --no-messages --fixed-regexp -- "MODE_38_8_W" /etc/nvpmodel.conf || sudo cp /etc/nvpmodel.conf /etc/nvpmodel/nvpmodel.conf.backup
    grep --quiet --no-messages --fixed-regexp -- "MODE_38_8_W" /etc/nvpmodel.conf || sudo cp $SETUP_SCRIPTS_DIR/files/jetson_orin_nvpmodel.conf /etc/nvpmodel.conf
    # Make sure the change to zero is applied by going to 1
    sudo nvpmodel -m 1 &> /dev/null
    sudo nvpmodel -m 0
elif [ $JETSON_MODEL = "not_jetson" ] ; then
    : # Not a Jetson, we don't do anything
else
    ECHO_IN_RED "Setting power mode not implemented for unknown jetson model [$JETSON_MODEL]"
    exit 1
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

    # TODO rotate the screen here using way that's scripted and persistent on reboot

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
if [ $(checkstamp install_tools) = "false" ] ; then
    sudo apt install -y htop python3-pip perl rsync scons
    sudo -H pip3 install -U jetson-stats
    makestamp install_tools
else
    SKIP_SECTION "Tools already installed, skipping"
fi
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

    sudo apt install -y \
        python3-rosdep \
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
    clone_git https://github.com/ethz-asl/libnabo.git -b 1.0.7
    cd libnabo
    cmake_build_install_native

    cd ~/deps
    clone_git https://github.com/ethz-asl/libpointmatcher.git -b 1.3.1
    cd libpointmatcher
    cmake_build_install_native

    cd ~/deps
    clone_git -b 0.21.1-noetic https://github.com/introlab/rtabmap.git
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

    clone_git -b 0.21.1-noetic https://github.com/introlab/rtabmap_ros.git
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

    cd ~/ros_catkin_ws
    rosdep install --from-paths ./src/image_transport_plugins --ignore-packages-from-source --rosdistro noetic -y

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
    sudo apt install -y \
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
        qt5-default \
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
        libspdlog-dev

    makestamp ttop_system_deps
else
    SKIP_SECTION "T-Top system dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ONNX Runtime"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp onnxruntime) = "false" ] ; then
    sudo -H pip3 install packaging==23.1
    cd ~/deps
    clone_git --depth 1 -b v1.14.1 https://github.com/microsoft/onnxruntime.git --recurse-submodule
    cd onnxruntime
    ./build.sh --config Release --update --build --parallel --build_wheel --build_shared_lib --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu
    cd build/Linux/Release
    sudo make install

    makestamp onnxruntime
else
    SKIP_SECTION "ONNX Runtime already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ARM Compute Library "
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp armcomputelibrary) = "false" ] ; then
    cd ~/deps
    clone_git --depth 1 -b v22.11 https://github.com/ARM-software/ComputeLibrary.git
    cd ComputeLibrary
    scons Werror=1 -j8 debug=0 neon=1 opencl=0 os=linux arch=armv8a build=native
    mv build lib

    makestamp armcomputelibrary
else
    SKIP_SECTION "ARM Compute Library already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install oneDNN "
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp onednn) = "false" ] ; then
    cd ~/deps
    clone_git --depth 1 -b v3.2.1 https://github.com/oneapi-src/oneDNN.git
    cd oneDNN
    mkdir -p build
    cd build
    export ACL_ROOT_DIR=~/deps/ComputeLibrary
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math -DDNNL_AARCH64_USE_ACL=ON"
    cmake --build . -j4
    sudo cmake --install .

    makestamp onednn
else
    SKIP_SECTION "oneDNN already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install general Python dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_python_deps) = "false" ] ; then
    sudo apt install -y \
        'libprotobuf*' \
        protobuf-compiler \
        ninja-build \
        python3-numpy \
        python3-scipy \
        python3-numba \
        python3-matplotlib \
        python3-sklearn \
        python3-tqdm \
        python3-audioread \
        python3-requests \
        python3-sphinx

    sudo -H pip3 install 'cython>=0.29.22,<0.30.0'
    sudo -H pip3 install -r $SETUP_SCRIPTS_DIR/files/requirements.txt

    makestamp ttop_python_deps
else
    SKIP_SECTION "T-Top Python dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install CTranslate2 "
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ctranslate2) = "false" ] ; then
    cd ~/deps
    clone_git --depth 1 -b v3.20.0 https://github.com/OpenNMT/CTranslate2.git --recurse-submodule
    cd CTranslate2
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DWITH_MKL=OFF -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_OPENBLAS=ON -DWITH_DNNL=ON -DWITH_RUY=ON
    cmake --build . -j4
    sudo cmake --install .
    sudo ldconfig
    cd ../python
    sudo -H pip3 install -r install_requirements.txt
    python3 setup.py bdist_wheel
    sudo -H pip3 install dist/*.whl --no-deps --force-reinstall

    makestamp ctranslate2
else
    SKIP_SECTION "CTranslate2 already installed, skipping"
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
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DWITH_CUDA=ON
    cmake --build . -j4
    sudo cmake --install .
    cd ~/deps/vision
    sudo -H python3 setup.py install

    cd ~/deps
    clone_git --depth 1 -b v0.12.0 https://github.com/pytorch/audio.git --recurse-submodule
    cd audio
    add_to_root_bashrc 'export PATH=/usr/local/cuda-11.4/bin:$PATH'
    add_to_root_bashrc 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH'
    sudo -H pip3 install kaldi_io==0.9.5
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

    makestamp opentera_deps
else
    SKIP_SECTION "OpenTera-WebRTC ROS Python dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build and install the T-Top hardware daemon"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_daemon) = "false" ] ; then
    cd $TTOP_REPO_PATH/system/daemon
    cmake_build_install_native

    sudo systemctl enable ttop_hardware_daemon.service
    sudo systemctl start ttop_hardware_daemon.service

    makestamp ttop_daemon
else
    SKIP_SECTION "T-Top daemon already built and installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build and install the T-Top hardware system tray"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_system_tray) = "false" ] ; then
    cd $TTOP_REPO_PATH/system/system_tray
    cmake_build_install_native

    makestamp ttop_system_tray
else
    SKIP_SECTION "T-Top system tray already built and installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build the T-Top workspace"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_ws_build) = "false" ] ; then
    source ~/.bashrc
    cd $TTOP_REPO_PATH/../..
    catkin config --init --cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    catkin config --profile release --init --space-suffix _release --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_WARN_DEPRECATED=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    catkin build

    makestamp ttop_ws_build
else
    SKIP_SECTION "T-Top workspace already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

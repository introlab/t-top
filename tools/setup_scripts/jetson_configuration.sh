#! /usr/bin/bash

export ROS_DISTRO=humble
export ROS_PACKAGE=desktop_full
export ROS_ROOT=/opt/ros/$ROS_DISTRO
export ROS_PYTHON_VERSION=3
export DEBIAN_FRONTEND=noninteractive
export PYTHONIOENCODING=utf-8

# Required for jetpack to find cuda
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

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
ECHO_IN_BLUE ">> Installing tools"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp install_tools) = "false" ] ; then
    sudo apt install -y htop python3-pip perl rsync scons
    sudo apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        ca-certificates \
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
    sudo -H pip3 install -U jetson-stats
    makestamp install_tools
else
    SKIP_SECTION "Tools already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setting System Locale to en_US.UTF-8"
ECHO_IN_BLUE "###############################################################"
sudo update-locale LANG=en_US.UTF-8
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Setting Python3 as default"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp python3_default) = "false" ] ; then
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
    makestamp python3_default
else
    SKIP_SECTION "Python3 is already the default, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"


ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Cloning the T-Top repo"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ttop_repo) = "false" ] ; then
    mkdir -p ~/t-top_ws/src
    cd ~/t-top_ws/src
    clone_git --recurse-submodules https://github.com/introlab/t-top.git -b ros2
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
    curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    sudo apt install -y nodejs
    makestamp node
else
    SKIP_SECTION "Node already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Install ROS2 build dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_build_deps) = "false" ] ; then

    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    sudo echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update

    sudo apt-get install -y --no-install-recommends \
		build-essential \
		libbullet-dev \
		libpython3-dev \
		python3-colcon-common-extensions \
		python3-flake8 \
		python3-pip \
		python3-numpy \
		python3-pytest-cov \
		python3-rosdep \
		python3-setuptools \
		python3-vcstool \
		python3-rosinstall-generator \
		libasio-dev \
		libtinyxml2-dev \
		libcunit1-dev

    sudo pip3 install --upgrade --no-cache-dir \
		argcomplete \
		flake8-blind-except \
		flake8-builtins \
		flake8-class-newline \
		flake8-comprehensions \
		flake8-deprecated \
		flake8-docstrings \
		flake8-import-order \
		flake8-quotes \
		pytest-repeat \
		pytest-rerunfailures \
		pytest

    sudo python3 -m pip install --upgrade pip
    sudo pip3 install --no-cache-dir scikit-build
    sudo pip3 install --upgrade --no-cache-dir --verbose cmake==3.22.1

    # remove other versions of Python3
    # workaround for 'Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS Development'
    sudo apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"

    makestamp ros_build_deps
else
    SKIP_SECTION "ROS build dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Generate ROS2 build workspace and install dependencies"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_ws_deps) = "false" ] ; then

    if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ] ; then
        sudo rosdep init
    fi
    rosdep update

    # create the ROS_ROOT directory
    mkdir -p ${ROS_ROOT}/src
    cd ${ROS_ROOT}

    rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PACKAGE} \
        launch_xml \
        launch_yaml \
        launch_testing \
        launch_testing_ament_cmake \
        demo_nodes_cpp \
        demo_nodes_py \
        example_interfaces \
        camera_calibration_parsers \
        camera_info_manager \
        cv_bridge \
        v4l2_camera \
        vision_opencv \
        vision_msgs \
        image_geometry \
        image_pipeline \
        image_transport \
        compressed_image_transport \
        compressed_depth_image_transport \
        rosbag2_storage_mcap \
        rtabmap \
        rtabmap_ros \
        diagnostics \
        imu_tools \
        rosbridge_suite \
        tf_transformations \
	    joint_state_publisher_gui \
        rqt_tf_tree \
        > ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall

    sudo vcs import --retry 100 src < ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall
    git -C ${ROS_ROOT}/src clone -b master https://github.com/Kapernikov/cv_camera.git --depth 1 --recurse-submodules

    SKIP_KEYS="libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv xsimd xtensor xtl"
    sudo rosdep update
    sudo rosdep install -y \
	--ignore-src \
	--from-paths src \
	--rosdistro ${ROS_DISTRO} \
	--skip-keys "$SKIP_KEYS"

    cd ${ROS_ROOT}/src
    clone_git -b 0.7.0 https://github.com/xtensor-stack/xtl.git --depth 1 --recurse-submodules
    clone_git -b 7.4.8 https://github.com/xtensor-stack/xsimd.git --depth 1 --recurse-submodules
    clone_git -b 0.23.10 https://github.com/xtensor-stack/xtensor --depth 1 --recurse-submodules

    sudo_apply_patch ${ROS_ROOT}/src/libg2o/CMakeLists.txt $PATCH_FILES_DIR/libg2o.patch
    sudo_apply_patch ${ROS_ROOT}/src/octomap_msgs/CMakeLists.txt $PATCH_FILES_DIR/octomap_msgs.patch

    makestamp ros_ws_deps
else
    SKIP_SECTION "ROS workspace dependencies already installed, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Build ROS workspace"
ECHO_IN_BLUE "###############################################################"
if [ $(checkstamp ros_ws_build) = "false" ] ; then
    cd ${ROS_ROOT}

    export ROS_VERSION=2
    sudo colcon build \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DCMAKE_PREFIX_PATH=$ROS_ROOT -DBUILD_WITH_CUDA=true -DBUILD_TESTING=OFF

    makestamp ros_ws_build
else
    SKIP_SECTION "ROS workspace already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

ECHO_IN_BLUE "###############################################################"
ECHO_IN_BLUE ">> Add ROS setup source to .bashrc"
ECHO_IN_BLUE "###############################################################"
add_to_bashrc 'source /opt/ros/humble/install/setup.bash'
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

    mkdir -p ~/.colcon
    cp $SETUP_SCRIPTS_DIR/files/colcon_defaults.yaml $TTOP_REPO_PATH/../../colcon_defaults.yaml

    colcon build

    makestamp ttop_ws_build
else
    SKIP_SECTION "T-Top workspace already built, skipping"
fi
ECHO_IN_BLUE "###############################################################\n"

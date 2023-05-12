# Computer Configuration

## Onboard Computer - Jetson AGX Xavier

### A. Install the SSD

#### Required Parts

| Part                                  | Quantity | Image                                                                                        |
| ------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `Nvidia Jetson AGX Xavier`            | 1        | ![Nvidia Jetson AGX Xavier](images/electronics/jetson-agx-xavier.jpg)                        |
| `SSD`                                 | 1        | ![SSD](images/electronics/SSD.jpg)                                                           |

#### Steps

1. Remove the following screws.

![Nvidia Jetson AGX Xavier Screws](images/assemblies/01A%20screws.jpg)

2. Remove the bottom PCB.
3. Install the SSD, as shown in the following picture.

![Nvidia Jetson AGX Xavier SSD](images/assemblies/01A%20SSD.jpg)

### B. Remove dV/dt protection feature (Jetson AGX Xavier)

#### Required Parts

| Part                                  | Quantity | Image                                                                                        |
| ------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `Nvidia Jetson AGX Xavier`            | 1        | ![Nvidia Jetson AGX Xavier](images/electronics/jetson-agx-xavier.jpg)                        |

#### Steps

1. Remove R135 (see [P2822_B03_PCB_assembly_drawing.pdf](http://developer.nvidia.com/embedded/dlc/jetson-xavier-developer-kit-carrier-board-design-files-b03)).
2. Replace the bottom PCB.
1. Install the following screws.

![Nvidia Jetson AGX Xavier Screws](images/assemblies/01A%20screws.jpg)

### C. Install the WiFi Card (Jetson AGX Xavier)

#### Required Parts

| Part                                  | Quantity | Image                                                                                        |
| ------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `Nvidia Jetson AGX Xavier`            | 1        | ![Nvidia Jetson AGX Xavier](images/electronics/jetson-agx-xavier.jpg)                        |
| `WiFi Card`                           | 1        | ![WiFi Card](images/electronics/wifi-card.jpg)                                               |

#### Steps

1. Install the `WiFi card` into the `Nvidia Jetson AGX Xavier`.

![WiFi Card](images/assemblies/01C%20wifi%20card.jpg)

### D. Install JetPack 5.0.2

1. Install JetPack 5.0.2 onto the computer SSD.

### E. OpenCR Dev Rule

1. Copy [99-opencr-custom.rules](../../tools/udev_rules/99-opencr-custom.rules) in `/etc/udev/rules.d/`.
2. Copy [99-teensy.rules](../../tools/udev_rules/99-teensy.rules) in `/etc/udev/rules.d/`.
2. Copy [99-camera-2d-wide.rules](../../tools/udev_rules/99-camera-2d-wide.rules) in `/etc/udev/rules.d/`.
3. Add the user to the `dialout` group.

```bash
sudo usermod -a -G dialout $USER
```

### F. Install NPM
1. Execute the following bash commands.

```bash
sudo apt install -y curl software-properties-common
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt install -y nodejs
```

### G. Install Tools
1. Execute the following bash commands.

```bash
sudo apt install htop python3-pip
sudo -H pip3 install -U jetson-stats

# Update CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install kitware-archive-keyring
sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
sudo apt update
sudo apt install cmake
```

### H. Install Librealsense 2
1. Execute the following bash commands.

```bash
cd ~/deps

git clone https://github.com/jetsonhacks/buildLibrealsense2Xavier
cd buildLibrealsense2Xavier
```

2. Change the version to `v2.50.0` in `installLibrealsense.sh`.
3. Add the folowing arguments to the `cmake` command in `installLibrealsense.sh`.

```bash
-DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
```

4. Execute the following bash commands.

```bash
./installLibrealsense.sh
```

### I. Install ROS
1. Execute the following bash commands.

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update

sudo apt install -y python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
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

# Install system dependencies
cd ~/deps
git clone https://github.com/ros/console_bridge
cd console_bridge
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
make -j
sudo make install

cd ~/deps
git clone https://github.com/ethz-asl/libnabo.git
cd libnabo
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
make -j
sudo make install

cd ~/deps
git clone https://github.com/ethz-asl/libpointmatcher.git
cd libpointmatcher
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
make -j
sudo make install

cd ~/deps
git clone -b 0.20.18-noetic https://github.com/introlab/rtabmap.git
cd rtabmap/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
make -j4
sudo make install

# Install ROS
sudo rosdep init
rosdep update

mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws

rosinstall_generator desktop_full --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
mkdir ./src
vcs import --input noetic-desktop.rosinstall ./src
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y

# Remove useless packages
rm -rf ~/ros_catkin_ws/src/gazebo_ros_pkgs/

# Add ROS packages
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

# Replace not complete packages
rm -rf geometry2 navigation_msgs vision_opencv image_common

git clone -b noetic-devel https://github.com/ros/geometry2.git
git clone -b ros1 https://github.com/ros-planning/navigation_msgs.git
git clone -b noetic https://github.com/ros-perception/vision_opencv.git
git clone -b noetic-devel https://github.com/ros-perception/image_common.git

cd ~/ros_catkin_ws
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCATKIN_ENABLE_TESTING=0

# Add ROS setup to .bashrc
echo "source ~/ros_catkin_ws/install_isolated/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### J. Install System Dependancies

1. Execute the following bash commands.

```bash
sudo apt install -y libasound2-dev \
    libpulse-dev \
    libconfig-dev \
    alsa-utils \
    gfortran \
    libgfortran-*-dev \
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
```

### K. Install Python Dependencies

1. Execute the following bash commands.

```bash
# Install general dependencies
sudo apt install -y 'libprotobuf*' protobuf-compiler ninja-build
sudo -H pip3 install numpy scipy numba cupy matplotlib google-cloud-texttospeech google-cloud-speech libconf tqdm sounddevice librosa audioread requests ipinfo pybind11-stubgen sphinx build
sudo -H pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install PyTorch for Jetson
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
sudo bash -c 'echo "export PATH=/usr/local/cuda-11.4/bin:\$PATH" >> /root/.bashrc'
sudo bash -c 'sudo echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc'
sudo -H pip3 install -r requirements.txt
sudo -H bash -c 'TORCH_CUDA_ARCH_LIST="7.2;8.7" CUDACXX=/usr/local/cuda/bin/nvcc python3 setup.py install'

cd ~/deps
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo -H python3 setup.py install --plugins
```

### L. Clone and Build the Repository

1. Execute the following bash commands.

```bash
mkdir ~/t-top_ws
cd ~/t-top_ws
mkdir src
catkin_make

cd src
git clone --recurse-submodules git@github.com:introlab/t-top.git
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffast-math" -DCMAKE_C_FLAGS="-march=native -ffast-math"
```

### Setup Screen

1. Rotate the display in the settings application.
2. Add `Option "CalibrationMatrix" "0 1 0 -1 0 1 0 0 1"` before `EndSection` in the following section of `/usr/share/X11/xorg.conf.d/40-libinput.conf`.

```
Section "InputClass"
        Identifier "libinput touchscreen catchall"
        MatchIsTouchscreen "on"
        MatchDevicePath "/dev/input/event*"
        Driver "libinput"
EndSection
```

## Onboard Computer - Jetson AGX Orin
**TODO**

## Development Computer (Ubuntu 20.04)

### A. OpenCR Dev Rule

1. Copy [99-opencr-custom.rules](../../tools/udev_rules/99-opencr-custom.rules) in `/etc/udev/rules.d/`.
2. Copy [99-teensy.rules](../../tools/udev_rules/99-teensy.rules) in `/etc/udev/rules.d/`.
2. Copy [99-camera-2d-wide.rules](../../tools/udev_rules/99-camera-2d-wide.rules) in `/etc/udev/rules.d/`.
3. Add the user to the `dialout` group.

```bash
sudo usermod -a -G dialout $USER
```

### B. Install NPM

1. Execute the following bash commands.

```bash
sudo apt install -y curl software-properties-common
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt install -y nodejs
```

### C. Install ROS

1. Execute the following bash commands.

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo rosdep init
rosdep update
sudo apt install -y python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    ros-noetic-rosserial-python \
    ros-noetic-rosbridge-suite \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-imu-filter-madgwick \
    ros-noetic-cv-bridge \
    ros-noetic-rtabmap-ros \
    ros-noetic-cv-camera
```

### D. Install System Dependancies

1. Execute the following bash commands.

```bash
sudo apt install -y libasound2-dev \
    libpulse-dev \
    libconfig-dev \
    alsa-utils \
    gfortran \
    libgfortran-*-dev \
    texinfo \
    libfftw3-dev \
    libsqlite3-dev \
    portaudio19-dev \
    python3-all-dev \
    libgecode-dev \
    qt5-default \
    v4l-utils
```

### E. Install Python Dependencies

1. Execute the following bash commands.

```bash
sudo apt install -y 'libprotobuf*' protobuf-compiler ninja-build python3-pip python3-sklearn
sudo -H pip3 install numpy scipy matplotlib torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 google-cloud-texttospeech google-cloud-speech libconf tqdm pyside2 sounddevice librosa requests ipinfo
sudo -H pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### F. Install CUDA Tools (Optional)

1. Install [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
2. Install torch2trt.

```bash
cd ~/deps
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins
```

3. Install CuPy.

```bash
sudo -H pip3 install cupy
```

### G. Clone the Repository

1. Execute the following bash commands.

```bash
mkdir ~/t-top_ws
cd ~/t-top_ws
mkdir src
catkin_make

cd src
git clone --recurse-submodules git@github.com:introlab/t-top.git
```

### H. Install the RealSense Packages
1. Install [Librealsense](https://github.com/IntelRealSense/librealsense/blob/v2.39.0/doc/installation.md) (use the tag
   v2.50.0).
2. Install [ROS node](https://github.com/IntelRealSense/realsense-ros#step-2-install-intel-realsense-ros-from-sources)
   in `~/t-top_ws/src` (use the tag 2.3.2)

## All computer

### A. Setup Google Cloud

1. Create
   a [service account JSON keyfile](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating) for
   Google Cloud Text-to-Speech and Google Cloud Speech-to-Text.
2. Add the following line to `~/.bashrc`

```bash
export GOOGLE_APPLICATION_CREDENTIALS="[Path to the service account JSON keyfile]"
```

### B. Setup OpenWeatherMap

1. Create an [OpenWeatherMap API Key](https://home.openweathermap.org/users/sign_up).
2. Add the following line to `~/.bashrc`

```bash
export OPEN_WEATHER_MAP_API_KEY="[The key]"
```

# Computer Configuration

## Onboard Computer (Jetson AGX Xavier)

TODO

## Development Computer (Ubuntu 20.04)

### A. OpenCR Dev Rule

1. Copy [100-opencr-custom.rules](../../firmwares/opencr_firmware/100-opencr-custom.rules) in `/etc/udev/rules.d/`.
2. Add the user to the `dialout` group.

```bash
sudo usermod -a -G dialout $USER
```

### B. Install NPM

1. Execute the following bash commands.

```bash
sudo apt install -y curl software-properties-common
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
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
    ros-noetic-rtabmap-ros
```

### D. Install System Dependancies

1. Execute the following bash commands.

```bash
sudo apt install -y libasound2-dev \
    libconfig-dev \
    alsa-utils \
    gfortran \
    texinfo \
    libfftw3-dev \
    libsqlite3-dev \
    portaudio19-dev \
    python3-all-dev \
    libgecode-dev \
    qt5-default
```

### E. Install Python Dependancies

1. Execute the following bash commands.

```bash
sudo apt install -y 'libprotobuf*' protobuf-compiler ninja-build python3-pip
sudo -H pip3 install numpy scipy matplotlib torch==1.6.0 torchvision==0.7.0 torchaudio==0.6.0 playsound google-cloud-texttospeech google-cloud-speech libconf tqdm pyside2 sounddevice librosa requests ipinfo
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
git clone --recurse-submodules git@github.com:introlab/tabletop_robot.git
```

### H. Install the RealSense Packages

1. Install [Librealsense](https://github.com/IntelRealSense/librealsense/blob/v2.39.0/doc/installation.md) (use the tag
   v2.39.0).
2. Install [ROS node](https://github.com/IntelRealSense/realsense-ros#step-2-install-intel-realsense-ros-from-sources)
   in `~/t-top_ws/src`  (use the tag 2.2.18)

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

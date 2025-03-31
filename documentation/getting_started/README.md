# Getting Started

This is a guide to help you get started with the project. It covers the basic setup, configuration, and usage of the project.
It is recommended to follow the steps in order to ensure a smooth experience. All the steps assume that you are working directly on the robot using the terminal or Visual Studio Code with SSH.

## Prerequisites

- A T-Top robot with Jetson Orin AGX or Xavier AGX with Jetpack 5.x.x
- ROS 2 Humble pre-installed on the robot using the [setup script](../../tools/setup_scripts/ros2_humble_install.sh).
- Visual Studio Code installed on your computer.

- Install Visual Studio Code Extensions (From the Extensions tab or menu in Visual Studio Code):
  - [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
  - [ROS 2 extension pack](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros)
  - [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
  - [CMake Tools extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)

## Setup SSH keys for easy access

1. Open a terminal on your computer and create SSH key.

> Skip this step if you already have SSH keys configured.

```bash
# Go to your ssh directory
cd ~/.ssh
# Generate a new SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"
# Copy the public key to the robot
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@robot_ip_address
# Test the SSH connection, this should not ask for a password
ssh username@robot_ip_address
```

2. Configure SSH config file for even easier access.

```bash
# Open the SSH config file
nano ~/.ssh/config
```

Add the following lines to the config file, replacing `username` and `robot_ip_address` with your own values.

```bash
Host ttop
    HostName robot_ip_address
    User username
    IdentityFile ~/.ssh/id_ed25519
```

Now you can SSH into the robot using the command `ssh ttop` instead of `ssh username@robot_ip_address`.

## Setup Workspace from Visual Studio Code

1. Open Visual Studio Code and click on the `-[]-` icon in the bottom left corner.
2. Select `Remote-SSH: Connect to Host...` and choose `ttop` from the list.
3. Once connected, open the terminal in Visual Studio Code by clicking on `Terminal` in the top menu and selecting `New Terminal`.
4. In the terminal, we will create a new workspace folder for the project. You can choose any name you like, but for this example, we will use `t_top_ws`. If the directory already exists, create a new one with another name :

```bash
# Create the workspace folder
mkdir -p ~/t_top_ws/src
cd ~/t_top_ws/src
# Clone the repository
git clone https://github.com/introlab/t-top.git --recurse-submodules
```

5. Go back to the root of the workspace to create the colcon configuration file

```bash
# Go to the root of the workspace
cd ~/t_top_ws
# Create the colcon_defaults.yaml file
touch colcon_defaults.yaml
```

6. Open the `colcon_defaults.yaml` file and add the following lines:

```yaml
# colcon_defaults.yaml
build:
  cmake-args:
    - -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    - --no-warn-unused-cli
    - -DPYTHON_EXECUTABLE=/usr/bin/python3
    - -DCMAKE_BUILD_TYPE=Debug
    - -DCMAKE_CXX_FLAGS=-march=native -ffast-math
    - -DCMAKE_C_FLAGS=-march=native -ffast-math
  symlink-install: true
```

6. Build the workspace.

```bash
# Go to the root of the workspace
cd ~/t_top_wsf
# Source ROS2 Humble
source /opt/ros/humble/install/setup.bash
# Build the workspace
colcon build
```

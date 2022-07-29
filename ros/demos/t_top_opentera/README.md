# t_top_opentera

This folder contains a node that makes T-Top a telepresence robot using OpenTera.


## Installation
1. Follow the installation guide for [T-Top](../../../documentation/assembly/01_COMPUTER_CONFIGURATION.md) dependencies.
2. Install the requirements for OpenTera from the [system](../../opentera-webrtc-ros/README.md#Requirements) and for [Python](../../opentera-webrtc-ros/README.md#3---Install-the-Python-requirements).
3. Build the ROS workspace using `catkin_make`.

## How to Launch
The OpenTera telepresence stack can be launched in stand-alone mode or using the OpenTera servers.

### Stand-alone mode
To launch in stand-alone mode, you will need to generate CA certificates and keys. Follow the guide [here](../../../tools/ca_certificates/ca_certificate_setup.md).

1. Start the ROS stack.
```bash
export DISPLAY=:0.0
roslaunch t_top_opentera t_top_opentera.launch
```
2. On your developpement machine, navigate to `https://<ROBOT_IP>:8080/index.html#/user?pwd=abc&robot=TTOP`, where `<ROBOT_IP>` is the IP address of your robot on the LAN.


### OpenTera Server
To launch using the OpenTera servers, you will need to configure the OpenTera servers:
1. Generate a unique token for the robot using `OpenTeraPlus`.
2. Copy the template config file to the destination:
```bash
mkdir -p ~/.ros/opentera
cp $(rospack find opentera_client_ros)/config/client_config.json ~/.ros/opentera/client_config.json
```
3. Edit the file `~/.ros/opentera/client_config.json` to set the token and url to the token from step 1 and `https://telesante.3it.usherbrooke.ca:40075`, respectively.

1. Start the ROS stack.
```bash
export DISPLAY=:0.0
roslaunch t_top_opentera t_top_opentera_online.launch
```

2. On your developpement machine, navigate to `https://telesante.3it.usherbrooke.ca:40075/robot/#/login`.
Login using your account, and select the robot you want to control.

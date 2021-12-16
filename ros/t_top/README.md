# t_top
This ROS package contains the configuration files, the launch files and common code files.

## Common Code Descriptions

### `vector_to_angles` function
This function convert a direction vector to angles.

### `MovementCommands` class
This is an helper function to move the head and the torso.

## Nodes
### `opencr_raw_data_interpreter.py`
This node interprets the raw data coming from the OpenCR into useful ROS messages.

#### Parameters
 - `base_link_torso_base_delta_z` (string): The Z offset between the base_link and torso_base frames.

#### Subscribed Topics
 - `opencr/raw_imu` ([std_msgs/Float32MultiArray](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32MultiArray.html)): The OpenCR IMU raw data.
 - `opencr/current_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The current torso orientation.
 - `opencr/current_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)): The current head pose.

#### Published Topics
 - `opencr_imu/data_raw` ([sensor_msgs/Imu](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html)): The OpenCR IMU data.
 - `opencr_imu/mag` ([sensor_msgs/MagneticField](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/MagneticField.html)): The OpenCR Magnetometer data.
 - `opencr/odom` ([nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)): The servo odometry.

#### Published TF Frames
 - `torso_base` to `base_link`
 - `stewart_base` to `head`

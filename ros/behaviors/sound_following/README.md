# sound_following

This folder contains the node to make T-Top follow the loudest sound.

## Nodes

### `sound_following_node.py`

This node makes T-Top follow the loudest sound.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.
- `control_frequency` (double): The frequency at which the pose messages are sent.
- `torso_control_alpha` (double): The low-pass filter parameter for the torso pose.
- `head_control_alpha` (double): The low-pass filter parameter for the head pose.
- `head_enabled` (bool): Indicates if the head will move.
- `min_head_pitch_rad` (double): The minimum pitch angle in radian of the head.
- `max_head_pitch_rad` (double): The maximum pitch angle in radian of the head.
- `min_activity` (double): The minimum activity level to consider the sound source valid.
- `min_valid_source_pitch_rad` (double): The minimum pitch angle in radian to consider the sound source valid.
- `max_valid_source_pitch_rad` (double): The maximum pitch angle in radian to consider the sound source valid.
- `direction_frame_id` (string): The audio analysis frame id.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `sst` ([odas_ros/OdasSstArrayStamped](https://github.com/introlab/odas_ros/blob/main/msg/OdasSstArrayStamped.msg)):
  The sound source tracking information.

#### Published Topics

- `sound_following/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `sound_following/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The
  torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

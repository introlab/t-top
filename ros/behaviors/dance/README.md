# dance

This folder contains the node to make T-Top dance.

## Nodes

### `head_dance_node.py`

This node makes T-Top head dance on the beat. The moves are randomly chosen.

#### Parameters

- `movement_file` (string): The JSON file path containing the movements.

#### Subscribed Topics

- `bpm` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The bpm topic from the
  beat_detector_node.
- `beat` ([std_msgs/Bool](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Bool.html)): The beat topic from the
  beat_detector_node.

#### Published Topics

- `dance/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)):
  The head pose.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.


### `torso_dance_node.py`

This node makes T-Top torso dance on the beat. The moves are randomly chosen.

#### Parameters

- `movement_file` (string): The JSON file path containing the movements.

#### Subscribed Topics

- `bpm` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The bpm topic from the
  beat_detector_node.
- `beat` ([std_msgs/Bool](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Bool.html)): The beat topic from the
  beat_detector_node.
- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status to get torso orientation.

#### Published Topics

- `dance/set_torso_orientation` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The
  torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.


### `led_dance_node.py`

This node makes T-Top torso dance on the beat. The moves are randomly chosen.

#### Parameters

- `led_colors_file` (string): The JSON file path containing the LED patterns.

#### Subscribed Topics

- `beat` ([std_msgs/Bool](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Bool.html)): The beat topic from the
  beat_detector_node.

#### Published Topics

- `dance/set_led_colors` ([daemon_ros_client/LedColors](../../daemon_ros_client/msg/LedColors.msg)):
  The LED colors.

#### Services

- `set_led_colors/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

# gesture

This folder contains the node to make T-Top perform gestures.

## Nodes

### `gesture_node.py`

This node makes T-Top perform gestures.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation. The default value is false.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `gesture/name` ([behavior_msgs/GestureName](../behavior_msgs/msg/GestureName.msg)): The gesture to perform.
    - Supported values : `yes`, `no`, `maybe`, `origin_all`, `origin_head`, `origin_torso`

#### Published Topics

- `gesture/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)):
  The head pose.
- `gesture/set_torso_orientation` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The
  torso orientation.
- `gesture/done` ([behavior_msgs/Done](../behavior_msgs/msg/Done.msg)): Indicates that the gesture is finished.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

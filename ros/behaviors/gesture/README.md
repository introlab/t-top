# gesture

This folder contains the node to make T-Top perform gestures.

## Nodes

### `gesture_node.py`

This node makes T-Top perform gestures.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `gesture/name` ([gesture/GestureName](msg/GestureName.msg)): The gesture to perform.
    - Supported values : `yes`, `no`, `maybe`, `origin_all`, `origin_head`, `origin_torso`

#### Published Topics

- `gesture/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `gesture/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The
  torso orientation.
- `gesture/done` ([gesture/Done](msg/Done.msg)): Indicates that the gesture is finished.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

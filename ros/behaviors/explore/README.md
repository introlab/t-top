# explore

This folder contains the node to make T-Top explore its surrounding.

## Nodes

### `explore_node.py`

This node makes T-Top explore its surrounding.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation. The default value is false.
- `explore_frequency` (double): The frequency at which T-Top explores its surrounding. The default value is 0.00833333333.
- `torso_speed_rad_sec` (double): The torso speed in rad/s. The default value is 0.5.
- `head_speed_rad_sec` (double): The head speed in rad/s. The default value is 0.5.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.

#### Published Topics

- `explore/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)):
  The head pose.
- `explore/set_torso_orientation` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The
  torso orientation.
- `explore/done` ([std_msgs/Empty](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Empty.html)): Indicates that the
  explore moves are finished.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

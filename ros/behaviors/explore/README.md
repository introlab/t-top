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

- `explore/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `explore/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The
  torso orientation.
- `explore/done` ([std_msgs/Empty](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Empty.html)): Indicates that the
  explore moves are finished.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

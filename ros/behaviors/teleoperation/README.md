# teleoperation

This folder contains the node to teleoperate T-Top using twist commands.

## Nodes

### `teleoperation_node.py`

This node allows T-Top to be teleoperated using twist commands.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `teleoperation/cmd_vel` ([geometry_msgs/Twist](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/Twist.html)): The twist commands.

#### Published Topics

- `teleoperation/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)):
  The head pose.
- `teleoperation/set_torso_orientation` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The
  torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.
- `teleop_do_action` ([opentera_webrtc_ros_msgs/SetString](https://github.com/introlab/opentera-webrtc-ros/blob/main/opentera_webrtc_ros_msgs/srv/SetString.srv)): The name of a specific action to do:
    - do_yes
    - do_no
    - do_maybe
    - goto_origin_head
    - goto_origin_torso
    - goto_origin

# teleoperation

This folder contains the node to teleoperate T-Top using twist commands.

## Nodes

### `teleoperation_node.py`

This node allows T-Top to be teleoperated using twist commands.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `teleoperation/cmd_vel` ([geometry_msgs/Twist]): The twist commands.

#### Published Topics

- `teleoperation/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `teleoperation/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The
  torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.
- `teleop_do_action` ([teleoperation/DoAction])(srv/DoAction.srv)): The name of a specific action to do:
    - do_yes
    - do_no
    - do_maybe
    - goto_origin_head
    - goto_origin_torso
    - goto_origin

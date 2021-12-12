# dance
This folder contains the node to make T-Top dance.

## Nodes
### `dance_node.py`
This node makes T-Top dance on the beat. The moves are randomly chosen.

#### Parameters
 - `movement_file` (string): The JSON file path containing the movements.
 - `head_z_zero` (double): The attack value of the sound level filter (default: 0.05).

#### Subscribed Topics
 - `bpm` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The bpm topic from the beat_detector_node.
 - `beat` ([std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html)): The beat topic from the beat_detector_node.

#### Published Topics
 - `opencr/head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)): The head pose.
 - `opencr/torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The torso orientation.

#### Services
 - `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the behavior.

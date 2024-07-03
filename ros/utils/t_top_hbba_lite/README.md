# recorders
This folder contains the HBBA Lite strategies and desires. Also, it contains an arbitration node for the head pose.

## Nodes
### `set_head_pose_arbitration_node`

This node applies priority arbitration to the head pose topic.

#### Parameters

- `topics` (string): The topic names.
- `priorities` (int): The priority for each topic (int, lower means higher priority).
- `timeout_s` (double): The timeout in seconds for each topic.
- `offset_topics` (array of string): The topic names that apply an offset to the head position

#### Published Topics

- `daemon/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)): The output topic.

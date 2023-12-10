# recorders
This folder contains the HBBA Lite strategies and desires. Also, it contains an arbitration node for the head pose.

## Nodes
### `set_head_pose_arbitration_node`

This node applies priority arbitration to the head pose topic.

#### Parameters

- `topics`: The topic descriptions containing the topic name (string), priority (int, lower means higher priority) and timeout_s value (double).
- `offset_topics` (array of string): The topic names that apply an offset to the head position
- `latch` (bool): Indicates if the output topic is latched.

#### Published Topics

- `daemon/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)): The output topic.

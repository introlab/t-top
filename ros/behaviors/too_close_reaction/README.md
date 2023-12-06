# too_close_reaction

This folder contains the node to move back the head when someone is too close to T-Top.

## Nodes

### `too_close_reaction_node.py`

This node makes T-Top move back its head when someone is too close to T-Top.

#### Parameters

- `max_offset_m` (double): The maximum distance in meter to move back the head.
- `too_close_start_distance_m` (double): The start distance to move back the head.
- `too_close_end_distance_m` (double): The end distance to move back the head.
- `pixel_ratio` (double): The pixel ratio to measure de distance.

#### Subscribed Topics

- `depth_image_raw` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The depth image.

#### Published Topics

- `too_close_reaction/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose offset.

#### Services

- `depth_image_raw/filter_state` ([hbba_lite/SetOnOffFilterState](../../utils/hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

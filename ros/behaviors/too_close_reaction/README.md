# too_close_reaction

This folder contains the node to move back the head when someone is too close to T-Top.

## Nodes

### `too_close_reaction_node.py`

This node makes T-Top move back its head when someone is too close to T-Top.

#### Parameters

- `max_offset_m` (double): The maximum distance in meter to move back the head. The default value is 0.01.
- `too_close_start_distance_m` (double): The start distance to move back the head. The default value is 0.5.
- `too_close_end_distance_m` (double): The end distance to move back the head. The default value is 0.25.
- `pixel_ratio` (double): The pixel ratio to measure de distance. The default value is 0.01.

#### Subscribed Topics

- `depth_image_raw` ([sensor_msgs/Image](https://docs.ros.org/en/humble/p/sensor_msgs/interfaces/msg/Image.html)): The depth image.

#### Published Topics

- `too_close_reaction/set_head_pose` ([geometry_msgs/PoseStamped](https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/PoseStamped.html)):
  The head pose offset.

#### Services

- `depth_image_raw/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

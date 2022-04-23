# face_following

This folder contains the node to make T-Top follow the nearest face.

## Nodes

### `face_following_node.py`

This node makes T-Top follow the nearest face.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.
- `control_frequency` (double): The frequency at which the pose messages are sent.
- `torso_control_alpha` (double): The low-pass filter parameter for the torso pose.
- `head_control_p_gain` (double): The controller proportional gain for the head pose.
- `nose_confidence_threshold` (double): The confidence threshold for the nose keypoint.
- `head_enabled` (bool): Indicates if the head will move.
- `min_head_pitch_rad` (double): The minimum pitch angle in radian of the head.
- `max_head_pitch_rad` (double): The maximum pitch angle in radian of the head.

#### Subscribed Topics

- `video_analysis` ([video_analyzer/VideoAnalysis](../../perceptions/video_analyzer/msg/VideoAnalysis.msg)): The video
  analysis containing the detected objects.

#### Published Topics

- `opencr/head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `opencr/torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The
  torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

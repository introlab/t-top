# face_following

This folder contains the node to make T-Top follow the nearest face.

## Nodes

### `nearest_face_following_node.py`

This node makes T-Top follow the nearest face.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation. The default value is false.
- `control_frequency` (double): The frequency at which the pose messages are sent. The default value is 30.
- `torso_control_alpha` (double): The low-pass filter parameter for the torso pose. The default value is 0.2.
- `head_control_p_gain` (double): The controller proportional gain for the head pose. The default value is 0.175.
- `head_enabled` (bool): Indicates if the head will move. The default value is true.
- `min_head_pitch_rad` (double): The minimum pitch angle in radian of the head. The default value is -0.35.
- `max_head_pitch_rad` (double): The maximum pitch angle in radian of the head. The default value is 0.35.
- `nose_confidence_threshold` (double): The confidence threshold for the nose keypoint. The default value is 0.4.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `video_analysis` ([video_analyzer/VideoAnalysis](../../perceptions/video_analyzer/msg/VideoAnalysis.msg)): The video
  analysis containing the detected objects. The video analysis must contain 3d positions.

#### Published Topics

- `nearest_face_following/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `nearest_face_following/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)):
  The torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.


### `specific_face_following_node.py`

This node makes T-Top follow the face of the specified person.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation. The default value is false.
- `control_frequency` (double): The frequency at which the pose messages are sent. The default value is 30.
- `torso_control_alpha` (double): The low-pass filter parameter for the torso pose. The default value is 0.2.
- `head_control_p_gain` (double): The controller proportional gain for the head pose. The default value is 0.175.
- `head_enabled` (bool): Indicates if the head will move. The default value is true.
- `min_head_pitch_rad` (double): The minimum pitch angle in radian of the head. The default value is -0.35.
- `max_head_pitch_rad` (double): The maximum pitch angle in radian of the head. The default value is 0.35.
- `direction_frame_id` (string): The audio analysis frame id. The default value is odas.

#### Subscribed Topics

- `person_names` ([person_identification/PersonNames](../../perceptions/person_identification/msg/PersonNames.msg)):
  The person names.
- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.

#### Published Topics

- `specific_face_following/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `specific_face_following/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)):
  The torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

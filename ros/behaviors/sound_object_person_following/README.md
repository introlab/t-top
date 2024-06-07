# sound_object_person_following

This folder contains the node to make T-Top follow the loudest sound, the biggest (nearest) person and some objects.

## Nodes

### `calibrate_sound_object_person_following_node.py`

This node matches the 3d camera with the 2d wide camera.

#### Parameters

- `match_count` (bool): Number of match to use. The default value is 100.

#### Subscribed Topics

- `camera_3d/color/image_raw` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The rectified color image of the 3d camera.
- `camera_2d_wide/image_rect` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The rectified color image of the 2d wide camera.

### `sound_object_person_following_node.py`

This node makes T-Top follow the loudest sound, the biggest (nearest) person and some objects.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation. The default value is false.
- `control_frequency` (double): The frequency at which the pose messages are sent. The default value is 30.
- `torso_control_alpha` (double): The low-pass filter parameter for the torso pose (sound following). The default value is 0.2.
- `torso_control_p_gain` (double): The controller proportional gain for the torso pose. The default value is 0.45.
- `head_control_p_gain` (double): The controller proportional gain for the head pose. The default value is 0.45.
- `min_head_pitch_rad` (double): The minimum pitch angle in radian of the head. The default value is -0.35.
- `max_head_pitch_rad` (double): The maximum pitch angle in radian of the head. The default value is 0.35.
- `object_person_follower_type` (string): The object person follower type (bounding_box or semantic_segmentation). The default value is bounding_box.

- `min_sst_activity` (double): The minimum activity level to consider the sound source valid.
- `min_valid_sst_pitch` (double): The minimum pitch angle in radian to consider the sound source valid.
- `max_valid_sst_pitch` (double): The maximum pitch angle in radian to consider the sound source valid.
- `direction_frame_id` (string): The audio analysis frame id.

- `object_classes` (list of strings): The followed object classes.
- `padding` (double): The padding of the person and the objects.
- `target_lambda` (double): The loss scale factor centering the person.

#### Subscribed Topics

- `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The motor status.
- `sst` ([odas_ros_msgs/OdasSstArrayStamped](https://github.com/introlab/odas_ros/blob/main/odas_ros_msgs/msg/OdasSstArrayStamped.msg)):
  The sound source tracking information.
- `video_analysis` ([perception_msgs/VideoAnalysis](../../perceptions/perception_msgs/msg/VideoAnalysis.msg)): The video
  analysis containing the detected objects of the 2d wide camera.

#### Published Topics

- `sound_object_person_following/set_head_pose` ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)):
  The head pose.
- `sound_object_person_following/set_torso_orientation` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)):
  The torso orientation.

#### Services

- `pose/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter
  state service to enable or disable the behavior.

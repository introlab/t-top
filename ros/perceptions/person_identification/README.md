# person_identification
This folder contains the node to perform person identification.

## Nodes
### `capture_face_node.py`
This node adds to `people.json` an averaged face descriptor for the person in front of the robot.

Use the following command to launch the nodes.
```bash
ros2 launch person_identification capture_face.launch name:=<person_name> neural_network_inference_type:=<cpu, torch_gpu or trt_gpu>
```

#### Parameters
 - `name` (string): The person name.
 - `mean_size` (int): How many descriptor to average. The default value is 10.
 - `face_sharpness_score_threshold` (double): The threshold to consider the face sharp enough. The default value is 0.5.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](../video_analyzer/msg/VideoAnalysis.msg)): The video analysis containing the detected objects.


### `capture_voice_node.py`
This node adds to `people.json` an averaged voice descriptor for the person talking near the robot.

Use the following command to launch the nodes.
```bash
roslaunch person_identification capture_voice.launch name:=<person_name> neural_network_inference_type:=<cpu, torch_gpu or trt_gpu>
```

#### Parameters
 - `name` (string): The person name.
 - `mean_size` (int): How many descriptor to average. The default value is 10.

#### Subscribed Topics
 - `audio_analysis` ([audio_analyzer/AudioAnalysis](../audio_analyzer/msg/AudioAnalysis.msg)): The audio analysis containing the audio classes, general audio embedding, voice embedding and the sound direction.


### `person_identification_node.py`
This node performs person identification. The people must be already added to `people.json` with the previous nodes.

#### Parameters
 - `face_sharpness_score_threshold` (double): The threshold to consider the face sharp enough. The default value is 0.5.
 - `face_descriptor_threshold` (double): The maximum distance between two face descriptors to be considered the same person. The default value is 0.7.
 - `voice_descriptor_threshold` (double): The maximum distance between two voice descriptors to be considered the same person. The default value is 1.266.
 - `face_voice_descriptor_threshold` (double): The maximum distance between two merged descriptors to be considered the same person. The default value is 1.5092.
 - `nose_confidence_threshold` (double): The confidence threshold for the nose keypoint. The default value is 0.4.
 - `direction_frame_id` (string): The audio analysis frame id. The default value is odas.
 - `direction_angle_threshold_rad` (double): The maximum angle between the face and voice directions to be considered the same person. The default value is 0.15.
 - `ignore_direction_z` (bool): Indicates if the angle between between the face and voice directions ignores the z-axis. The default value is true.
 - `search_frequency` (double): The frequency at which the search occurs. The default value is 2.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](../video_analyzer/msg/VideoAnalysis.msg)): The video analysis containing the detected objects.
 - `audio_analysis` ([audio_analyzer/AudioAnalysis](../audio_analyzer/msg/AudioAnalysis.msg)): The audio analysis containing the audio classes, general audio embedding, voice embedding and the sound direction.

#### Published Topics
 - `person_names` ([person_identification/PersonNames](msg/PersonNames.msg)): The person names.

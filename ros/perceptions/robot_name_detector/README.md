# robot_name_detector
This folder contains the node to detect the robot name.

## Nodes
### `robot_name_detector_node.py`
This node measures the sound level and detects the robot name.

#### Parameters
 - `message_rate` (int): The frequency of `sound_rms` and `sound_presence` messages.
 - `sound_rms_attack` (double): The attack value of the sound level filter (default: 0.05).
 - `sound_rms_release` (double): The release value of the sound level filter (default: 0.99).
 - `sound_rms_presence_threshold` (double): The threshold of the sound precense (default: 0.05).
 - `inference_type` (string): Indicates where to run the neural network (cpu, torch_gpu or trt_gpu).
 - `robot_name_model_probability_threshold` (double): The joint probability threshold.
 - `robot_name_model_interval` (int): The interval between inferences in sample.
 - `robot_name_model_analysis_delay` (int): The delay after the rising edge in sample.
 - `robot_name_model_analysis_count` (int): The number of inference to perform.

#### Subscribed Topics
 - `audio_in` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The sound topic processed, which must be signed_16, at 16000 Hz and 1 channel.

#### Published Topics
 - `sound_rms` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): The sound level.
 - `sound_presence` ([std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html)): Indicates if sound is presence.
 - `robot_name_detected` ([std_msgs/Empty](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Empty.html)): Indicates that the robot name is detected.
 - `robot_name_detector/set_led_colors` ([daemon_ros_client/LedColors](../../daemon_ros_client/msg/LedColors.msg)): The LED colors for the status.

#### Services
 - `audio_in/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the processing.

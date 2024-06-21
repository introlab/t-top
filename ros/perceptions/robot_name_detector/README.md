# robot_name_detector
This folder contains the node to detect the robot name.

## Nodes
### `robot_name_detector_node.py`
This node measures the sound level and detects the robot name.

#### Parameters
 - `led_status_duration_s` (double): The LED status duration en seconds. The default value is 1.0.
 - `sound_presence_relative_threshold` (double): The relative threshold of the sound precense between the slow and fast filter. The default value is 1.05.
 - `robot_name_model_probability_threshold` (double): The joint probability threshold. The default value is 0.05.
 - `robot_name_model_interval` (int): The interval between inferences in sample. The default value is 800.
 - `robot_name_model_analysis_delay` (int): The delay after the rising edge in sample. The default value is 10400.
 - `robot_name_model_analysis_count` (int): The number of inference to perform. The default value is 3.
 - `fast_sound_rms_attack` (double): The fast attack value of the sound level filter. The default value is 0.05.
 - `fast_sound_rms_release` (double): The fast release value of the sound level filter The default value is 0.9.
 - `slow_sound_rms_attack` (double): The slow attack value of the sound level filter. The default value is 0.975.
 - `slow_sound_rms_release` (double): The slow release value of the sound level filter The default value is 0.975.
 - `inference_type` (string): Indicates where to run the neural network (cpu, torch_gpu or trt_gpu). The default value is cpu.

#### Subscribed Topics
 - `audio_in` ([audio_utils_msgs/AudioFrame](https://github.com/introlab/audio_utils/blob/main/audio_utils_msgs/msg/AudioFrame.msg)): The sound topic processed, which must be signed_16, at 16000 Hz and 1 channel.

#### Published Topics
 - `sound_rms` ([std_msgs/Float32](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Float32.html)): The sound level.
 - `sound_presence` ([std_msgs/Bool](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Bool.html)): Indicates if sound is presence.
 - `robot_name_detected` ([std_msgs/Empty](https://docs.ros.org/en/humble/p/std_msgs/interfaces/msg/Empty.html)): Indicates that the robot name is detected.
 - `robot_name_detector/set_led_colors` ([daemon_ros_client/LedColors](../../daemon_ros_client/msg/LedColors.msg)): The LED colors for the status.

#### Services
 - `audio_in/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the processing.

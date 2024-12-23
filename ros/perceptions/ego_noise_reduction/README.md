# ego_noise_reduction
This folder contains the node to perform ego noise reduction.

## Nodes
### `data_gathering.py`
This node gathers the data needed to perform ego noise reduction. To use this node, the OpenCR must be flashed with `ego_noise_opencr_firmware`.

Use the following command to launch the nodes.
```bash
ros2 launch ego_noise_reduction data_gathering.launch
```

#### Parameters
 - `n_fft` (int): The FFT frame size. The default value is `1024`.
 - `sampling_frequency` (int): The sampling frequency of the audio. The default value is `16000`.
 - `channel_count` (int): The channel count of the audio. The default value is `16`.
 - `audio_queue_size` (int): The audio queue size. The default value is `1`.

### `test.py`
This node tests the ego noise reduction.

Use the following command to launch the nodes.
```bash
roslaunch ego_noise_reduction test.launch output_directory:=<output_directory>
```

### `ego_noise_reduction_node`
This node performs ego noise reduction.

#### Parameters
 - `type` (string): The noise reduction algorithm to use (`spectral_subtraction` or `log_mmse`). The default value is `log_mmse`.
 - `format` (string): The audio format (see [audio_utils_msgs/AudioFrame](../../utils/audio_utils/audio_utils_msgs/msg/AudioFrame.msg)). The default value is `signed_32`.
 - `channel_count` (int): The channel count of the audio. The default value is `16`.
 - `sampling_frequency` (int): The sampling frequency of the audio. The default value is `16000`.
 - `frame_sample_count` (int): The number of samples in each frame. The default value is `1024`.
 - `n_fft` (int): The FFT frame size. It must be a multiple of `frame_sample_count`.
 - `noise_directory` (string): The directory path containing the noise data.
 - `noise_estimator_epsilon` (double): The sensitivity of the noise estimation algorithm. The default value is `4.0`.
 - `noise_estimator_alpha` (double): The recursive average of the noise magnitude. The default value is `0.9`.
 - `noise_estimator_delta` (double): The recursive average of the noise magnitude variance. The default value is `0.9`.

 - `spectral_subtraction_alpha0` (double): This parameter is useful when the type `spectral_subtraction`. The default value is `5.0`.
 - `spectral_subtraction_gamma` (double): This parameter is useful when the type `spectral_subtraction`. The default value is `0.1`.
 - `spectral_subtraction_beta` (double): This parameter is useful when the type `spectral_subtraction`. The default value is `0.01`.

 - `log_mmse_alpha` (double): This parameter is useful when the type `log_mmse`. The default value is `0.9`.
 - `log_mmse_max_a_posteriori_snr` (double): This parameter is useful when the type `log_mmse`. The default value is `0.9`.
 - `log_mmse_min_a_priori_snr` (double): This parameter is useful when the type `log_mmse`. The default value is `0.9`.

#### Subscribed Topics
 - `daemon/motor_status` ([daemon_ros_client/MotorStatus](../../daemon_ros_client/msg/MotorStatus.msg)): The current motor status.
 - `audio_in` ([audio_utils_msgs/AudioFrame](https://github.com/introlab/audio_utils/blob/ros2/audio_utils_msgs/msg/AudioFrame.msg)): The sound topic to process.

#### Published Topics
- `audio_out` ([audio_utils_msgs/AudioFrame](https://github.com/introlab/audio_utils/blob/ros2/audio_utils_msgs/msg/AudioFrame.msg)): The processed sound topic.

## Acknowledgments

Thanks to Admiral Bob for the song [Choice](http://dig.ccmixter.org/files/admiralbob77/61638) and [LibriSpeech](https://www.openslr.org/12) for the speech.
The tests use this song and this speech.

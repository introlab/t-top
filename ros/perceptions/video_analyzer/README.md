# video_analyzer
This folder contains visualization node and the node to analyze the video.

## Nodes
### `video_analyzer.py`
This node detects objects. If the audio contains a person, it estimates the pose of the person and extracts an embedding for the face.

#### Parameters
 - `use_descriptor_yolo_v4` (bool): Indicates to use the network extracting a embedding or not.
 - `confidence_threshold` (double): The object confidence threshold.
 - `nms_threshold` (double): The Non-Maximum Suppresion threshold.
 - `person_probability_threshold` (double): The person confidence threshold.
 - `pose_confidence_threshold` (double): The pose key point confidence threshold.
 - `inference_type` (string): Indicates where to run the neural network (cpu, torch_gpu or trt_gpu).
 - `depth_mean_offset` (int): The rectangle offset when calulating the depth of the object.

#### Subscribed Topics
 - `color_image_raw` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The color image.
 - `depth_image_raw` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The depth image.
 - `depth_camera_info` ([sensor_msgs/CameraInfo](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html)): The depth camera info used to preject 2D point in 3D.

#### Published Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](msg/VideoAnalysis.msg)): The video analysis containing the detected objects.
 - `analysed_image` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): A rendering of the video analysis.

#### Services
 - `image_raw/filter_state` ([hbba_lite/SetThrottlingFilterState](../../hbba_lite/srv/SetThrottlingFilterState.srv)) The HBBA filter state service to enable, disable or throttle the processing.
 - `analysed_image/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)) The HBBA filter state service to enable or disable the topic `analysed_image`.

### `video_analysis_visualizer_node.py`
This node create a mosaic containing the objects in the video analysis.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](msg/VideoAnalysis.msg)): The video analysis containing the detected objects.

#### Published Topics
 - `video_analysis_mosaic` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The mosaic containing the objects in the video analysis.

### `video_analysis_markers_node.py`
This node create a marker array for the objects in the video analysis.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](msg/VideoAnalysis.msg)): The video analysis containing the detected objects.

#### Published Topics
 - `video_analysis_markers` ([visualization_msgs/MarkerArray](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html)): The marker array for the objects in the video analysis.

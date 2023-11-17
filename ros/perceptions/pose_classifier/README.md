# pose_classifier
This folder contains the node to perform pose classification from the pose extracted by the [video_analyzer](../video_analyzer).

## Nodes
### `pose_classifier_node.py`
This node performs pose classification from the pose extracted by the [video_analyzer](../video_analyzer).

#### Parameters
 - `pose_confidence_threshold` (double): The pose confidence threshold to determine if a joint is present.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](../video_analyzer/msg/VideoAnalysis.msg)): The video analysis topic.

#### Published Topics
- `pose_classification` ([pose_classifier/PoseClassifications](msg/PoseClassifications.msg)): The pose classification topic.

## Pose classes
- **head_vertical_class**
    - `down`
    - `straight`
    - `up`
- **head_horizontal_class**
    - `left`
    - `straight`
    - `right`
- **left_arm_class** or **right_arm_class**
    - `down_straight`: arm extended downward
    - `down_right_angle_external`: arm down, bent 90 degrees outward
    - `down_right_angle_internal`: arm down, bent 90 degrees inward
    - `horizontal_straight`: arm extended horizontally
    - `horizontal_right_angle_up`: arm horizontally, bent 90 degrees upwards
    - `horizontal_right_angle_down`: arm horizontally, bent 90 degrees downward
    - `up`: arm up

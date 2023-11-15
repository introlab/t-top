geometry_msgs/Point center_2d # Normalized
geometry_msgs/Point center_3d

float32 width_2d # Normalized
float32 height_2d # Normalized

string object_class
float32 object_confidence
float32 object_class_probability
float32[] object_descriptor
sensor_msgs/Image object_image

geometry_msgs/Point[] person_pose_2d # Normalized
geometry_msgs/Point[] person_pose_3d
float32[] person_pose_confidence
sensor_msgs/Image person_pose_image

float32[] face_descriptor
int32 face_alignment_keypoint_count
float32 face_sharpness_score
sensor_msgs/Image face_image

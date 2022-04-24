geometry_msgs/Point center_2d # Normalized
geometry_msgs/Point center_3d

string object_class
float32 object_confidence
float32[] object_descriptor
sensor_msgs/Image object_image

geometry_msgs/Point[] person_pose_2d # Normalized
geometry_msgs/Point[] person_pose_3d
float32[] person_pose_confidence
sensor_msgs/Image person_pose_image

float32[] face_descriptor
sensor_msgs/Image face_image

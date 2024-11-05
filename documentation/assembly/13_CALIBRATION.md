# Calibration

## A. 2D Wide Camera

1. Execute the following command.
```bash
ros2 launch t_top platform.launch.xml camera_2d_wide_enabled:=true
```
2. Follow the following [steps](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration).
```bash
# Create calibration path
mkdir -p ~/.ros/t-top/calibration
ros2 run camera_calibration cameracalibrator --size 9x6 --square 0.051 --ros-args -r image:=/camera_2d_wide_full_hd/image -r camera/set_camera_info:=/camera_2d_wide_full_hd/set_camera_info
```

## B. sound_object_person_following

1. Place an object having visual features in front of the robot.
2. Execute the following command.
```bash
ros2 launch sound_object_person_following calibrate_sound_object_person_following.launch.xml
```

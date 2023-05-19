# Calibration

## A. 2D Wide Camera

1. Execute the following command.
```bash
roslaunch t_top platform.launch camera_2d_wide_enabled:=true
```
2. Follow the following [steps](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration).

## B. sound_object_person_following

1. Place an object having visual features in front of the robot.
2. Execute the following command.
```bash
roslaunch sound_object_person_following calibrate_sound_object_person_following.launch
```

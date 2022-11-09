# Service ROS Protocol
This document presents the communication protocol between the systemmd service and ROS. It is a JSON protocol encoded with UTF-8.

## General Format

```json
{
    "type": "name",
    "payload": {

    }
}
```

### Message Types
| Message Type            | Name                                                 | Source  | Destination | Description                                                                           |
| ----------------------- | ---------------------------------------------------- | ------- | ----------- | -------------------------------------------------------------- |
| `base_status`           | [Base Status](#base-Status-payload)                     | Service | ROS         |
| `base_button_pressed`   | [Button Pressed](#button-pressed-payload)               | Service | ROS         |
| `set_volume`            | [Set Volume](#set-volume-payload)                       | ROS     | Service     |
| `set_led_colors`        | [Set LED Colors](#set-led-colors-payload)               | ROS     | Service     |                |
| `motor_status`          | [Motor Status](#motor-status-payload)                   | Service | ROS         |
| `imu_data`              | [IMU Data](#imu-data-payload)                           | Service | ROS         |                        |
| `set_torso_orientation` | [Set Torso Orientation](#set-torso-orientation-payload) | ROS     | Service     |                            |
| `set_head_pose`         | [Set Head Pose](#set-head-pose-payload)                 | ROS     | Service     |                             |


## Base Status Payload

```json
{
    "is_psu_connected": true, // This field indicates whether the PSU is connected to the robot.
    "has_charger_error": false, // This field indicates whether there is a charger error.
    "is_battery_charging": true, // This field indicates whether the battery is charging.
    "has_battery_error": false, // This field indicates whether there is a battery error.
    "state_of_charge": 75.2, // This field contains the actual battery state of charge in percent (0 to 100).
    "current": 1.78, // This field contains the actual current in A.
    "voltage": 19.2, // This field contains the actual voltage in V.
    "onboard_temperature": 30.1, // This field contains the actual temperature in °C on the PSU Control PCB.
    "external_temperature": 26.5, // This field contains the actual temperature in °C in the base (invalid).
    "front_light_sensor": 0.75, // This field contains the front light level (0 to 1).
    "back_light_sensor": 0.65, // This field contains the back light level (0 to 1).
    "left_light_sensor": 0.55, // This field contains the left light level (0 to 1).
    "right_light_sensor": 0.45, // This field contains the right light level (0 to 1).
    "volume": 24, // This field contains the actual volume (0 to 63).
    "maximum_volume": 63 // This field contains the actual maximum volume (0 to 63). The maximum depends on whether the power supply is connected.
}
```

## Button Pressed Payload
```json
{
    "button": "start" // This field indicates which button was pressed (start or stop).
}
```

## Set Volume Payload
```json
{
    "volume": 32 // This field contains the new volume (0 to 63).
}
```

## Set Led Colors Payload
```json
{
    [
        // This object indicates the first LED color.
        {
            "red": 125,
            "green": 200,
            "blue": 150
        },
        ...
    ]
}
```

## Motor Status Payload
```json
{
    "torso_orientation": 0.1, // This field contains the actual torso orientation in radian (0 to 2PI).
    "torso_servo_speed": 90, // This field contains the actual torso servo speed in 0.229 rpm (-1023 to 1023).
    "head_servo_angle_1": 0.1, // This field contains the actual head servo angle 1 in radian (0 to 2PI).
    "head_servo_angle_2": 0.1, // This field contains the actual head servo angle 2 in radian (0 to 2PI).
    "head_servo_angle_3": 0.1, // This field contains the actual head servo angle 3 in radian (0 to 2PI).
    "head_servo_angle_4": 0.1, // This field contains the actual head servo angle 4 in radian (0 to 2PI).
    "head_servo_angle_5": 0.1, // This field contains the actual head servo angle 5 in radian (0 to 2PI).
    "head_servo_angle_6": 0.1, // This field contains the actual head servo angle 6 in radian (0 to 2PI).
    "head_servo_speed_1": 80, // This field contains the actual head servo speed 1 in 0.229 rpm (-1023 to 1023).
    "head_servo_speed_2": 81, // This field contains the actual head servo speed 2 in 0.229 rpm (-1023 to 1023).
    "head_servo_speed_3": 82, // This field contains the actual head servo speed 3 in 0.229 rpm (-1023 to 1023).
    "head_servo_speed_4": 83, // This field contains the actual head servo speed 4 in 0.229 rpm (-1023 to 1023).
    "head_servo_speed_5": 84, // This field contains the actual head servo speed 5 in 0.229 rpm (-1023 to 1023).
    "head_servo_speed_6": 87, // This field contains the actual head servo speed 6 in 0.229 rpm (-1023 to 1023).
    "head_pose_position_x": 0.01, // This field contains the actual X head position in meters in frame stewart_base.
    "head_pose_position_x": 0.02, // This field contains the actual Y head position in meters in frame stewart_base.
    "head_pose_position_x": 0.17, // This field contains the actual Z head position in meters in frame stewart_base.
    "head_pose_orientation_w": 0.989, // This field contains the actual W head orientation in meters in frame stewart_base.
    "head_pose_orientation_x": 0.001, // This field contains the actual X head orientation in meters in frame stewart_base.
    "head_pose_orientation_y": 0.002, // This field contains the actual Y head orientation in meters in frame stewart_base.
    "head_pose_orientation_z": 0.003, // This field contains the actual Z head orientation in meters in frame stewart_base.
    "is_head_pose_reachable": true // This field indicates if the last desired head pose is reachable.
}
```

## IMU Data Payload
```json
{
    "acceleration_x": 0.01, // This field contains the actual IMU X Acceleration in m/s².
    "acceleration_y": 0.02, // This field contains the actual IMU Y Acceleration in m/s².
    "acceleration_z": 0.03, // This field contains the actual IMU Z Acceleration in m/s².
    "angular_rate_x": 0.04, // This field contains the actual IMU X Angular Rate in rad/s.
    "angular_rate_y": 0.05, // This field contains the actual IMU X Angular Rate in rad/s.
    "angular_rate_z": 0.06, // This field contains the actual IMU X Angular Rate in rad/s.
}
```

## Set Torso Orientation Payload
```json
{
    "torso_orientation": 0.2 // This field contains the new torso orientation in radian (0 to 2PI).
}
```

## Set Head Pose Payload
```json
{
    "head_pose_position_x":, // This field contains the new X head position in meters in frame stewart_base.
    "head_pose_position_y":, // This field contains the new Y head position in meters in frame stewart_base.
    "head_pose_position_z":, // This field contains the new Z head position in meters in frame stewart_base.
    "head_pose_orientation_w":, // This field contains the new W head orientation in meters in frame stewart_base.
    "head_pose_orientation_x":, // This field contains the new X head orientation in meters in frame stewart_base.
    "head_pose_orientation_y":, // This field contains the new Y head orientation in meters in frame stewart_base.
    "head_pose_orientation_z":, // This field contains the new Z head orientation in meters in frame stewart_base.
}
```

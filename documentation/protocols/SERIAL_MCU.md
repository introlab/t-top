# Serial MCU Communucation Protocol


## General Format
All fields use big-endian ordering.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="9">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
            <th>...</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Full Message Length</td>
            <td>Message Source Id</td>
            <td>Message Destination Id</td>
            <td>Acknowledgment Needed</td>
            <td colspan="2">Message Id</td>
            <td colspan="2">Message Type</td>
            <td>Payload</td>
        </tr>
        <tr>
            <td>CRC8</td>
            <td colspan="8"></td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name             | Type   | Description                                                              |
| ---------------------- | ------ | ------------------------------------------------------------------------ |
| Full Message Length    | uint8  | This field is equal to the message length in bytes including this field. |
| Message Source Id      | uint8  | This field contains the device id of the source device.                  |
| Message Destination Id | uint8  | This field contains the device id of the destination device.             |
| Acknowledgment Needed  | bool   | This field indicates if an acknowledgment is needed.                     |
| Message Id             | uint16 | This field contains a random id identifying the message.                 |
| Message Type           | uint16 | This field contains the message type.                                    |
| Payload                |        | The payload depending on the message type.                               |
| CRC8                   | uint8  | This field contains the CRC8 value excluding this field.                 |

### Device Id Descriptions
| Device Id | Name              |
| --------- | ----------------- |
| 0         | psu_control       |
| 1         | dynamixel_control |
| 2         | computer          |

### Message Types
| Message Type | Name                                                    | Source            | Destination       | Description                                          |
| ------------ | ------------------------------------------------------- | ----------------- | ----------------- | ---------------------------------------------------- |
| 1            | [Base Status](#base-Status-payload)                     | psu_control       | computer          | This message contains the status of the base.        |
| 2            | [Button Pressed](#button-pressed-payload)               | psu_control       | computer          | This message indicates that a button is pressed.     |
| 3            | [Set Volume](#set-volume-payload)                       | computer          | psu_control       | This message sets the volume of the audio amplifier. |
| 4            | [Set LED Colors](#set-led-colors-payload)               | computer          | psu_control       | This message sets the LED colors of the base.        |
| 5            | [Motor Status](#motor-status-payload)                   | dynamixel_control | computer          | This message contains the status of all motors.      |
| 6            | [IMU Data](#imu-data-payload)                           | dynamixel_control | computer          | This message contains the IMU data.                  |
| 7            | [Set Torso Orientation](#set-torso-orientation-payload) | computer          | dynamixel_control | This message sets the torso orientation.             |
| 8            | [Set Head Pose](#set-head-pose-payload)                 | computer          | dynamixel_control | This message sets the head pose.                     |

### Behaviors
- All devices route the message according to the destination id.
- The message is dropped if the destination id is invalid.
- The message is dropped if the source id is invalid.
- The message is dropped if the CRC8 invalid.
- The message is dropped if the length does not match with the message type.
- The destination device sends an acknowledgment to the source device if the CRC8 is valid.


## Base Status Payload
This message contains the status of the base.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Is PSU Connected</td>
            <td>Has Charger Error</td>
            <td>Is Battery Charging</td>
            <td>Has Battery Error</td>
            <td colspan="4">State of Charge</td>
        </tr>
        <tr>
            <td colspan="4">Current</td>
            <td colspan="4">Voltage</td>
        </tr>
        <tr>
            <td colspan="4">Onboard Temperature</td>
            <td colspan="4">External Temperature</td>
        </tr>
        <tr>
            <td colspan="4">Front Light Sensor</td>
            <td colspan="4">Back Light Sensor</td>
        </tr>
        <tr>
            <td colspan="4">Left Light Sensor</td>
            <td colspan="4">Right Light Sensor</td>
        </tr>
        <tr>
            <td>Volume</td>
            <td colspan="7"></td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name             | Type   | Description                                                              |
| ---------------------- | ------ | ------------------------------------------------------------------------ |
| Is PSU Connected       | bool   | This field indicates if the PSU is connected to the robot.               |
| Has Charger Error      | bool   | This field indicates if there is a charger error.                        |
| Is Battery Charging    | bool   | This field indicates the battery is charging.                            |
| Has Battery Error      | bool   | This field indicates if there is a battery error.                        |
| State of Charge        | float  | This field contains the actual battery state of charge (0 to 100).       |
| Current                | float  | This field contains the actual current in A.                             |
| Voltage                | float  | This field contains the actual voltage in V.                             |
| Onboard Temperature    | float  | This field contains the actual temperature in °C on the psu_control PCB. |
| External Temperature   | float  | This field contains the actual temperature in °C in the base (invalid).  |
| Front Light Sensor     | float  | This field contains the front light level (0 to 1).                      |
| Back Light Sensor      | float  | This field contains the back light level (0 to 1).                       |
| Left Light Sensor      | float  | This field contains the left light level (0 to 1).                       |
| Right Light Sensor     | float  | This field contains the right light level (0 to 1).                      |
| Volume                 | uint8  | This field contains the actual volume (0 to 63).                         |


## Button Pressed Payload
This message indicates that a button is pressed.

### Table View
<table>
    <thead>
        <tr>
            <th>Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Button Id</td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name | Type  | Description                                        |
| ---------- | ----- | -------------------------------------------------- |
| Button Id  | uint8 | This field contains the button id that is pressed. |

### Device Id Descriptions
| Button Id | Name  |
| --------- | ----- |
| 0         | Start |
| 1         | End   |


## Set Volume Payload
This message sets the volume of the audio amplifier.

### Table View
<table>
    <thead>
        <tr>
            <th>Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Volume</td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name | Type  | Description                                   |
| ---------- | ----- | --------------------------------------------- |
| Volume     | uint8 | This field contains the new volume (0 to 63). |


## Set LED Colors Payload
This message sets the volume of the audio amplifier.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>LED 1 Red Value</td>
            <td>LED 1 Green Value</td>
            <td>LED 1 Blue Value</td>
            <td>LED 2 Red Value</td>
            <td>LED 2 Green Value</td>
            <td>LED 2 Blue Value</td>
            <td colspan="2">...</td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name        | Type  | Description                                               |
| ----------------- | ----- | --------------------------------------------------------- |
| LED 1 Red Value   | uint8 | This field contains the LED 1 red value (0 to 255).       |
| LED 1 Green Value | uint8 | This field contains the LED 1 red value (0 to 255).       |
| LED 1 Blue Value  | uint8 | This field contains the LED 1 red value (0 to 255).       |
| LED 2 Red Value   | uint8 | This field contains the LED 2 red value (0 to 255).       |
| LED 2 Green Value | uint8 | This field contains the LED 2 red value (0 to 255).       |
| LED 2 Blue Value  | uint8 | This field contains the LED 2 red value (0 to 255).       |
| ...               |       | Those fields are repeated for each leds. There is 31 LEDs |


## Motor Status Payload
This message contains the status of all motors.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="4">Torso Orientation</td>
            <td colspan="2">Torso Servo Speed</td>
            <td colspan="2">Head Servo Angle 1</td>
        </tr>
        <tr>
            <td colspan="2">Head Servo Angle 1</td>
            <td colspan="4">Head Servo Angle 2</td>
            <td colspan="2">Head Servo Angle 3</td>
        </tr>
        <tr>
            <td colspan="2">Head Servo Angle 3</td>
            <td colspan="4">Head Servo Angle 4</td>
            <td colspan="2">Head Servo Angle 5</td>
        </tr>
        <tr>
            <td colspan="2">Head Servo Angle 5</td>
            <td colspan="4">Head Servo Angle 6</td>
            <td colspan="2">Head Servo Speed 1</td>
        </tr>
        <tr>
            <td colspan="2">Head Servo Speed 2</td>
            <td colspan="2">Head Servo Speed 3</td>
            <td colspan="2">Head Servo Speed 4</td>
            <td colspan="2">Head Servo Speed 5</td>
        </tr>
        <tr>
            <td colspan="2">Head Servo Speed 6</td>
            <td colspan="4">Head Pose Position X</td>
            <td colspan="2">Head Pose Position Y</td>
        </tr>
        <tr>
            <td colspan="2">Head Pose Position Y</td>
            <td colspan="4">Head Pose Position Z</td>
            <td colspan="2">Head Pose Orientation W</td>
        </tr>
        <tr>
            <td colspan="2">Head Pose Orientation W</td>
            <td colspan="4">Head Pose Orientation X</td>
            <td colspan="2">Head Pose Orientation Y</td>
        </tr>
        <tr>
            <td colspan="2">Head Pose Orientation Y</td>
            <td colspan="4">Head Pose Orientation Z</td>
            <td>Is Head Pose Reachable</td>
            <td></td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name              | Type  | Description                                                                          |
| ----------------------- | ----- | ------------------------------------------------------------------------------------ |
| Torso Orientation       | float | This field contains the actual torso orientation in radian (0 to 2PI).               |
| Torso Servo Speed       | int16 | This field contains the actual torso servo speed in 0.229 rpm (-1023 to 1023).       |
| Head Servo Angle 1      | float | This field contains the actual head servo angle 1 in radian (0 to 2PI).              |
| Head Servo Angle 2      | float | This field contains the actual head servo angle 2 in radian (0 to 2PI).              |
| Head Servo Angle 3      | float | This field contains the actual head servo angle 3 in radian (0 to 2PI).              |
| Head Servo Angle 4      | float | This field contains the actual head servo angle 4 in radian (0 to 2PI).              |
| Head Servo Angle 5      | float | This field contains the actual head servo angle 5 in radian (0 to 2PI).              |
| Head Servo Angle 6      | float | This field contains the actual head servo angle 6 in radian (0 to 2PI).              |
| Head Servo Speed 1      | int16 | This field contains the actual head servo speed 1 in 0.229 rpm (-1023 to 1023).      |
| Head Servo Speed 2      | int16 | This field contains the actual head servo speed 2 in 0.229 rpm (-1023 to 1023).      |
| Head Servo Speed 3      | int16 | This field contains the actual head servo speed 3 in 0.229 rpm (-1023 to 1023).      |
| Head Servo Speed 4      | int16 | This field contains the actual head servo speed 4 in 0.229 rpm (-1023 to 1023).      |
| Head Servo Speed 5      | int16 | This field contains the actual head servo speed 5 in 0.229 rpm (-1023 to 1023).      |
| Head Servo Speed 6      | int16 | This field contains the actual head servo speed 6 in 0.229 rpm (-1023 to 1023).      |
| Head Pose Position X    | float | This field contains the actual X head position in meters in frame `stewart_base`.    |
| Head Pose Position Y    | float | This field contains the actual Y head position in meters in frame `stewart_base`.    |
| Head Pose Position Z    | float | This field contains the actual Z head position in meters in frame `stewart_base`.    |
| Head Pose Orientation W | float | This field contains the actual W head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation X | float | This field contains the actual X head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation Y | float | This field contains the actual Y head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation Z | float | This field contains the actual Z head orientation in meters in frame `stewart_base`. |
| Is Head Pose Reachable  | bool  | This field indicates if the last desired head pose is reachable.                     |


## IMU Data Payload
This message contains the status of all motors.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="4">Acceleration X</td>
            <td colspan="4">Acceleration Y</td>
        </tr>
        <tr>
            <td colspan="4">Acceleration Z</td>
            <td colspan="4">Angular Rate X</td>
        </tr>
        <tr>
            <td colspan="4">Angular Rate Y</td>
            <td colspan="4">Angular Rate Z</td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name     | Type  | Description                                                 |
| -------------- | ----- | ----------------------------------------------------------- |
| Acceleration X | float | This field contains the actual IMU X Acceleration in m/s².  |
| Acceleration Y | float | This field contains the actual IMU Y Acceleration in m/s².  |
| Acceleration Z | float | This field contains the actual IMU Z Acceleration in m/s².  |
| Angular Rate X | float | This field contains the actual IMU X Angular Rate in rad/s. |
| Angular Rate Y | float | This field contains the actual IMU X Angular Rate in rad/s. |
| Angular Rate Z | float | This field contains the actual IMU X Angular Rate in rad/s. |


## Set Torso Orientation Payload
This message sets the torso orientation.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="4">Torso Orientation</td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name              | Type  | Description                                                         |
| ----------------------- | ----- | ------------------------------------------------------------------- |
| Torso Orientation       | float | This field contains the new torso orientation in radian (0 to 2PI). |


## Set Head Pose Payload
This message sets the head pose.

### Table View
<table>
    <thead>
        <tr>
            <th colspan="8">Bytes</th>
        </tr>
    </thead>
    <thead>
        <tr>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>8</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="4">Head Pose Position X</td>
            <td colspan="4">Head Pose Position Y</td>
        </tr>
        <tr>
            <td colspan="4">Head Pose Position Z</td>
            <td colspan="4">Head Pose Orientation W</td>
        </tr>
        <tr>
            <td colspan="4">Head Pose Orientation X</td>
            <td colspan="4">Head Pose Orientation Y</td>
        </tr>
        <tr>
            <td colspan="4">Head Pose Orientation Z</td>
            <td colspan="4"></td>
        </tr>
    </tbody>
</table>

### Field Description
| Field Name              | Type  | Description                                                                       |
| ----------------------- | ----- | --------------------------------------------------------------------------------- |
| Head Pose Position X    | float | This field contains the new X head position in meters in frame `stewart_base`.    |
| Head Pose Position Y    | float | This field contains the new Y head position in meters in frame `stewart_base`.    |
| Head Pose Position Z    | float | This field contains the new Z head position in meters in frame `stewart_base`.    |
| Head Pose Orientation W | float | This field contains the new W head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation X | float | This field contains the new X head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation Y | float | This field contains the new Y head orientation in meters in frame `stewart_base`. |
| Head Pose Orientation Z | float | This field contains the new Z head orientation in meters in frame `stewart_base`. |
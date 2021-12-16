# 6. Audio Base Assembly

## X. Setup the battery charger

### Required Parts
| Part                     | Quantity | Image                                                          |
| ------------------------ | -------- | ---------------------------------------------------------------|
| `Micro USB Cable`        | 1        |                                                                |
| `19V Power Adapter`      | 1        |                                                                |

### Steps
1. Open the [project](../../firmwares/psu_control) in PlatformIO.
2. Change the value `FIRMWARE_MODE` in [config.h](../../firmwares/psu_controlinclude/config.h) to `FIRMWARE_MODE_SETUP_BATTERY_CHARGER`.
3. Connect the micro USB cable to the computer and the Teensy LC.
5. Turn off the robot.
6. Connect the `19V Power Adapter` to the base.
7. Turn on the robot.
8. Click on the upload button.

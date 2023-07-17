# 12. MCU Configuration

## Required Parts

| Part                     | Quantity |
| ------------------------ | -------- |
| `Micro USB Cable`        | 1        |
| `19V Power Adapter`      | 1        |
| `Assembled Robot`        | 1        |

## A. Computer Setup
1. Install [Visual Studio Code](https://code.visualstudio.com/download).
2. Install the [PlatformIO extension](https://platformio.org/platformio-ide).

## B. Setup the PSU Control PCB

1. Open the [project](../../firmwares/psu_control) in PlatformIO.
3. Change the value `FIRMWARE_MODE` in [config.h](../../firmwares/psu_control/include/config.h)
   to `FIRMWARE_MODE_NORMAL`.
4. Connect the micro USB cable to the computer and the robot base.
5. Turn off the robot.
6. Connect the `19V Power Adapter` to the base.
7. Turn on the robot.
8. Click on the upload button.

## C. Setup the Dynamixel Control PCB

1. Open the [project](../../firmwares/dynamixel_control) in PlatformIO.
3. Change the value `FIRMWARE_MODE` in [config.h](../../firmwares/psu_control/include/config.h)
   to `FIRMWARE_MODE_NORMAL`.
4. Connect the micro USB cable to the computer and the `Dynamixel Control PCB`.
5. Turn off the robot.
6. Connect the `19V Power Adapter` to the base.
7. Turn on the robot.
8. Click on the upload button.

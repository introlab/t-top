# 11. MCU Configuration

## Required Parts
| Part                     | Quantity | Image                                                          |
| ------------------------ | -------- | ---------------------------------------------------------------|
| `Micro USB Cable`        | 1        |                                                                |
| `Assembled Robot`        | 1        |                                                                |

## A. Computer Setup
1. Install [Visual Studio Code](https://code.visualstudio.com/download).
2. Install the [PlatformIO extension](https://platformio.org/platformio-ide).
3. Install the Arduino IDE.
    - [Windows](https://www.arduino.cc/en/Guide/Windows)
    - [Linux](https://www.arduino.cc/en/Guide/Linux)
4. Install the [OpenCR Board](https://emanual.robotis.com/docs/en/parts/controller/opencr10/#install-on-linux).

## B. Setup the Teensy LC
1. Open the [project](../../firmwares/psu_control) in PlatformIO.
2. Change the value `INA_TYPE` in [config.h](../../firmwares/psu_controlinclude/config.h) for the chip you have soldered (`INA220_TYPE` or `INA226_TYPE`).
3. Connect the micro USB cable to the computer and the robot base.
4. Click on the upload button.

## C. Setup the OpenCR
1. Modify `dynamixel_driver.cpp` (~/.arduino15/packages/OpenCR/hardware/OpenCR/1.4.9/libraries/DynamixelWorkbench/src/dynamixel_workbench_toolbox/dynamixel_driver.cpp).
    1. Comment out the `delay` line in `bool writeRegister(uint8_t, uint16_t, uint16_t, uint8_t*, const char**);`.
    2. Comment out the `delay` line in `bool writeRegister(uint8_t, const char*, int32_t, const char**);`.
    3. Comment out the `delay` line in `bool writeOnlyRegister(uint8_t, uint16_t, uint16_t, uint8_t*, const char**);`.
    4. Comment out the `delay` line in `bool writeOnlyRegister(uint8_t, const char*, int32_t, const char**);`.
2. Open the [project](../../firmwares/opencr_firmware) in the Arduino IDE.
3. Click on the upload button.

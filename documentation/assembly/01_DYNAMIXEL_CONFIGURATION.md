# Dynamixel Configuration

## Required Parts
| Part                     | Quantity | Image                                                          |
| ------------------------ | -------- | ---------------------------------------------------------------|
| `Dynamixel XL430-W250-T` | 7        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T.jpg) |
| `OpenCR`                 | 1        | ![OpenCR](images/electronics/OpenCR.jpg)                       |
| `Dynamixel cable`        | 1        | ![Dynamixel Cable](images/electronics/dynamixel%20cable.jpg)   |
| `Micro USB cable`        | 1        |                                                                |
| `12V power supply`       | 1        |     |


## Computer Setup
1. Install the Aarduino IDE.
    - [Windows](https://www.arduino.cc/en/Guide/Windows)
    - [Linux](https://www.arduino.cc/en/Guide/Linux)
2. Install the [OpenCR Board](https://emanual.robotis.com/docs/en/parts/controller/opencr10/#install-on-linux).

## Baud Rate Changes
1. Connect the `12V power supply` to the OpenCR.
2. Connect the `OpenCR` to the computer with the `micro USB cable`.
3. Make sure the `OpenCR` switch is OFF.
4. Connect the `OpenCR` and a `Dynamixel XL430-W250-T` with the `Dynamixel cable`.
5. Turn ON the `OpenCR` switch.
6. Use the `d_BPS_Change` example (`OpenCR/08. DynamixelWorkbench/d_BPS_Change`) to change the baud rate to 1000000.
    - Change the value of `DXL_ID` according to the current configuration.
    - Change the value of `BAUDRATE` according to the current configuration.
    - Change the value of `NEW_BAUDRATE` to 1000000.
7. Program the OpenCR.
8. Open the serial monitor.
9. Wait until the baud rate has changed.
10. Turn OFF the `OpenCR` switch.
11. Disconnect the `Dynamixel XL430-W250-T`.
12. Repeat steps 4 to 11 for each `Dynamixel XL430-W250-T`.

## Id Changes
1. Connect the `12V power supply` to the OpenCR.
2. Connect the `OpenCR` to the computer with the `micro USB cable`.
3. Make sure the `OpenCR` switch is OFF.
4. Connect the `OpenCR` and a `Dynamixel XL430-W250-T` with the `Dynamixel cable`.
5. Turn ON the `OpenCR` switch.
6. Use the `c_ID_Change` example (`OpenCR/08. DynamixelWorkbench/c_ID_Change`) to change the id.
    - Change the value of `BAUDRATE` to 1000000.
    - Change the value of `DXL_ID` according to the initial configuration.
    - Change the value of `NEW_DXL_ID` to 1.
7. Program the OpenCR.
8. Open the serial monitor.
9. Wait until the id has changed.
10. Turn OFF the `OpenCR` switch.
11. Disconnect the `Dynamixel XL430-W250-T`.
12. Identify the ID on the `Dynamixel XL430-W250-T`.
13. Repeat steps 4 to 12 for each `Dynamixel XL430-W250-T`, but increment the value of `NEW_DXL_ID`.

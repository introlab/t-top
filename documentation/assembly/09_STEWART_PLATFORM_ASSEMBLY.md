# 9. Stewart Platform Assembly

## A. Dynamixel Centering
### Required Parts
| Part                                             | Quantity | Image                                                          |
| ------------------------------------------------ | -------- | ---------------------------------------------------------------|
| `Dynamixel XL430-W250-T (ID 1, 2 3, 4, 5 and 6)` | 6        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T.jpg) |
| `OpenCR`                                         | 1        | ![OpenCR](images/electronics/OpenCR.jpg)                       |
| `Dynamixel Cable`                                | 1        | ![Dynamixel Cable](images/electronics/dynamixel%20cable.jpg)   |
| `Micro USB Cable`                                | 1        |                                                                |
| `12V Power Supply`                               | 1        |                                                                |
| `Computer`                                       | 1        |                                                                |

### Steps
1. Open the Arduino IDE.
2. Connect the `12V power supply` to the OpenCR.
3. Connect the `OpenCR` to the computer with the `micro USB cable`.
4. Make sure the `OpenCR` switch is OFF.
5. Connect the `OpenCR` and a `Dynamixel XL430-W250-T` with the `Dynamixel cable`.

![OpenCR, Dynamixel XL430-W250-T, Dynamixel cable](images/assemblies/02%20dynamixel.jpg)

6. Turn ON the `OpenCR` switch.
7. Use the `h_Position` example (`OpenCR/08. DynamixelWorkbench/h_Position`) to change the baud rate to 1000000.
    1. Change the value of `DXL_ID` according to the selected `Dynamixel XL430-W250-T`.
    2. Change the value of `BAUDRATE` to 1000000.
    3. Comment the line `dxl_wb.goalPosition(dxl_id, (int32_t)1023);`.
8. Program the OpenCR.
10. Wait until the `Dynamixel XL430-W250-T` is centered.
11. Turn OFF the `OpenCR` switch.
12. Disconnect the `Dynamixel XL430-W250-T`.
13. Identify the ID on the `Dynamixel XL430-W250-T`.
14. Repeat steps 4 to 13 for each `Dynamixel XL430-W250-T`.

## B. Odd-Id Dynamixel Pre-Assembly
### Required Parts
| Part                                     | Quantity | Image                                                                             |
| ---------------------------------------- | -------- | --------------------------------------------------------------------------------- |
| `Dynamixel XL430-W250-T (ID 1, 3 and 5)` | 3        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T%20ID1.jpg)              |
| `Dynamixel XL430-W250-T Screw`           | 1        | ![Dynamixel XL430-W250-T Screw](images/hardware/Dynamixel%20screw.jpg)            |
| `Stewart Horn (Right)`                   | 3        | ![Stewart Horn (Right)](images/3d%20printed%20parts/stewart%20horn%20(right).jpg) |
| `M2x12mm Socket`                         | 12       | ![M2x12mm Socket](images/hardware/M2x12mm%20socket.jpg)                           |

### Steps
1. Install the `stewart horn (right)` onto the `Dynamixel XL430-W250-T` with the `M2x12mm socket` screws, as shown in the following picture.

![Dynamixel Right](images/assemblies/09B%20dynamixel%20right.jpg)

2. Repeat step 1 for each `Dynamixel XL430-W250-T`.

![Dynamixel ID 1, 3 and 5](images/assemblies/09B%20dynamixel%201%203%205.jpg)

## C. Even-Id Dynamixel Pre-Assembly
### Required Parts
| Part                                     | Quantity | Image                                                                             |
| ---------------------------------------- | -------- | --------------------------------------------------------------------------------- |
| `Dynamixel XL430-W250-T (ID 2, 4 and 6)` | 3        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T%20ID2.jpg)              |
| `Dynamixel XL430-W250-T Screw`           | 1        | ![Dynamixel XL430-W250-T Screw](images/hardware/Dynamixel%20screw.jpg)            |
| `Stewart Horn (Left)`                    | 3        | ![Stewart Horn (Right)](images/3d%20printed%20parts/stewart%20horn%20(left).jpg)  |
| `M2x12mm Socket`                         | 12       | ![M2x12mm Socket](images/hardware/M2x12mm%20socket.jpg)                           |

### Steps
1. Install the `stewart horn (left)` onto the `Dynamixel XL430-W250-T` with the `M2x12mm socket` screws, as shown in the following picture.

![Dynamixel Left](images/assemblies/09C%20dynamixel%20left.jpg)

2. Repeat step 1 for each `Dynamixel XL430-W250-T`.

![Dynamixel ID 2, 4 and 6](images/assemblies/09C%20dynamixel%202%204%206.jpg)


## D. Stewart Platform Rod Pre-Assembly
### Required Parts
| Part                           | Quantity | Image                                                              |
| ------------------------------ | -------- | ------------------------------------------------------------------ |
| `Ball Joint - M3xL26mm Silver` | 12       | ![Ball joint - M3xL26mm Silver](images/hardware/ball%20joint.jpg)  |
| `Threaded Rod - M3x170mm`      | 6        | ![Threaded Rod - M3x170mm](images/hardware/stewart%20rod.jpg)      |

### Steps
1. Apply threadlocker the ends of a `threaded rod`.
2. Screw a ball joint to each ends of the `threaded rod` until the distance between the centers of the ball joint holes is 181 mm.
3. Unscrew one of the ball joints by 60°.
4. Repeat steps 1 to 3 for each `threaded rod`.

![Threaded Rod](images/assemblies/09D%20stewart%20rod.jpg)

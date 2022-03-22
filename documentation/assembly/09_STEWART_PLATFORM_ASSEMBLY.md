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
| `M2x12mm Socket Screw`                   | 12       | ![M2x12mm Socket Screw](images/hardware/M2x12mm%20socket.jpg)                     |

### Steps

1. Install the `stewart horn (right)` onto the `Dynamixel XL430-W250-T` with the `M2x12mm socket screws`, as shown in
   the following picture.

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
| `M2x12mm Socket Screw`                   | 12       | ![M2x12mm Socket Screw](images/hardware/M2x12mm%20socket.jpg)                     |

### Steps

1. Install the `stewart horn (left)` onto the `Dynamixel XL430-W250-T` with the `M2x12mm socket screws`, as shown in the
   following picture.

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
2. Screw a ball joint to each ends of the `threaded rod` until the distance between the centers of the ball joint holes
   is 181 mm.
3. Repeat steps 1 to 2 for each `threaded rod`.

![Threaded Rod](images/assemblies/09D%20stewart%20rod.jpg)

## E. OpenCR Pre-Assembly

### Required Parts

| Part                 | Quantity | Image                                                                     |
| -------------------- | -------- | ------------------------------------------------------------------------- |
| `OpenCR`             | 1        | ![OpenCR](images/electronics/OpenCR.jpg)                                  |
| `Grove Base Shield`  | 1        | ![Grove Base Shield](images/electronics/Grove%20base%20shield.jpg)        |
| `OpenCR support 1`   | 1        | ![OpenCR support 1](images/3d%20printed%20parts/OpenCR%20support%201.jpg) |
| `OpenCR support 2`   | 1        | ![OpenCR support 2](images/3d%20printed%20parts/OpenCR%20support%202.jpg) |
| `M3x8 Plastic Screw` | 4        | ![M3x8 Plastic Screw](images/hardware/M3x8mm%20plastic.jpg)              |

### Steps

1. Install the `Grove Base Shield` onto the `OpenCR`.

![OpenCR + Grove Base Shield](images/assemblies/09E%20OpenCR.jpg)

2. Install the `ÒpenCR` onto the `OpenCR supports` with `M3x8 plastic screws`, as shown in the following picture.

![OpenCR Supports](images/assemblies/09E%20OpenCR%20supports.jpg)

## F. Stewart Platform

### Required Parts

| Part                                                       | Quantity | Image                                                                                      |
| ---------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| `Stewart Bottom`                                           | 1        | ![Stewart Bottom](images/3d%20printed%20parts/stewart%20bottom.jpg)                        |
| `Stewart Top`                                              | 1        | ![Stewart Top](images/3d%20printed%20parts/stewart%20top.jpg)                              |
| `USB Hub Support`                                          | 4        | ![USB Hub Support](images/3d%20printed%20parts/USB%20hub%20support.jpg)                    |
| `Assembled Dynamixel XL430-W250-T (ID 1, 2 3, 4, 5 and 6)` | 6        | ![Dynamixel XL430-W250-T](images/assemblies/09B%20dynamixel%201%203%205.jpg) ![Dynamixel ID 2, 4 and 6](images/assemblies/09C%20dynamixel%202%204%206.jpg) |
| `Dynamixel Cable`                                          | 6        | ![Dynamixel Cable](images/electronics/dynamixel%20cable.jpg)                               |
| `Dynamixel XL430-W250-T Cable Cover`                       | 6        | ![Dynamixel XL430-W250-T Cable Cover](images/electronics/XL430-W250-T%20cable%20cover.jpg) |
| `Assembled OpenCR`                                         | 1        | ![OpenCR](images/assemblies/09E%20OpenCR%20supports.jpg)                                   |
| `Assembled Stewart Platform Rod`                           | 6        | ![Threaded Rod](images/assemblies/09D%20stewart%20rod.jpg)                                 |
| `USB Hub`                                                  | 2        | ![USB Hub](images/electronics/USB%20hub.jpg)                                               |
| `Micro-USB Cable Included with the Touchscreen`            | 1        | ![Micro-USB Cable](images/electronics/Micro-USB%20cable%20screen.jpg)                      |
| `M2.5x8 Plastic Screw`                                     | 36       | ![M2.5x8 Plastic Screw](images/hardware/M2.5x8mm%20plastic.jpg)                            |
| `M3x12 Plastic Screw`                                      | 24       | ![M3x12 Plastic Screw](images/hardware/M3x12mm%20plastic.jpg)                              |

### Steps

1. Place the `Assembled Dynamixel XL430-W250-T with the ID 1` under the `Stewart bottom`, according to the following
   picture.

![Dynamixel IDS](images/assemblies/09F%20ids.jpg)

2. Install the `Assembled Dynamixel XL430-W250-T with the ID 1` with `M2.5x8 plastic screws`.

![Dynamixel Screws](images/assemblies/09F%20dynamixel%20screws%201.jpg)
![Dynamixel Screws](images/assemblies/09F%20dynamixel%20screws%202.jpg)
![Dynamixel Screws](images/assemblies/09F%20dynamixel%20screws%203.jpg)

3. Repeat steps 1 to 2 for the other `Assembled Dynamixel XL430-W250-T`.

![All Dynamixel](images/assemblies/09F%20all%20dynamixel.jpg)

4. Connect the `Dynamixel with the ID 1` and the `Dynamixel with the ID 2` together with a `Dynamixel cable`.

![Dynamixel Cable](images/assemblies/09F%20dynamixel%20cable.jpg)

5. Connect the `Dynamixel with the ID 3` and the `Dynamixel with the ID 4` together with a `Dynamixel cable`.
6. Connect the `Dynamixel with the ID 5` and the `Dynamixel with the ID 6` together with a `Dynamixel cable`.
7. Connect the `Dynamixel with the ID 2` and the `Dynamixel with the ID 3` together with a `Dynamixel cable`.
8. Connect a `Dynamixel cable` into the `Dynamixel with the ID 1`.
9. Connect a `Dynamixel cable` into the `Dynamixel with the ID 6`.
10. Install the `Dynamixel XL430-W250-T cable covers` onto all `Dynamixel`.
11. Install tie wraps to hold the cables, as shown in the following picture.

![Dynamixel Tie Wrap](images/assemblies/09F%20cable%20tie%20wrap.jpg)

12. Install the `USB hub supports` with `M3x12 plastic screws`, as shown in the following picture.

![USB Hub Supports](images/assemblies/09F%20USB%20hub%20supports.jpg)

13. Connect the `Dynamixel cable` of the `Dynamixel with the ID 1` to the OpenCR.
14. Connect the `Dynamixel cable` of the `Dynamixel with the ID 6` to the OpenCR.

![OpenCR Cables](images/assemblies/09F%20OpenCR%20cables.jpg)

15. Install the `OpenCR` with `M3x12 plastic screws`, as shown in the following pictures.

![OpenCR Scews](images/assemblies/09F%20OpenCR%20screws%201.jpg)
![OpenCR Scews](images/assemblies/09F%20OpenCR%20screws%202.jpg)

16. Install the `USB hubs` with tie wraps, as shown in the following picture.

![USB Hubs](images/assemblies/09F%20USB%20hubs.jpg)

17. Install an `assembled Stewart platform rod` onto each `Stewart horn` with a `M3x12 plastic screw`, as shown in the
    following picture.

![Bottom Rod](images/assemblies/09F%20bottom%20rod.jpg)

18. Align the top `ball joints` with the nearest red line, as shown in the following picture.

![Ball Joint](images/assemblies/09F%20ball%20joint.jpg)

19. Install the `Stewart top` onto the top `ball joints` with `M3x12 plastic screws`, as shown in the following
    pictures.

![Stewart Top](images/assemblies/09F%20stewart%20top%201.jpg)
![Stewart Top](images/assemblies/09F%20stewart%20top%202.jpg)

20. Connect the `OpenCR` to a `USB hub` with the `micro-USB cable`, as shown in the following picture.

![OpentCR Cable](images/assemblies/09F%20USB%20cable.jpg)

21. Connect the `Dynamixel cables` to the `OpenCR`.

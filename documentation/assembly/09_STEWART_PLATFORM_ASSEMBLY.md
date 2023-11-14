# 9. Stewart Platform Assembly

## A. Dynamixel Centering

### Required Parts

| Part                                             | Quantity | Image                                                          |
| ------------------------------------------------ | -------- | ---------------------------------------------------------------|
| `Dynamixel XL430-W250-T (ID 1, 2 3, 4, 5 and 6)` | 6        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T.jpg) |
| `OpenCR`                                         | 1        | ![OpenCR](images/electronics/OpenCR.jpg)                       |
| `Dynamixel Cable`                                | 1        | ![Dynamixel Cable](images/electronics/dynamixel-cable.jpg)     |
| `Micro USB Cable`                                | 1        |                                                                |
| `12V Power Supply`                               | 1        |                                                                |
| `Computer`                                       | 1        |                                                                |

### Steps

1. Open the [project](../../firmwares/dynamixel_centering/) in PlatformIO.
2. Connect the `Dynamixel control PCB` to the computer with the `micro USB cable`.
3. Connect the `Dynamixel control PCB` and a `Dynamixel XL430-W250-T` with the `Dynamixel cable`.
4. Connect the `12V power supply` to the `Dynamixel control PCB`.
5. Change the value of `MOTOR_ID` according to the current `Dynamixel XL430-W250-T.
6. Program the `Dynamixel control PCB`.
7. Open the serial monitor.
8. Wait until the `Dynamixel XL430-W250-T` is centered.
9. Repeat steps 3 to 8 for each `Dynamixel XL430-W250-T`.

## B. Dynamixel Pre-Assembly

### Required Parts

| Part                                           | Quantity | Image                                                                             |
| ---------------------------------------------- | -------- | --------------------------------------------------------------------------------- |
| `Dynamixel XL430-W250-T (ID 1, 2, 3, 4 and 5)` | 6        | ![Dynamixel XL430-W250-T](images/electronics/XL430-W250-T-id1.jpg)                |
| `Stewart Horn Left M2.5`                       | 2        | ![Stewart Horn Left M2.5](images/3d-printed-parts/stewart-horn-left-M2.5.jpg)     |
| `Stewart Horn Left M3`                         | 2        | ![Stewart Horn Left M3](images/3d-printed-parts/stewart-horn-left-M3.jpg)         |
| `Stewart Horn Right M2.5`                      | 2        | ![Stewart Horn Right M2.5](images/3d-printed-parts/stewart-horn-right-M2.5.jpg)   |
| `Stewart Horn Right M3`                        | 2        | ![Stewart Horn Right M3](images/3d-printed-parts/stewart-horn-right-M3.jpg)       |
| `M2x12mm Socket Screw`                         | 24       | ![M2x12mm Socket Screw](images/hardware/M2x12mm-socket.jpg)                       |

### Steps

1. Install a `stewart horn left M2.5` onto each `Dynamixel XL430-W250-T with the ID 2 and 6` with the `M2x12mm socket screws`, as shown in
   the following picture.

![Stewart Horn Left M2.5](images/assemblies/09/stewart-horn-left-M2.5.jpg)

2. Install the `stewart horn left M3` onto the `Dynamixel XL430-W250-T with the ID 4` with the `M2x12mm socket screws`, as shown in
   the following picture.

![Stewart Horn Left M3](images/assemblies/09/stewart-horn-left-M3.jpg)

3. Install a `stewart horn right M2.5` onto each `Dynamixel XL430-W250-T with the ID 1 and 5` with the `M2x12mm socket screws`, as shown in
   the following picture.

![Stewart Horn Right M2.5](images/assemblies/09/stewart-horn-right-M2.5.jpg)

4. Install the `stewart horn right M3` onto the `Dynamixel XL430-W250-T with the ID 3` with the `M2x12mm socket screws`, as shown in
   the following picture.

![Stewart Horn Right M3](images/assemblies/09/stewart-horn-right-M3.jpg)

## C. Stewart Platform Rod Pre-Assembly

### Required Parts

| Part                           | Quantity | Image                                                              |
| ------------------------------ | -------- | ------------------------------------------------------------------ |
| `Ball Joint - M3xL20mm Silver` | 12       | ![Ball joint - M3xL26mm Silver](images/hardware/ball-joint.jpg)    |
| `Threaded Rod - M3x170mm`      | 6        | ![Threaded Rod - M3x180mm](images/hardware/stewart-rod.jpg)        |

### Steps

1. Apply threadlocker to the ends of a `threaded rod`.
2. Screw a ball joint to each ends of the `threaded rod` until the distance between the centers of the ball joint holes
   is 191 mm.
3. Repeat steps 1 to 2 for each `threaded rod`.

![Threaded Rod](images/assemblies/09/stewart-rod.jpg)

## D. Stewart Platform

### Required Parts

| Part                                                       | Quantity | Image                                                                                      |
| ---------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| `Stewart Bottom 1`                                         | 1        | ![Stewart Bottom 1](images/3d-printed-parts/stewart-bottom-1.jpg)                          |
| `Stewart Bottom 2`                                         | 1        | ![Stewart Bottom 2](images/3d-printed-parts/stewart-bottom-2.jpg)                          |
| `Stewart Bottom 3`                                         | 1        | ![Stewart Bottom 3](images/3d-printed-parts/stewart-bottom-3.jpg)                          |
| `Stewart Top`                                              | 1        | ![Stewart Top](images/3d-printed-parts/stewart-top.jpg)                                    |
| `USB Hub Stand`                                            | 4        | ![USB Hub Support](images/3d-printed-parts/USB-hub-stand.jpg)                              |
| `Assembled Dynamixel XL430-W250-T (ID 1, 2 3, 4, 5 and 6)` | 6        | ![Stewart Horn Left M2.5](images/assemblies/09/stewart-horn-left-M2.5.jpg) ![Stewart Horn Left M3](images/assemblies/09/stewart-horn-left-M3.jpg) ![Stewart Horn Right M2.5](images/assemblies/09/stewart-horn-right-M2.5.jpg) ![Stewart Horn Right M3](images/assemblies/09/stewart-horn-right-M3.jpg) |
| `Dynamixel Cable`                                          | 7        | ![Dynamixel Cable](images/electronics/dynamixel-cable.jpg)                                 |
| `Dynamixel XL430-W250-T Cable Cover`                       | 6        | ![Dynamixel XL430-W250-T Cable Cover](images/electronics/XL430-W250-T-cable-cover.jpg)     |
| `Dynamixel Control PCB`                                    | 1        | ![Dynamixel Control PCB](images/assemblies/02/dynamixel-control-7.jpg)                     |
| `Assembled Stewart Platform Rod`                           | 6        | ![Threaded Rod](images/assemblies/09/stewart-rod.jpg)                                      |
| `USB Hub`                                                  | 2        | ![USB Hub](images/electronics/usb-hub.jpg)                                                 |
| `M3x6 Plastic Screw`                                       | 6        | ![M3x6 Plastic Screw](images/hardware/M3x6mm-plastic.jpg)                                  |
| `M2.5x8 Plastic Screw`                                     | 46       | ![M2.5x8 Plastic Screw](images/hardware/M2.5x8mm-plastic.jpg)                              |
| `M2.5x12 Plastic Screw`                                    | 4        | ![M2.5x12 Plastic Screw](images/hardware/M2.5x12mm-plastic.jpg)                            |
| `M3x8 Plastic Screw`                                       | 4        | ![M3x8 Plastic Screw](images/hardware/M3x8mm-plastic.jpg)                                  |
| `M3x12 Plastic Screw`                                      | 14       | ![M3x12 Plastic Screw](images/hardware/M3x12mm-plastic.jpg)                                |
| `Big Spring`                                               | 2        | ![Big Spring](images/hardware/big-spring.jpg)                                              |
| `Small Spring`                                             | 4        | ![Small Spring](images/hardware/small-spring.jpg)                                          |

### Steps

1. Install the `Stewart bottom` parts together with the `M3x6 plastic screws`, as shown in the first following picture. Sand the part if needed.

![Stewart Bottom](images/assemblies/09/stewart-bottom.jpg)

2. Place the `Assembled Dynamixel XL430-W250-T with the ID 1` under the `Stewart bottom`, according to the following
   picture.

![Dynamixel IDS](images/assemblies/09/ids.jpg)

3. Install the `Assembled Dynamixel XL430-W250-T with the ID 1` with `M2.5x8 plastic screws`.

![Dynamixel Screws](images/assemblies/09/dynamixel-screws-1.jpg)
![Dynamixel Screws](images/assemblies/09/dynamixel-screws-2.jpg)
![Dynamixel Screws](images/assemblies/09/dynamixel-screws-3.jpg)

4. Repeat steps 2 to 3 for the other `Assembled Dynamixel XL430-W250-T`.

5. Install a `M3x8 plastic screw` into the horn of the `Assembled Dynamixel XL430-W250-T with the ID 3`, as as shown in the following picture.

![Spring Screw](images/assemblies/09/big-spring-M3x8.jpg)

6. Install a  `M3x12 plastic screw` into the Stewart Bottom near the `Assembled Dynamixel XL430-W250-T with the ID 3`, as as shown in the following picture.

![Spring Screw](images/assemblies/09/big-spring-M3x12.jpg)

7. Install a `big spring` as as shown in the following picture.

![Spring Screw](images/assemblies/09/big-spring.jpg)

8. Repeat steps 5 to 7 for the`Assembled Dynamixel XL430-W250-T with the ID 4`.

9. Install a `M2.5x8 plastic screw` and a `small spring` into the horn of the `Assembled Dynamixel XL430-W250-T with the ID 5`, as as shown in the following picture.

![Spring Screw](images/assemblies/09/small-spring-M2.5x8.jpg)

10. Install a `M2.5x12 plastic screw` and the `small spring` into the Stewart Bottom near the `Assembled Dynamixel XL430-W250-T with the ID 5`, as as shown in the following picture.

![Spring Screw](images/assemblies/09/small-spring-M2.5x12.jpg)

11. Repeat steps 9 to 10 for the`Assembled Dynamixel XL430-W250-T with the ID 1, 2 and 6`.

12. Connect the `Dynamixel with the ID 3` and the `Dynamixel with the ID 4` together with a `Dynamixel cable`.
13. Connect the `Dynamixel with the ID 4` and the `Dynamixel with the ID 5` together with a `Dynamixel cable`.
14. Connect the `Dynamixel with the ID 2` and the `Dynamixel with the ID 1` together with a `Dynamixel cable`.
15. Connect the `Dynamixel with the ID 1` and the `Dynamixel with the ID 6` together with a `Dynamixel cable`.
16. Connect a `Dynamixel cable` into the `Dynamixel with the ID 5`.
17. Connect a `Dynamixel cable` into the `Dynamixel with the ID 6`.
18. Connect a `Dynamixel cable` into the `Dynamixel with the ID 3`.
19. Install the `Dynamixel XL430-W250-T cable covers` onto all `Dynamixel`.
20. Install the `USB hub stands` with `M3x12 plastic screws`, as shown in the following picture.

![USB Hub Stands](images/assemblies/09/USB-hub-stands.jpg)

21. Connect the `Dynamixel cable` of the `Dynamixel with the ID 5` to the `Dynamixel Control PCB`.
22. Connect the `Dynamixel cable` of the `Dynamixel with the ID 6` to the `Dynamixel Control PCB`.
23. Install the `Dynamixel Control PCB` with `M3x8 plastic screws`, as shown in the following picture.

![Dynamixel Control PCB Scews](images/assemblies/09/dynamixel-control-PCB.jpg)

24. Install tie wraps to hold the cables, as shown in the following picture.

![Dynamixel Tie Wrap](images/assemblies/09/cable-tie.jpg)

25. Install an `assembled Stewart platform rod` onto each `Stewart horn` with a `M3x12 plastic screw`, as shown in the
    following picture.

![Bottom Rod](images/assemblies/09/bottom-rod.jpg)

26. Align the top `ball joints` with the nearest red line, as shown in the following picture.

![Ball Joint](images/assemblies/09/ball-joint-alignment.jpg)

27. Install the `Stewart top` onto the top `ball joints` with `M3x12 plastic screws`, as shown in the following
    pictures.

![Stewart Top](images/assemblies/09/stewart-top-1.jpg)
![Stewart Top](images/assemblies/09/stewart-top-2.jpg)

28. Install the `USB hubs` with tie wraps, as shown in the following picture.

![USB Hubs](images/assemblies/09/USB-hubs.jpg)

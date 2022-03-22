# 11. Robot Assembly

## A. Base/Torso Bottom Assembly

### Required Parts

| Part                         | Quantity | Image                                                                             |
| ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| `Base Assembly`              | 1        |                                                                                   |
| `Assembled Slip Ring`        | 1        |                                                                                   |
| `1/8-Inch Stereo Jack Cable` | 2        |                                                                                   |
| `16SoundsUSB`                | 1        | ![16SoundsUSB](images/electronics/16SoundsUSB.jpg)                                |
| `M3x8mm Plastic Screw`       | 4        | ![M3x8mm Plastic Screw](images/hardware/M3x8mm%20plastic.jpg)                     |
| `16SoundsUSB Cable`          | 16       | ![16SoundsUSB Cable](images/assemblies/04A%20crossover%20RJ12%20flat%20cable.jpg) |
| `Torso Bottom Assembly`      | 1        |                                                                                   |
| `M5x10mm Plastic Screw`      | 3        | ![M5x10mm Plastic Screw](images/hardware/M5x10mm%20plastic.jpg)                   |
| `M3x12mm Plastic Screw`      | 16       | ![M3x12mm Plastic Screw](images/hardware/M3x12mm%20plastic.jpg)                   |
| `Computer Power Connector`   | 1        | ![Computer Power Connector](images/assemblies/04J%20computer.jpg)                 |
| `OpenCR Power Connector`     | 1        | ![OpenCR Power Connector](images/assemblies/04J%20opencr.jpg)                     |

### Steps

1. Connect the bottom `Ethernet connector` of the `assembled slip sing` to the base connector.
2. Connect the bottom `power connectors` of the `assembled slip sing` to `power connectors` connected
   the `Buck-Boost PCBs`.
3. Connect the `JST XH connector` of the `assembled slip sing` to the following connector of the `PSU Control PCB`.

![JST XH](images/assemblies/11A%20JST%20XH.jpg)

4. Connect the `1/8-Inch stereo jack cables` to the `ground loop noise isolators`.
5. Connect the `1/8-Inch stereo jack cables` to the `16SoundsUSB`.
6. Connect the `Mini-USB cable` to the `16SoundsUSB`.
7. Install the `16SoundsUSB` into the base with `M3x8mm plastic screws`, as shown in the following picture.

![16SoundsUSB](images/assemblies/11A%20sound%20card.jpg)

8. Install the `assembled slip ring` into the `torso bottom assembly` with `M5x10mm plastic screws`, as shown in the
   following picture.

![Slip Ring](images/assemblies/11A%20torso%20slip%20ring.jpg)

9. Connect all `16SoundsUSB cables` as shown in the following picture.

![16SoundsUSB Cables](images/assemblies/11A%20sound%20card%20cables.jpg)

10. Install the `torso bottom assembly` onto the base with `M3x12mm plastic screws`, as shown in the following picture.

![Torso](images/assemblies/11A%20torso.jpg)

11. Connect the `computer power connector` to the top of the `power connector` of the `assembled slip sing`.
12. Connect the `OpenCR power connector` to the top of the `power connector` of the `assembled slip sing`.
13. Connect the `computer power connector` and the the `Ethernet connector` to the `Nvidia Jetson AGX Xavier`, as shown
    in the following picture.

![Computer Connectors](images/assemblies/11A%20computer%20cables.jpg)

14. Place the `assembled slip sing cables` as shown in the following pictures.

![Cables 1](images/assemblies/11A%20cables%201.jpg)
![Cables 2](images/assemblies/11A%20cables%202.jpg)

## B. Torso Bottom/Stewart Platform Assembly

### Required Parts

| Part                         | Quantity | Image                                                                             |
| ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| `Torso Bottom Assembly`      | 1        |                                                                                   |
| `Stewart Platform Assembly`  | 1        |                                                                                   |
| `M3x12mm Plastic Screw`      | 6        | ![M3x12mm Plastic Screw](images/hardware/M3x12mm%20plastic.jpg)                   |

### Steps

1. Place the `Stewart platform assembly` onto the `torso bottom assembly` without installing the screws, as shown in the
   following picture.

![Stewart Platform Assembly](images/assemblies/11B%20stewart.jpg)

2. Connect the `OpenCR power connector` to the `OpenCR` (the cable must pass through the `Stewart bottom` hole).
3. Connect the `limit switch connector` to the `OpenCR` (the cable must pass through the `Stewart bottom` hole).

![Limit Switch](images/assemblies/11B%20limit%20switch.jpg)

4. Connect the `slip ring Grove connector` to the `OpenCR` (the cable must pass through the `Stewart bottom` hole).

![Slip Ring Grove](images/assemblies/11B%20slip%20ring%20grove.jpg)

5. Connect the `USB hubs` to `Nvidia Jetson AGX Xavier` (the cables must pass through the `Stewart bottom` hole).
6. Connect the `HDMI cable`to `Nvidia Jetson AGX Xavier` (the cable must pass through the `Stewart bottom` hole).
7. Install the `M3x12mm plastic screws` to fix the `Stewart platform assembly` onto the `torso bottom assembly`.

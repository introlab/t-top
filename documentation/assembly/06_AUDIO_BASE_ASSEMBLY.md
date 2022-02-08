# 6. Audio Base Assembly

## A. Audio Base Pre-Assembly

### Required Parts
| Part                                       | Quantity | Image                                                                                                              |
| ------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------ |
| `Base Audio 1`                             | 1        | ![Base Audio 1](images/3d%20printed%20parts/base%20audio%201.jpg)                                                  |
| `Base Audio 2`                             | 1        | ![Base Audio 2](images/3d%20printed%20parts/base%20audio%202.jpg)                                                  |
| `Assembled Speaker`                        | 4        | ![Assembled Speaker](images/electronics/assembled%20speaker.jpg)                                                   |
| `Assembled Fan`                            | 2        | ![Assembled Fan](images/electronics/assembled%20fan.jpg)                                                           |
| `Micro USB Connector`                      | 1        | ![Micro USB Connector](images/electronics/micro%20usb%20connector.jpg)                                             |
| `Assembled Power Switch`                   | 1        | ![Assembled Power Switch](images/electronics/assenbled%20power%20switch.jpg)                                       |
| `Ethernet Connector`                       | 1        | ![Ethernet Connector](images/electronics/ethernet%20connector.jpg)                                                 |
| `Assembled Robot Power Connector - Female` | 1        | ![Assembled Robot Power Connector - Female](images/electronics/assembled%20robot%20power%20connector%20female.jpg) |
| `Assembled LED`                            | 6        | ![Assembled LED](images/electronics/assembled%20LEDs.jpg)                                                          |
| `M3x12mm Plastic Screw`                    | 4        | ![M3x12mm Plastic Screw](images/hardware/M3x12mm%20plastic.jpg)                                                    |
| `M3x8mm Plastic Screw`                     | 16       | ![M3x8mm Plastic Screw](images/hardware/M3x8mm%20plastic.jpg)                                                      |
| `Fan Screw`                                | 8        | ![Fan Screw](images/hardware/fan%20screw.jpg)                                                                      |

### Steps
1. Insert the `base audio 1` part into the `base audio 2` part, as shown in the following picture.

![Base Audio Parts](images/assemblies/06A%20base%20audio.jpg)

2. Put 4 `M3x12 plastic screws` to fix the parts together, as shown in the following pictures.

![Base Audio Screw 1](images/assemblies/06A%20base%20audio%20screw%201.jpg)

![Base Audio Screw 1](images/assemblies/06A%20base%20audio%20screw%202.jpg)

3. Install the `assembled speakers` with `M3x8 plastic screws`, as shown in the following pictures. The wire lengths are shown in the picture.

![Base Audio Speaker 1](images/assemblies/06A%20speaker%201.jpg)

![Base Audio Speaker 2](images/assemblies/06A%20speaker%202.jpg)

4. Install the `assembled fans` with `fan screws`, as shown in the following pictures. The wire lengths are shown in the picture.

![Base Audio Fan 1](images/assemblies/06A%20fan%201.jpg)
![Base Audio Fan 2](images/assemblies/06A%20fan%202.jpg)

5. Install the `Micro USB connector` as shown in the following picture.

![Base Audio Micro USB](images/assemblies/06A%20micro%20USB%20connector.jpg)

6. Install the `assembled power switch` as shown in the following picture.

![Base Audio Power Switch](images/assemblies/06A%20power%20switch.jpg)

7. Install the `Ethernet connector` as shown in the following picture.

![Base Audio Ethernet](images/assemblies/06A%20ethernet%20connector.jpg)

8. Install the `assembled robot power connector` with epoxy glue, as shown in the following picture.

![Base Audio Power Connector](images/assemblies/06A%20power%20connector.jpg)

9. Install the `assenbled LEDs` with epoxy glue, as shown in the following picture.

![Base Audio LEDs](images/assemblies/06A%20LEDs.jpg)

## X. Setup the battery charger

### Required Parts
| Part                     | Quantity | Image                                                          |
| ------------------------ | -------- | -------------------------------------------------------------- |
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

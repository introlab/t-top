# T-Top

T-Top is a tabletop robot designed with advanced audio and vision sensors, deep learning perceptual processing and
telecommunication capabilities to provide richer interaction modalities and develop higher cognitive abilities from
interacting with people.

[![T-Top](images/t_top_video.jpg)](https://www.youtube.com/watch?v=2jpqTp6jozc)

![T-Top](images/t_top.jpg)

## Authors

- Marc-Antoine Maheux (@mamaheux)
- Charles Caya (@chcaya)
- Alexandre Filion (@alexfilion)
- Dominic Létourneau (@doumdi)
- Philippe Warren (@philippewarren)
- Ian-Mathieu Joly (@joli-1801)

## Licensing

- Source code files: [GPLv3](LICENSE_SOURCE_CODE)
- Other files: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](LICENSE_OTHER)

## Features

| Category         | Type              | Description                                                                                                                                                                                                                                             |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Power            | Power Adapter     | 19 V                                                                                                                                                                                                                                                    |
|                  | Battery           | 1x [RRC2054-2](https://www.rrc-ps.com/en/battery-packs/standard-battery-packs/products/RRC2054-2)                                                                                                                                                       |
|                  | Battery Charger   | 1x [RRC-PMM240](https://www.rrc-ps.com/en/battery-packs/standard-battery-packs/products/RRC-PMM240)                                                                                                                                                     |
| Sensors          | Microphone Array  | 16x [xSoundsMicrophones](https://github.com/introlab/xSoundsMicrophones), 1x [16SoundsUSB](https://github.com/introlab/16SoundsUSB)                                                                                                                     |
|                  | RGB-D Camera      | 1x [Intel RealSense D435i](https://www.intelrealsense.com/depth-camera-d435i/)                                                                                                                                                                          |
|                  | Wide Angle Camera | 1x [Arducam AR0230](https://www.uctronics.com/arducam-1080p-hd-wide-angle-wdr-usb-camera-module-for-computer-2mp-1-2-7-cmos-ar0230-100-degree-mini-uvc-usb2-0-spy-webcam-board-with-3-3ft-1m-cable-for-windows-linux-mac-os-android.html)               |
|                  | Touchscreen       | 1x 7 inch 1024x600 capacitive touchscreen                                                                                                                                                                                                               |
|                  | Current/Voltage   | [INA220](https://www.ti.com/product/INA220)                                                                                                                                                                                                             |
|                  | Light Sensors     | 4x [Adafruit ALS-PT19 ](https://www.adafruit.com/product/2748)                                                                                                                                                                                          |
|                  | Buttons           | 4x buttons                                                                                                                                                                                                                                              |
| Actuators        | Stewart Platform  | Displacement range: ±3 cm (x, y and z), ±20° (x and y), ±30° (z). Motor: [Dynamixel XL430-W250](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/)                                                                                                  |                                                         |
|                  | Rotating Base     | Displacement range: illimited. Motor: [Dynamixel XL430-W250](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/)                                                                                                                                     |
|                  | Speakers          | 4x [Dayton Audio DMA45-8](https://www.daytonaudio.com/product/1613/dma45-8-1-1-2-dual-magnet-aluminum-cone-full-range-driver-8-ohm), 2x [MAX9744](https://www.adafruit.com/product/1752)                                                                |
|                  | Cooling           | 2x [Noctua NF-A4x20 5V](https://noctua.at/en/products/fan/nf-a4x20-5v)                                                                                                                                                                                  |
|                  | Touchscreen       | 1x 7 inch 1024x600 capacitive touchscreen                                                                                                                                                                                                               |
|                  | LED               | Battery status, volume level, led strip                                                                                                                                                                                                                 |
| Network          | WiFi              | Intel Dual Band Wireless-AC 8265 NGW                                                                                                                                                                                                                    |
|                  | Ethernet          | 100 Mbps                                                                                                                                                                                                                                                |
| Processing       | Computer          | [NVIDIA Jetson AGX Xavier Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit) or [NVIDIA Jetson AGX Xavier Orin Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit)                   |
|                  | Motor MCU         | [Teensy 4.0](https://www.pjrc.com/store/teensy40.html)                                                                                                                                                                                                  |
|                  | Battery MCU       | [Teensy LC](https://www.pjrc.com/teensy/teensyLC.html)                                                                                                                                                                                                  |
| Perceptions      |                   | SLAM, object detection, person pose estimation, face recognition, sound classification, speaker identification, robot name detection, speech to text, person identification, music beat detection, source source localization, ego noise reduction, vad |
| Behaviors        |                   | Telepresence, emotions, talking, greeting, face following, dancing, exploring, sound following                                                                                                                                                          |

## Repository Structure

- The [documentation](documentation) folder contains the documentation to build and configure T-Top.
- The [CAD](CAD) folder contains the SolidWorks files of the robot. Il also contains the DXF and STL files to cut and
  print custom parts.
- The [firmwares](firmwares) folder contains the firmware for the MCUs.
- The [PCB](PCB) folder contains the KiCad files of the custom PCBs. Il also contains the Gerber files to manufacture
  the PCBs.
- The [ros](ros) folder contains the ROS packages to use the robot.
- The [tools](tools) folder contains the tools to develop and use the robot.

## Development Computer Setup

See [01_COMPUTER_CONFIGURATION.md](documentation/assembly/01_COMPUTER_CONFIGURATION.md#development-computer-ubuntu-2004)

## Papers

- [M.-A. Maheux, D. Létourneau, P. Warren, A. M. Panchea, J. M. Robillard and F. Michaud, “‍Designing a Tabletop SAR as an Advanced HRI Experimentation Platform,” in Proceedings of the 2024 International Symposium on Technological Advances in Human-Robot Interaction (TAHRI 2024), Boulder, CO, USA, 2024](https://dl.acm.org/doi/10.1145/3648536.3648538)
    - [YouTube Video](https://youtu.be/HSd65F055VE)
- [A. M. Panchea, M.-A. Maheux, P. Warren, D. Létourneau and F. Michaud, “Exploratory Evaluation of a Tabletop Robot With Older Adults,” in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, Boulder, CO, USA, 2024, pp. 817–821.](https://dl.acm.org/doi/10.1145/3610978.3640621)
- [M.-A. Maheux, A. M. Panchea, P. Warren, D. Létourneau and F. Michaud, “T-Top, an Open Source Tabletop Robot with Advanced Onboard Audio, Vision and Deep Learning Capabilities,” in Proceedings of the 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Detroit, MI, USA, 2023, pp. 7862-7869.](https://ieeexplore.ieee.org/abstract/document/10342252)
- [M.-A. Maheux, C. Caya, D. Létourneau and F. Michaud, “T-Top, a sar experimental platform,” in Proceedings of the 2022 ACM/IEEE International Conference on Human-Robot Interaction, 2022, pp. 904–908.](https://dl.acm.org/doi/abs/10.5555/3523760.3523902)

## Sponsor

![IntRoLab](https://introlab.3it.usherbrooke.ca/IntRoLab.png)

[IntRoLab - Intelligent / Interactive / Integrated / Interdisciplinary Robot Lab](https://introlab.3it.usherbrooke.ca)

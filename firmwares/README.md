# Firmwares

T-Top uses two microcontrollers (MCU). This folder contains the firmwares for those.

## Folder Structure

- THe [common](common) folder contains common libraries.
- The [dynamixel_control](dynamixel_control) folder contains the firmware of the MCU that controls the motors.
- The [dynamixel_setup](dynamixel_setup)folder contains the firmware to setup the motors.
- The [psu_control](psu_control) folder contains the firmware of the MCU that controls and monitors the battery, audio
  amplifier, fans and LEDs.
- The [ego_noise_opencr_firmware](ego_noise_opencr_firmware) folder contains the firmware to get the ego noise data.


## Credits

The class `TrustRegionReflectiveSolver` in the [opencr_firmware](opencr_firmware) folder is heavily inspired
by [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
.

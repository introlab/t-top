# Firmwares

T-Top uses two microcontrollers (MCU). This folder contains the firmwares for those.

## Folder Structure

- The [opencr_firmware](opencr_firmware) folder contains the firmware of the MCU that controls the motors. The MCU is
  programmed with the Arduino IDE (see
  the [instructions](../documentation/assembly/11_MCU_CONFIGURATION.md#d-setup-the-opencr)).
- The [psu_control](psu_control) folder contains the firmware of the MCU that controls and monitors the battery, audio
  amplifier, fans and LEDs. The MCU is programmed with [PlatformIO](https://platformio.org/platformio-ide) (see
  the [instructions](../documentation/assembly/11_MCU_CONFIGURATION.md#c-setup-the-teensy-lc)).

## Credits

The class `TrustRegionReflectiveSolver` in the [opencr_firmware](opencr_firmware) folder is heavily inspired
by [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
.

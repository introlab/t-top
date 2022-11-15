#ifndef PSU_CONTROL_CONFIG_H
#define PSU_CONTROL_CONFIG_H

#include <cstdint>
#include <cstddef>

// Firmware configuration
#define FIRMWARE_MODE_SETUP_BATTERY_CHARGER 1
#define FIRMWARE_MODE_NORMAL                2
#define FIRMWARE_MODE_PCB_TEST              3
#define FIRMWARE_MODE_COMMUNICATION_TEST    4
#define FIRMWARE_MODE                       FIRMWARE_MODE_NORMAL

// Device configuration
#define DEBUG_SERIAL Serial
constexpr long DEBUG_SERIAL_BAUD_RATE = 250000;

constexpr uint32_t WIRE_CLOCK = 100000;

constexpr uint32_t PWM_RESOLUTION = 12;
constexpr float PWM_MAX_VALUE = 4095;

constexpr uint32_t ADC_RESOLUTION = 10;
constexpr float ADC_MAX_VALUE = 1023;

constexpr uint8_t POWER_OFF_PIN = 2;
constexpr uint8_t POWER_SWITCH_PIN = 4;
constexpr uint32_t SHUTDOWN_REQUEST_TIMEOUT_MS = 120000;
constexpr float SHUTDOWN_COMPLETED_FOR_COMPUTER_AND_DYNAMIXELS_POWER_THRESHOLD_W = 5;

#define BATTERY_WIRE Wire
constexpr uint8_t BATTERY_STATUS_PIN = 20;
constexpr uint8_t CHARGER_STATUS_PIN = 21;
constexpr float BATTERY_CHARGER_CHARGE_CURRENT_LIMIT = 1.0;
constexpr float BATTERY_CHARGER_INPUT_CURRENT_LIMIT = 9.47;

constexpr uint8_t LED_STRIP_PIN = 1;
constexpr size_t STATE_OF_CHARGE_LED_COUNT = 5;
constexpr size_t VOLUME_LED_COUNT = 5;
constexpr size_t BASE_LED_COUNT = 31;
constexpr size_t LED_COUNT = STATE_OF_CHARGE_LED_COUNT + VOLUME_LED_COUNT + BASE_LED_COUNT;
constexpr uint8_t LED_STRIP_MINIMUM_BRIGHTNESS = 32;
constexpr uint8_t LED_STRIP_MAXIMUM_BRIGHTNESS = 128;

constexpr uint8_t FAN_PIN = 3;
constexpr float FAN_PWM_FREQUENCY = 46875;
constexpr float FAN_HYSTERESIS = 1;
constexpr float FAN_TEMPERATURE_STEP_1 = 40;
constexpr float FAN_TEMPERATURE_STEP_2 = 60;

constexpr uint8_t BUZZER_PIN = 6;
constexpr float BUZZER_PWM_FREQUENCY = 2400;
constexpr uint32_t BUZZER_ON_OFF_INTERVAL_US = 1000000;
constexpr float BUZZER_STATE_OF_CHARGE_LIMIT = 5;

#define AUDIO_POWER_AMPLIFIER_WIRE Wire1
constexpr size_t AUDIO_POWER_AMPLIFIER_COUNT = 2;
constexpr uint8_t AUDIO_POWER_AMPLIFIER_I2C_ADDRESSES[AUDIO_POWER_AMPLIFIER_COUNT] = {0x4a, 0x4b};
constexpr uint8_t AUDIO_POWER_AMPLIFIER_DEFAULT_VOLUME = 24;
constexpr uint8_t AUDIO_POWER_AMPLIFIER_BATTERY_MAXIMUM_VOLUME = 45;
constexpr uint8_t AUDIO_POWER_AMPLIFIER_MAXIMUM_VOLUME = 63;

#define CURRENT_VOLTAGE_SENSOR_WIRE Wire1
constexpr uint8_t CURRENT_VOLTAGE_SENSOR_ADDRESS = 0x40;
constexpr float CURRENT_VOLTAGE_SENSOR_SHUNT_RESISTOR = 0.004;
constexpr uint8_t CURRENT_VOLTAGE_SENSOR_MAX_CURRENT = 10;

constexpr uint8_t ONBOARD_TEMPERATURE_PIN = 14;
constexpr float ONBOARD_TEMPERATURE_NTC_R = 10000;  // @ 25°C
constexpr float ONBOARD_TEMPERATURE_NTC_BETA = 3940;
constexpr float ONBOARD_TEMPERATURE_R = 10000;
constexpr uint8_t EXTERNAL_TEMPERATURE_PIN = 15;
constexpr float EXTERNAL_TEMPERATURE_NTC_R = 10000;  // @ 25°C
constexpr float EXTERNAL_TEMPERATURE_NTC_BETA = 3936;
constexpr float EXTERNAL_TEMPERATURE_R = 10000;

constexpr uint8_t FRONT_LIGHT_SENSOR_PIN = 16;
constexpr uint8_t BACK_LIGHT_SENSOR_PIN = 25;
constexpr uint8_t LEFT_LIGHT_SENSOR_PIN = 17;
constexpr uint8_t RIGHT_LIGHT_SENSOR_PIN = 24;
constexpr float LIGHT_SENSOR_MINIMUM_VALUE = 0;
constexpr float LIGHT_SENSOR_MAXIMUM_VALUE = ADC_MAX_VALUE;

constexpr uint8_t START_BUTTON_PIN = 9;
constexpr uint8_t STOP_BUTTON_PIN = 10;
constexpr uint8_t VOLUME_UP_BUTTON_PIN = 12;
constexpr uint8_t VOLUME_DOWN_BUTTON_PIN = 11;
constexpr uint32_t VOLUME_BUTTON_INTERVAL_MS = 250;

// Timing configuration
constexpr uint32_t ERROR_DELAY_MS = 1000;
constexpr uint32_t STATUS_TICKER_INTERVAL_MS = 250;
constexpr uint32_t BUTTON_TICKER_INTERVAL_MS = 100;

#endif

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>


#define FIRMWARE_MODE_SETUP_BATTERY_CHARGER 1
#define FIRMWARE_MODE_NORMAL                2
#define FIRMWARE_MODE_TEST                  3
#define FIRMWARE_MODE                       FIRMWARE_MODE_NORMAL

#define DEBUG_SERIAL Serial
constexpr long DEBUG_SERIAL_BAUD_RATE = 9600;

#define OPENCR_SERIAL Serial1
constexpr long OPENCR_SERIAL_BAUD_RATE = 9600;

constexpr uint32_t WIRE_CLOCK = 100000;

constexpr uint32_t PWM_RESOLUTION = 12;
constexpr float PWM_MAX_VALUE = 4095;

constexpr uint32_t ADC_RESOLUTION = 10;
constexpr float ADC_MAX_VALUE = 1023;

#define BATTERY_WIRE Wire
constexpr uint8_t BATTERY_STATUS_PIN = 7;
constexpr uint8_t CHARGER_STATUS_PIN = 8;
constexpr float BATTERY_CHARGER_CHARGE_CURRENT_LIMIT = 1.0;
constexpr float BATTERY_CHARGER_INPUT_CURRENT_LIMIT = 7.5;

constexpr uint8_t BATTERY_25_LED_PIN = 9;
constexpr uint8_t BATTERY_50_LED_PIN = 6;
constexpr uint8_t BATTERY_75_LED_PIN = 10;
constexpr uint8_t BATTERY_100_LED_PIN = 20;
constexpr float BATTERY_LED_PWM_FREQUENCY = 732.4218;

constexpr uint8_t BUZZER_PIN = 16;
constexpr float BUZZER_PWM_FREQUENCY = 2400;
constexpr uint32_t BUZZER_ON_OFF_INTERVAL_US = 1000000;
constexpr float BUZZER_STATE_OF_CHARGE_LIMIT = 5;

constexpr uint8_t ONBOARD_TEMPERATURE_ADC_CHANNEL = 0;
constexpr float ONBOARD_TEMPERATURE_NTC_R = 10000;  // @ 25°C
constexpr float ONBOARD_TEMPERATURE_NTC_BETA = 3940;
constexpr float ONBOARD_TEMPERATURE_R = 10000;
constexpr uint8_t EXTERNAL_TEMPERATURE_ADC_CHANNEL = 1;
constexpr float EXTERNAL_TEMPERATURE_NTC_R = 10000;  // @ 25°C
constexpr float EXTERNAL_TEMPERATURE_NTC_BETA = 3936;
constexpr float EXTERNAL_TEMPERATURE_R = 10000;

constexpr uint8_t FAN_PIN = 3;
constexpr float FAN_PWM_FREQUENCY = 46875;
constexpr float FAN_HYSTERESIS = 1;
constexpr float FAN_TEMPERATURE_STEP_1 = 40;
constexpr float FAN_TEMPERATURE_STEP_2 = 60;

#define AUDIO_POWER_AMPLIFIER_WIRE Wire1
constexpr int AUDIO_POWER_AMPLIFIER_COUNT = 2;
constexpr int AUDIO_POWER_AMPLIFIER_I2C_ADDRESSES[AUDIO_POWER_AMPLIFIER_COUNT] = {0x4a, 0x4b};

#define INA220_TYPE 1
#define INA226_TYPE 2
#define INA_TYPE    INA220_TYPE
constexpr uint8_t INA_ADDRESS = 0x40;
constexpr float INA_SHUNT_RESISTOR = 0.004;
constexpr float INA_MAX_CURRENT = 10;
#define CURRENT_VOLTAGE_SENSOR_WIRE Wire1

constexpr uint32_t STATUS_TICKER_INTERVAL_MS = 1000;

#endif

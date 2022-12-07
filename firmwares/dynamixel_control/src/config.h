#ifndef DYNAMIXEL_CONTROL_CONFIG_H
#define DYNAMIXEL_CONTROL_CONFIG_H

#include "sensors/Icm40627.h"

#include <cstdint>
#include <cstddef>

// Firmware configuration
#define FIRMWARE_MODE_NORMAL             1
#define FIRMWARE_MODE_PCB_TEST           2
#define FIRMWARE_MODE_COMMUNICATION_TEST 3
#define FIRMWARE_MODE                    FIRMWARE_MODE_NORMAL
// TODO add dynamixel setup mode

// Device configuration
#define DEBUG_SERIAL Serial1
constexpr long DEBUG_SERIAL_BAUD_RATE = 250000;

constexpr uint32_t WIRE_CLOCK = 1000000;

#define IMU_WIRE Wire;
constexpr uint8_t IMU_ADDRESS = 0b1101000;
constexpr uint8_t IMU_INT1_PIN = 14;
constexpr uint8_t IMU_INT2_PIN = 15;
constexpr Icm40627::AccelerometerRange IMU_ACCELEROMETER_RANGE = Icm40627::AccelerometerRange::RANGE_2G;
constexpr Icm40627::GyroscopeRange IMU_GYROSCOPE_RANGE = Icm40627::GyroscopeRange::RANGE_250_DPS;
constexpr Icm40627::Odr IMU_ODR = Icm40627::Odr::ODR_100_HZ;
constexpr Icm40627::AntiAliasFilterBandwidth IMU_ANTI_ALIAS_FILTER_BANDWIDTH = Icm40627::AntiAliasFilterBandwidth::BANDWIDTH_42_HZ;

#define DYNAMIXEL_SERIAL Serial2
constexpr uint8_t DYNAMIXEL_ENABLE_PIN = 2;
constexpr uint8_t DYNAMIXEL_DIR_PIN = 9;

constexpr uint8_t LIMIT_SWITCH_PIN = 22;

// Communication configuration
#define COMPUTER_COMMUNICATION_SERIAL    Serial3
#define PSU_CONTROL_COMMUNICATION_SERIAL Serial5
constexpr uint8_t PSU_CONTROL_COMMUNICATION_RS232_INVALID = 23;
constexpr long COMMUNICATION_SERIAL_BAUD_RATE = 250000;
constexpr uint32_t COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS = 20;
constexpr uint32_t COMMUNICATION_MAXIMUM_TRIAL_COUNT = 5;

// Timing configuration
constexpr uint32_t ERROR_DELAY_MS = 1000;
constexpr uint32_t MOTOR_STATUS_TICKER_INTERVAL_MS = 10;

#endif

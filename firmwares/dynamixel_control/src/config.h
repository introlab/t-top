#ifndef DYNAMIXEL_CONTROL_CONFIG_H
#define DYNAMIXEL_CONTROL_CONFIG_H

#include <cstdint>
#include <cstddef>

// Firmware configuration
#define FIRMWARE_MODE_NORMAL             1
#define FIRMWARE_MODE_PCB_TEST           2
#define FIRMWARE_MODE_COMMUNICATION_TEST 3
#define FIRMWARE_MODE                    FIRMWARE_MODE_NORMAL

// Device configuration
#define DEBUG_SERIAL Serial1
constexpr long DEBUG_SERIAL_BAUD_RATE = 250000;

constexpr uint32_t WIRE_CLOCK = 100000;

#define IMU_WIRE Wire;
constexpr uint8_t IMU_INT1_PIN = 14;
constexpr uint8_t IMU_INT2_PIN = 15;

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
constexpr uint32_t MOTOR_STATUS_TICKER_INTERVAL_MS = 250;  // TODO change
constexpr uint32_t IMU_TICKER_INTERVAL_MS = 100;  // TODO change

#endif

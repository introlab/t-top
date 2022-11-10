#ifndef IO_MAPPING_H
#define IO_MAPPING_H

#include <cstdint>
#include <cstddef>

constexpr size_t STEWART_SERVO_COUNT = 6;
const uint8_t STEWART_PLATFORM_DYNAMIXEL_IDS[STEWART_SERVO_COUNT] = {1, 2, 3, 4, 5, 6};

const uint8_t TORSO_DYNAMIXEL_ID = 7;
const int TORSO_LIMIT_SWITCH_PIN = 2;

#endif

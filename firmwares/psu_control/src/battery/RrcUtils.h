#ifndef PSU_CONTROL_BATTERY_RRC_UTILS_H
#define PSU_CONTROL_BATTERY_RRC_UTILS_H

#include <Arduino.h>

constexpr size_t RRC_DEFAULT_TRIAL_COUNT = 5;

union RrcWordUnion
{
    uint16_t word;
    uint8_t bytes[2];
};

void initRandom();
void smBusRandomDelay();

#endif

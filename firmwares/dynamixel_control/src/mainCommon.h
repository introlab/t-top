#ifndef PSU_CONTROL_MAIN_COMMON_H
#define PSU_CONTROL_MAIN_COMMON_H

#include "config.h"

#include <Arduino.h>
#include <Wire.h>

#define CRITICAL_ERROR(message)                                                                                        \
    while (true)                                                                                                       \
    {                                                                                                                  \
        DEBUG_SERIAL.println((message));                                                                               \
        delay(ERROR_DELAY_MS);                                                                                         \
    }

extern Icm40627 imu;

void setupDebugSerial();
void setupWire();
void setupImu();
void setupImu(void (*dataReadyInterrupt)());

#endif

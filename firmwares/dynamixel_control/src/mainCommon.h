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

void setupDebugSerial();
void setupWire();

#endif

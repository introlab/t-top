#ifndef MCU_ERROR_H
#define MCU_ERROR_H

#include "config.h"

#include <cstring>

#define __FUNCTION_NAME__ __PRETTY_FUNCTION__
#define __FILENAME__      (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

#define CRITICAL_ERROR(reason)                                                                                         \
    while (true)                                                                                                       \
    {                                                                                                                  \
        DEBUG_SERIAL.println((reason));                                                                                \
        DEBUG_SERIAL.print("Function name: ");                                                                         \
        DEBUG_SERIAL.println(__FUNCTION_NAME__);                                                                       \
        DEBUG_SERIAL.print("Filename: ");                                                                              \
        DEBUG_SERIAL.println(__FILENAME__);                                                                            \
        pinMode(ERROR_LED_PIN, OUTPUT);                                                                                \
        digitalWrite(ERROR_LED_PIN, true);                                                                             \
        delay(ERROR_DELAY_MS);                                                                                         \
    }

#endif

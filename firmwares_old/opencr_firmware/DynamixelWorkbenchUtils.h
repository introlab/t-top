#ifndef DYNAMIXEL_WORKBENCH_UTILS_H
#define DYNAMIXEL_WORKBENCH_UTILS_H

#include <DynamixelWorkbench.h>

inline bool readPresentVelocityData(DynamixelWorkbench& dynamixelWorkbench, uint8_t id, int32_t* value)
{
    constexpr uint16_t ADDRESS = 128;
    constexpr uint16_t LENGTH = 4;

    return dynamixelWorkbench.readRegister(id, ADDRESS, LENGTH, reinterpret_cast<uint32_t*>(value));
}

#endif

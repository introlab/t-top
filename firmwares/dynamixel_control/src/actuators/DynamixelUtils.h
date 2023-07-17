#ifndef DYNAMIXEL_CONTROL_ACTUATORS_DYNAMIXEL_UTILS_H
#define DYNAMIXEL_CONTROL_ACTUATORS_DYNAMIXEL_UTILS_H

#include <Dynamixel2Arduino.h>

constexpr float PI_FLOAT = 3.1415926535897932384626433832795f;

inline float radToDeg(float rad)
{
    return rad * 180.f / PI_FLOAT;
}

inline float degToRad(float deg)
{
    return deg * PI_FLOAT / 180.f;
}

inline bool dynamixelSetNormalDirectionIfNeeded(Dynamixel2Arduino& dynamixel, uint8_t id)
{
    constexpr uint8_t NORMAL_DIRECTION_VALUE = 0b00000000;
    if (dynamixel.readControlTableItem(ControlTableItem::DRIVE_MODE, id) == NORMAL_DIRECTION_VALUE)
    {
        return true;
    }
    return dynamixel.writeControlTableItem(ControlTableItem::DRIVE_MODE, id, NORMAL_DIRECTION_VALUE);
}

inline bool dynamixelSetReverseDirectionIfNeeded(Dynamixel2Arduino& dynamixel, uint8_t id)
{
    constexpr uint8_t REVERSE_DIRECTION_VALUE = 0b00000001;
    if (dynamixel.readControlTableItem(ControlTableItem::DRIVE_MODE, id) == REVERSE_DIRECTION_VALUE)
    {
        return true;
    }
    return dynamixel.writeControlTableItem(ControlTableItem::DRIVE_MODE, id, REVERSE_DIRECTION_VALUE);
}

#endif

#ifndef SERVO_SPEED_CONTROLLER_H
#define SERVO_SPEED_CONTROLLER_H

#include <DynamixelWorkbench.h>

class ServoSpeedController
{
    DynamixelWorkbench& m_dynamixelWorkbench;
    uint8_t m_id;

public:
    ServoSpeedController(DynamixelWorkbench& dynamixelWorkbench, uint8_t id);
    ~ServoSpeedController();

    void setSpeed(int32_t speed);
    int32_t readSpeed();
    float readPosition();
};

#endif

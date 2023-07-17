#ifndef DYNAMIXEL_CONTROL_ACTUATORS_TORSO_CONTROLLER_H
#define DYNAMIXEL_CONTROL_ACTUATORS_TORSO_CONTROLLER_H

#include <Dynamixel2Arduino.h>

constexpr float TORSO_ORIENTATION_OFFSET = 0.0;
constexpr float TORSO_GEAR_RATIO = 46.0 / 130.0;
constexpr int32_t TORSO_MAX_VELOCITY = 80;  // unit : 0.229 rev/min

float fmodRadian(float v);

class TorsoController
{
    Dynamixel2Arduino& m_dynamixel;
    float m_zeroOffset;

public:
    TorsoController(Dynamixel2Arduino& dynamixel);
    ~TorsoController();

    void begin();

    void setOrientation(float orientation);
    float readOrientation();
    int16_t readServoSpeed();

private:
    void setMaxVelocityIfNeeded();
    void findZeroOffset();
    float getOrientationFromDynamixelPosition(float dynamixelPosition);
    float getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta);
};

#endif

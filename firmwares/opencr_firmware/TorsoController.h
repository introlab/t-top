#ifndef TORSO_CONTROLLER_H
#define TORSO_CONTROLLER_H

#include <DynamixelWorkbench.h>

constexpr float TORSO_ORIENTATION_OFFSET = 0.0;
constexpr float TORSO_GEAR_RATIO = 46.0 / 130.0;
constexpr int32_t TORSO_MAX_VELOCITY = 80;  // unit : 0.229 rev/min

float fmodRadian(float v);

class TorsoController
{
    DynamixelWorkbench& m_dynamixelWorkbench;

    bool m_isZeroOffsetFound;
    float m_zeroOffset;

public:
    TorsoController(DynamixelWorkbench& dynamixelWorkbench);
    ~TorsoController();

    void init();

    void setOrientation(float orientation);
    float readOrientation();
    int32_t readServoSpeed();

private:
    void setMaxVelocityIfNeeded();
    void findZeroOffset();
    float getOrientationFromDynamixelPosition(float dynamixelPosition);
    float getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta);
};

#endif

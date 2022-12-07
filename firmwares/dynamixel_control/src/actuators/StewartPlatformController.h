#ifndef DYNAMIXEL_CONTROL_ACTUATORS_STEWART_PLATFORM_CONTROLLER_H
#define DYNAMIXEL_CONTROL_ACTUATORS_STEWART_PLATFORM_CONTROLLER_H

#include "StewartForwardKinematics.h"
#include "StewartInverseKinematics.h"

#include <Dynamixel2Arduino.h>


struct HeadPose
{
    float positionX;
    float positionY;
    float positionZ;
    float orientationW;
    float orientationX;
    float orientationY;
    float orientationZ;
};

class StewartPlatformController
{
    Dynamixel2Arduino& m_dynamixel;

    StewartForwardKinematics m_forwardKinematics;
    StewartInverseKinematics m_inverseKinematics;

public:
    StewartPlatformController(Dynamixel2Arduino& dynamixel);
    ~StewartPlatformController();

    void begin();

    void setPose(const HeadPose& pose);

    bool isPoseReachable();
    void readCurrentPose(float servoAngles[], HeadPose& pose);
    void readServoSpeeds(int16_t servoSpeeds[]);
};

#endif

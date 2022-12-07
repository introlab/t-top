#include "StewartPlatformController.h"
#include "DynamixelUtils.h"


#define POW_2(x) ((x) * (x))

StewartPlatformController::StewartPlatformController(Dynamixel2Arduino& dynamixel)
    : m_dynamixel(dynamixel)
{
}

StewartPlatformController::~StewartPlatformController() {}

void StewartPlatformController::begin()
{
    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixel.ping(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

        m_dynamixel.torqueOff(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
        dynamixelSetNormalDirectionIfNeeded(m_dynamixel, STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
        m_dynamixel.torqueOn(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

        m_dynamixel.setOperatingMode(STEWART_PLATFORM_DYNAMIXEL_IDS[i], OP_POSITION);
    }

    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixel.setGoalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], 0.f, UNIT_DEGREE);
    }
}

void StewartPlatformController::setPose(const HeadPose& pose)
{
    Eigen::Vector3f position(pose.positionX, pose.positionY, pose.positionZ);
    Eigen::Quaternionf orientation(
        pose.orientationW,
        pose.orientationX,
        pose.orientationY,
        pose.orientationZ);

    Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> servoAngles;
    m_inverseKinematics.calculateServoAngles(servoAngles, position, orientation);

    if (!m_inverseKinematics.isPoseReachable())
    {
        return;
    }

    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixel.setGoalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], radToDeg(servoAngles[i]), UNIT_DEGREE);
    }
}

bool StewartPlatformController::isPoseReachable()
{
    return m_inverseKinematics.isPoseReachable();
}

void StewartPlatformController::readCurrentPose(float servoAngles[], HeadPose& pose)
{
    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        servoAngles[i] = degToRad(m_dynamixel.getPresentPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], UNIT_DEGREE));
    }

    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
    Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> servoAnglesVec(servoAngles);
    m_forwardKinematics.calculatePose(position, orientation, servoAnglesVec);

    pose.positionX = position.x();
    pose.positionY = position.y();
    pose.positionZ = position.z();
    pose.orientationW = orientation.w();
    pose.orientationX = orientation.x();
    pose.orientationY = orientation.y();
    pose.orientationZ = orientation.z();
}

void StewartPlatformController::readServoSpeeds(int16_t servoSpeeds[])
{
    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        servoSpeeds[i] = static_cast<int16_t>(m_dynamixel.getPresentVelocity(STEWART_PLATFORM_DYNAMIXEL_IDS[i], UNIT_RAW));
    }
}

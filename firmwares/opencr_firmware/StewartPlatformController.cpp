#include "StewartPlatformController.h"
#include "DynamixelWorkbenchUtils.h"

#include <Eigen/Geometry>
#include <cstring>
#include <cmath>

#define POW_2(x) ((x) * (x))

constexpr int MOTOR_INIT_DELAY_MS = 250;

StewartPlatformController::StewartPlatformController(DynamixelWorkbench& dynamixelWorkbench)
    : m_dynamixelWorkbench(dynamixelWorkbench),
      m_isValidFrame(true)
{
}

StewartPlatformController::~StewartPlatformController() {}

void StewartPlatformController::init()
{
    for (int i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixelWorkbench.ping(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

        m_dynamixelWorkbench.torqueOff(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
        m_dynamixelWorkbench.setNormalDirection(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
        m_dynamixelWorkbench.torqueOn(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

        m_dynamixelWorkbench.jointMode(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
        delay(MOTOR_INIT_DELAY_MS);
    }

    for (int i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixelWorkbench.goalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], 0.f);
        delay(MOTOR_INIT_DELAY_MS);
    }
}

void StewartPlatformController::setPose(const geometry_msgs::PoseStamped& pose)
{
    if (std::strcmp(pose.header.frame_id, "stewart_base") != 0)
    {
        m_isValidFrame = false;
        return;
    }
    m_isValidFrame = true;

    Eigen::Vector3f position(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
    Eigen::Quaternionf orientation(
        pose.pose.orientation.w,
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z);

    Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> servoAngles;
    m_inverseKinematics.calculateServoAngles(servoAngles, position, orientation);

    if (!m_inverseKinematics.isPoseReachable())
    {
        return;
    }

    for (int i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixelWorkbench.goalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], servoAngles[i]);
    }
}

bool StewartPlatformController::isPoseReachable()
{
    return m_isValidFrame && m_inverseKinematics.isPoseReachable();
}

void StewartPlatformController::readCurrentPose(float servoAngles[], geometry_msgs::PoseStamped& pose)
{
    for (int i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        m_dynamixelWorkbench.getRadian(STEWART_PLATFORM_DYNAMIXEL_IDS[i], servoAngles + i);
    }

    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
    Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> servoAnglesVec(servoAngles);
    m_forwardKinematics.calculatePose(position, orientation, servoAnglesVec);

    pose.header.frame_id = "stewart_base";
    pose.pose.position.x = position.x();
    pose.pose.position.y = position.y();
    pose.pose.position.z = position.z();
    pose.pose.orientation.w = orientation.w();
    pose.pose.orientation.x = orientation.x();
    pose.pose.orientation.y = orientation.y();
    pose.pose.orientation.z = orientation.z();
}

void StewartPlatformController::readServoSpeeds(int32_t servoSpeeds[])
{
    for (int i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        readPresentVelocityData(m_dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[i], servoSpeeds + i);
    }
}

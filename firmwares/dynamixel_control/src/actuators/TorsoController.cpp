#include "TorsoController.h"
#include "DynamixelUtils.h"
#include "../config.h"

#include <cmath>

constexpr float MIN_DYNAMIXEL_POSITION = std::ceil(-255 * 2 * M_PI);
constexpr float MAX_DYNAMIXEL_POSITION = std::floor(255 * 2 * M_PI);

float fmodRadian(float v)
{
    return std::fmod(std::fmod(v, 2 * M_PI) + 2 * M_PI, 2 * M_PI);
}

TorsoController::TorsoController(Dynamixel2Arduino& dynamixel)
    : m_dynamixel(dynamixel),
      m_zeroOffset(0.f)
{
}

TorsoController::~TorsoController() {}

void TorsoController::begin()
{
    m_dynamixel.ping(TORSO_DYNAMIXEL_ID);
    m_dynamixel.torqueOff(TORSO_DYNAMIXEL_ID);
    dynamixelSetNormalDirectionIfNeeded(m_dynamixel, TORSO_DYNAMIXEL_ID);
    m_dynamixel.setOperatingMode(TORSO_DYNAMIXEL_ID, OP_EXTENDED_POSITION);
    setMaxVelocityIfNeeded();
    m_dynamixel.torqueOn(TORSO_DYNAMIXEL_ID);

    findZeroOffset();
}

void TorsoController::setOrientation(float orientation)
{
    float dynamixelPosition = degToRad(m_dynamixel.getPresentPosition(TORSO_DYNAMIXEL_ID, UNIT_DEGREE));
    float currentOrientation = getOrientationFromDynamixelPosition(dynamixelPosition);
    float orientationDelta = fmodRadian(orientation) - currentOrientation;


    float newDynamixelPosition = getNewDynamixelPositionFromOrientationDelta(dynamixelPosition, orientationDelta);
    m_dynamixel.setGoalPosition(TORSO_DYNAMIXEL_ID, radToDeg(newDynamixelPosition), UNIT_DEGREE);
}

float TorsoController::readOrientation()
{
    float dynamixelPosition = degToRad(m_dynamixel.getPresentPosition(TORSO_DYNAMIXEL_ID, UNIT_DEGREE));
    return getOrientationFromDynamixelPosition(dynamixelPosition);
}

int16_t TorsoController::readServoSpeed()
{
    return static_cast<int16_t>(m_dynamixel.getPresentVelocity(TORSO_DYNAMIXEL_ID, UNIT_RAW));
}

void TorsoController::setMaxVelocityIfNeeded()
{
    if (m_dynamixel.readControlTableItem(ControlTableItem::PROFILE_VELOCITY, TORSO_DYNAMIXEL_ID) != TORSO_MAX_VELOCITY)
    {
        m_dynamixel.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, TORSO_DYNAMIXEL_ID, TORSO_MAX_VELOCITY);
    }
}

void TorsoController::findZeroOffset()
{
    pinMode(TORSO_LIMIT_SWITCH_PIN, INPUT);

    float goalPosition = 2.5 * M_PI / TORSO_GEAR_RATIO;
    float dynamixelPosition = 0.f;
    bool isZeroOffsetFound = false;

    m_dynamixel.setGoalPosition(TORSO_DYNAMIXEL_ID, radToDeg(goalPosition), UNIT_DEGREE);
    do
    {
        dynamixelPosition = degToRad(m_dynamixel.getPresentPosition(TORSO_DYNAMIXEL_ID, UNIT_DEGREE));
        isZeroOffsetFound = !digitalRead(TORSO_LIMIT_SWITCH_PIN);

    } while (std::abs(dynamixelPosition - goalPosition) > 0.01 && !isZeroOffsetFound);

    if (isZeroOffsetFound)
    {
        m_zeroOffset = dynamixelPosition + TORSO_ORIENTATION_OFFSET / TORSO_GEAR_RATIO;
    }
    else
    {
        m_zeroOffset = 0.f;
    }
    m_dynamixel.setGoalPosition(TORSO_DYNAMIXEL_ID, radToDeg(m_zeroOffset), UNIT_DEGREE);
}

float TorsoController::getOrientationFromDynamixelPosition(float dynamixelPosition)
{
    return fmodRadian((dynamixelPosition - m_zeroOffset) * TORSO_GEAR_RATIO);
}

float TorsoController::getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta1)
{
    float orientationDelta2 = 0.0;
    if (orientationDelta1 < 0.0)
    {
        orientationDelta2 = (2 * M_PI + orientationDelta1);
    }
    else
    {
        orientationDelta2 = -(2 * M_PI - orientationDelta1);
    }

    float newDynamixelPosition1 = dynamixelPosition + orientationDelta1 / TORSO_GEAR_RATIO;
    float newDynamixelPosition2 = dynamixelPosition + orientationDelta2 / TORSO_GEAR_RATIO;

    if (newDynamixelPosition1 > MAX_DYNAMIXEL_POSITION || newDynamixelPosition1 < MIN_DYNAMIXEL_POSITION)
    {
        return newDynamixelPosition2;
    }
    else if (newDynamixelPosition2 > MAX_DYNAMIXEL_POSITION || newDynamixelPosition2 < MIN_DYNAMIXEL_POSITION)
    {
        return newDynamixelPosition1;
    }
    else if (std::abs(newDynamixelPosition1 - dynamixelPosition) < std::abs(newDynamixelPosition2 - dynamixelPosition))
    {
        return newDynamixelPosition1;
    }
    else
    {
        return newDynamixelPosition2;
    }
}

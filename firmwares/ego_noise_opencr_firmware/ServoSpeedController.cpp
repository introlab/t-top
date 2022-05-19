#include "ServoSpeedController.h"
#include "DynamixelWorkbenchUtils.h"

ServoSpeedController::ServoSpeedController(DynamixelWorkbench& dynamixelWorkbench, uint8_t id)
    : m_dynamixelWorkbench(dynamixelWorkbench),
      m_id(id)
{
    m_dynamixelWorkbench.ping(m_id);
    m_dynamixelWorkbench.torqueOff(m_id);
    m_dynamixelWorkbench.setNormalDirection(m_id);
    m_dynamixelWorkbench.setVelocityControlMode(m_id);
    m_dynamixelWorkbench.torqueOn(m_id);
}

ServoSpeedController::~ServoSpeedController()
{
    setSpeed(0);
    m_dynamixelWorkbench.torqueOff(m_id);
}

void ServoSpeedController::setSpeed(int32_t speed)
{
    m_dynamixelWorkbench.goalVelocity(m_id, speed);
}

int32_t ServoSpeedController::readSpeed()
{
    int32_t speed = 0;
    readProfileVelocityData(m_dynamixelWorkbench, m_id, &speed);
    return speed;
}

float ServoSpeedController::readPosition()
{
    float position;
    m_dynamixelWorkbench.getRadian(m_id, &position);
    return position;
}

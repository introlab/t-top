#include "TorsoControllerDyna.h"
#include "IoMapping.h"

#include <cmath>

static const float ORIENTATION_OFFSET = 0.0;
static const float GEAR_RATIO = 46.0 / 130.0;

static const float MIN_DYNAMIXEL_POSITION = std::ceil(-255 * 2 * M_PI);
static const float MAX_DYNAMIXEL_POSITION = std::floor(255 * 2 * M_PI);

static const int32_t MAX_VELOCITY = 80; // unit : 0.229 rev/min


TorsoControllerDyna::TorsoControllerDyna(DynamixelWorkbench& dynamixelWorkbench) : 
  m_dynamixelWorkbench(dynamixelWorkbench),
  m_isZeroOffsetFound(false) {
}

TorsoControllerDyna::~TorsoControllerDyna() {
  
}

void TorsoControllerDyna::init() {
  m_dynamixelWorkbench.ping(TORSO_DYNAMIXEL_ID);
  m_dynamixelWorkbench.torqueOff(TORSO_DYNAMIXEL_ID);
  m_dynamixelWorkbench.setNormalDirection(TORSO_DYNAMIXEL_ID);
  m_dynamixelWorkbench.setExtendedPositionControlMode(TORSO_DYNAMIXEL_ID);
  m_dynamixelWorkbench.itemWrite(TORSO_DYNAMIXEL_ID, "Profile_Velocity", MAX_VELOCITY);
  m_dynamixelWorkbench.torqueOn(TORSO_DYNAMIXEL_ID);

  findZeroOffset();
}

void TorsoControllerDyna::setOrientation(float orientation) {  
  float dynamixelPosition = 0.f;
  m_dynamixelWorkbench.getRadian(TORSO_DYNAMIXEL_ID, &dynamixelPosition);
  float currentOrientation = getOrientationFromDynamixelPosition(dynamixelPosition);
  float orientationDelta = std::fmod(orientation, 2 * M_PI) - currentOrientation;


  float newDynamixelPosition = getNewDynamixelPositionFromOrientationDelta(dynamixelPosition, orientationDelta);
  m_dynamixelWorkbench.goalPosition(TORSO_DYNAMIXEL_ID, newDynamixelPosition);
}

float TorsoControllerDyna::readOrientation() {
  float dynamixelPosition = 0.f;
  m_dynamixelWorkbench.getRadian(TORSO_DYNAMIXEL_ID, &dynamixelPosition);
  return getOrientationFromDynamixelPosition(dynamixelPosition);
}

static bool* isZeroOffsetFound;

void onLimitSwitchInterrupt() {
  detachInterrupt(digitalPinToInterrupt(TORSO_LIMIT_SWITCH_PIN));
  *isZeroOffsetFound = true;
}

void TorsoControllerDyna::findZeroOffset() {
  isZeroOffsetFound = &m_isZeroOffsetFound;
  pinMode(TORSO_LIMIT_SWITCH_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TORSO_LIMIT_SWITCH_PIN), onLimitSwitchInterrupt, FALLING);
  
  float goalPosition = 2.5 * M_PI / GEAR_RATIO;
  float dynamixelPosition = 0.f;

  m_dynamixelWorkbench.goalPosition(TORSO_DYNAMIXEL_ID, goalPosition);
  do {
    m_dynamixelWorkbench.getRadian(TORSO_DYNAMIXEL_ID, &dynamixelPosition);
  } while (std::abs(dynamixelPosition - goalPosition) > 0.01 && !m_isZeroOffsetFound);

  if (m_isZeroOffsetFound) {
    m_zeroOffset = dynamixelPosition + ORIENTATION_OFFSET / GEAR_RATIO;
    m_dynamixelWorkbench.goalPosition(TORSO_DYNAMIXEL_ID, m_zeroOffset);
  }
  else {
    detachInterrupt(digitalPinToInterrupt(TORSO_LIMIT_SWITCH_PIN));
    m_zeroOffset = 0;
    m_dynamixelWorkbench.goalPosition(TORSO_DYNAMIXEL_ID, 0.f);
  }
}

float TorsoControllerDyna::getOrientationFromDynamixelPosition(float dynamixelPosition) {
  return std::fmod((dynamixelPosition - m_zeroOffset) * GEAR_RATIO, 2 * M_PI);
}

float TorsoControllerDyna::getNewDynamixelPositionFromOrientationDelta(float dynamixelPosition, float orientationDelta1) {
  float orientationDelta2 = 0.0;
  if (orientationDelta1 < 0.0) {
    orientationDelta2 = (2 * M_PI + orientationDelta1);
  } else {
    orientationDelta2 = -(2 * M_PI - orientationDelta1);
  }

  float newDynamixelPosition1 = dynamixelPosition + orientationDelta1 / GEAR_RATIO;
  float newDynamixelPosition2 = dynamixelPosition + orientationDelta2 / GEAR_RATIO;

  if (newDynamixelPosition1 > MAX_DYNAMIXEL_POSITION || newDynamixelPosition1 < MIN_DYNAMIXEL_POSITION) {
    return newDynamixelPosition2;
  }
  else if (newDynamixelPosition2 > MAX_DYNAMIXEL_POSITION || newDynamixelPosition2 < MIN_DYNAMIXEL_POSITION) {
    return newDynamixelPosition1;
  }
  else if (std::abs(newDynamixelPosition1 - dynamixelPosition) < std::abs(newDynamixelPosition2 - dynamixelPosition)) {
    return newDynamixelPosition1;
  }
  else {
    return newDynamixelPosition2;
  }
}

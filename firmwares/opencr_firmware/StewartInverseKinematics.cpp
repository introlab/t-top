#include "StewartInverseKinematics.h"
#include "StewartKinematicsParameters.h"

#include <cmath>

#define POW_2(x) ((x) * (x))

StewartInverseKinematics::StewartInverseKinematics() : m_isPoseReachable(true) {
}

void StewartInverseKinematics::calculateServoAngles(Eigen::Matrix<float, 6, 1>& servoAngles,
    const Eigen::Vector3f& position,
    const Eigen::Quaternionf& orientation) {
  for (int i = 0; i < STEWART_SERVO_COUNT; i++) {
    Eigen::Vector3f l = orientation * STEWART_TOP_ANCHORS[i] + position - STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[i];

    float ek = 2 * STEWART_HORN_LENGTH * l.z();
    float fk = 2 * STEWART_HORN_LENGTH * (std::cos(STEWART_HORN_ORIENTATION_ANGLES[i]) * l.x() +
        std::sin(STEWART_HORN_ORIENTATION_ANGLES[i]) * l.y());
    float gk = POW_2(l.x()) + POW_2(l.y()) + POW_2(l.z()) - POW_2(STEWART_ROD_LENGTH) + POW_2(STEWART_HORN_LENGTH);

    float asinParameter = gk / std::sqrt(POW_2(ek) + POW_2(fk));
    if (asinParameter < -1.f || asinParameter > 1.f) {
      m_isPoseReachable = false;
      return;
    }
    float servoAngle = std::asin(asinParameter) - std::atan2(fk, ek);
    if (STEWART_IS_HORN_ORIENTATION_REVERSED[i]) {
      servoAngle = -servoAngle;
    }

    if (servoAngle < STEWART_SERVO_ANGLE_MIN || servoAngle > STEWART_SERVO_ANGLE_MAX) {
      m_isPoseReachable = false;
      return;
    }

    servoAngles[i] = servoAngle;
  }

  m_isPoseReachable = true;
}

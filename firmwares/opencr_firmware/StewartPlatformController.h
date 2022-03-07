#ifndef STEWART_PLATFORM_CONTROLLER_H
#define STEWART_PLATFORM_CONTROLLER_H

#include <DynamixelWorkbench.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen.h>

#include "IoMapping.h"
#include "StewartForwardKinematics.h"
#include "StewartInverseKinematics.h"


class StewartPlatformController {
  DynamixelWorkbench& m_dynamixelWorkbench;

  bool m_isValidFrame;

  StewartForwardKinematics m_forwardKinematics;
  StewartInverseKinematics m_inverseKinematics;

public:
  StewartPlatformController(DynamixelWorkbench& dynamixelWorkbench);
  ~StewartPlatformController();

  void init();

  void setPose(const geometry_msgs::PoseStamped& pose);

  bool isPoseReachable();
  void readCurrentPose(float servoAngles[], geometry_msgs::PoseStamped& pose);
};

#endif

#ifndef STEWART_PLATFORM_CONTROLLER_DYNA_H
#define STEWART_PLATFORM_CONTROLLER_DYNA_H

#include <DynamixelWorkbench.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen.h>

#include "IoMapping.h"
#include "StewartForwardKinematicsNeuralNetwork.h"

class StewartPlatformControllerDyna {
  DynamixelWorkbench& m_dynamixelWorkbench;

  bool m_isPoseReachable;
  
  Eigen::Vector3f m_position;
  Eigen::Matrix3f m_orientation;

  float m_inverseKinematicsServoAngles[STEWART_PLATFORM_DYNAMIXEL_COUNT];

  StewartForwardKinematicsNeuralNetwork m_stewartForwardKinematicsNeuralNetwork;

public:
  StewartPlatformControllerDyna(DynamixelWorkbench& dynamixelWorkbench);
  ~StewartPlatformControllerDyna();

  void init();

  void setPose(const geometry_msgs::PoseStamped& pose);

  bool isPoseReachable();
  void readCurrentPose(float servoAngles[], geometry_msgs::PoseStamped& pose);

private:
  bool calculateInverseKinematicsServoAngles();
};

#endif

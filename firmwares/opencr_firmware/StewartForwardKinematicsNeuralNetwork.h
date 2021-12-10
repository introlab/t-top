#ifndef STEWART_FORWARD_KINEMATICS_NEURAL_NETWORK_H
#define STEWART_FORWARD_KINEMATICS_NEURAL_NETWORK_H

#include <geometry_msgs/PoseStamped.h>
#include <Eigen.h>

class StewartForwardKinematicsNeuralNetwork {
  Eigen::Matrix<float, 6, 1> m_input;
  
  Eigen::Matrix<float, 64, 1> m_position_h0;
  Eigen::Matrix<float, 32, 1> m_position_h1;
  Eigen::Matrix<float, 16, 1> m_position_h2;
  Eigen::Matrix<float, 3, 1> m_position_output;

  Eigen::Matrix<float, 64, 1> m_orientation_h0;
  Eigen::Matrix<float, 32, 1> m_orientation_h1;
  Eigen::Matrix<float, 16, 1> m_orientation_h2;
  Eigen::Matrix<float, 4, 1> m_orientation_output;

public:
  StewartForwardKinematicsNeuralNetwork();
  ~StewartForwardKinematicsNeuralNetwork();

  void calculateCurrentPose(float servoAngles[], geometry_msgs::PoseStamped& pose);
};

#endif

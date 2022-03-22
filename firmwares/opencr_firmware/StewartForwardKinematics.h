#ifndef STEWART_FORWARD_KINEMATICS_H
#define STEWART_FORWARD_KINEMATICS_H

#include "StewartKinematicsParameters.h"
#include "TrustRegionReflectiveSolver.h"

#include <Eigen.h>
#include <Eigen/Geometry>

constexpr int SOLVER_X_SIZE = 6;

class StewartForwardKinematics
{
    TrustRegionReflectiveSolver m_solver;
    Eigen::Vector3f m_bottomAnchors[STEWART_SERVO_COUNT];

public:
    StewartForwardKinematics();

    void calculatePose(
        Eigen::Vector3f& position,
        Eigen::Quaternionf& orientation,
        const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1>& servoAngles);

private:
    void calculateBottomAnchors(const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1>& servoAngles);

    void solverFunction(
        Eigen::Matrix<float, SOLVER_X_SIZE, 1>& f,
        Eigen::Matrix<float, SOLVER_X_SIZE, SOLVER_X_SIZE>& J,
        const Eigen::Matrix<float, SOLVER_X_SIZE, 1>& x);
    void calculateRodLengthErrorAndGrad(
        float& error,
        Eigen::Matrix<float, SOLVER_X_SIZE, 1>& grad,
        float bax,
        float bay,
        float baz,
        float itax,
        float itay,
        float itaz,
        float tx,
        float ty,
        float tz,
        float rx,
        float ry,
        float rz,
        float d);
};

#endif

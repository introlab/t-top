#ifndef STEWART_INVERSE_KINEMATICS_H
#define STEWART_INVERSE_KINEMATICS_H

#include <Eigen.h>
#include <Eigen/Geometry>

class StewartInverseKinematics
{
    bool m_isPoseReachable;

public:
    StewartInverseKinematics();

    bool isPoseReachable();
    void calculateServoAngles(
        Eigen::Matrix<float, 6, 1>& servoAngles,
        const Eigen::Vector3f& position,
        const Eigen::Quaternionf& orientation);
};

inline bool StewartInverseKinematics::isPoseReachable()
{
    return m_isPoseReachable;
}

#endif

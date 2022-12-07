#include "StewartForwardKinematics.h"

#include <cmath>

StewartForwardKinematics::StewartForwardKinematics() : m_solver(TrustRegionReflectiveParameters()) {}

void StewartForwardKinematics::calculatePose(
    Eigen::Vector3f& position,
    Eigen::Quaternionf& orientation,
    const Eigen::Matrix<float, 6, 1>& servoAngles)
{
    calculateBottomAnchors(servoAngles);

    auto func = [this](
                    Eigen::Matrix<float, SOLVER_X_SIZE, 1>& f,
                    Eigen::Matrix<float, SOLVER_X_SIZE, SOLVER_X_SIZE>& J,
                    const Eigen::Matrix<float, SOLVER_X_SIZE, 1>& x) { solverFunction(f, J, x); };

    TrustRegionReflectiveSolverResult<SOLVER_X_SIZE> result = m_solver.solve(
        func,
        STEWART_FORWARD_KINEMATICS_X0,
        STEWART_FORWARD_KINEMATICS_MIN_BOUNDS,
        STEWART_FORWARD_KINEMATICS_MAX_BOUNDS);

    position.x() = result.x[0];
    position.y() = result.x[1];
    position.z() = result.x[2];

    orientation = Eigen::AngleAxisf(result.x[3], Eigen::Vector3f::UnitX()) *
                  Eigen::AngleAxisf(result.x[4], Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(result.x[5], Eigen::Vector3f::UnitZ());
}

void StewartForwardKinematics::calculateBottomAnchors(const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1>& servoAngles)
{
    for (size_t i = 0; i < STEWART_SERVO_COUNT; i++)
    {
        float servoAngle = servoAngles[i];
        if (STEWART_IS_HORN_ORIENTATION_REVERSED[i])
        {
            servoAngle = -servoAngle;
        }

        float cosServoAngle = std::cos(servoAngle);
        m_bottomAnchors[i].x() = STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[i].x() +
                                 STEWART_HORN_LENGTH * cosServoAngle * std::cos(STEWART_HORN_ORIENTATION_ANGLES[i]);
        m_bottomAnchors[i].y() = STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[i].y() +
                                 STEWART_HORN_LENGTH * cosServoAngle * std::sin(STEWART_HORN_ORIENTATION_ANGLES[i]);
        m_bottomAnchors[i].z() =
            STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[i].z() + STEWART_HORN_LENGTH * std::sin(servoAngle);
    }
}

void StewartForwardKinematics::solverFunction(
    Eigen::Matrix<float, SOLVER_X_SIZE, 1>& f,
    Eigen::Matrix<float, SOLVER_X_SIZE, SOLVER_X_SIZE>& J,
    const Eigen::Matrix<float, SOLVER_X_SIZE, 1>& x)
{
    for (int i = 0; i < SOLVER_X_SIZE; i++)
    {
        float error;
        Eigen::Matrix<float, SOLVER_X_SIZE, 1> grad;
        calculateRodLengthErrorAndGrad(
            error,
            grad,
            m_bottomAnchors[i].x(),
            m_bottomAnchors[i].y(),
            m_bottomAnchors[i].z(),
            STEWART_TOP_ANCHORS[i].x(),
            STEWART_TOP_ANCHORS[i].y(),
            STEWART_TOP_ANCHORS[i].z(),
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            STEWART_ROD_LENGTH);

        f[i] = error;
        J.row(i) = grad.transpose();
    }
}

void StewartForwardKinematics::calculateRodLengthErrorAndGrad(
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
    float d)
{
    float sinRx = std::sin(rx);
    float cosRx = std::cos(rx);
    float sinRy = std::sin(ry);
    float cosRy = std::cos(ry);
    float sinRz = std::sin(rz);
    float cosRz = std::cos(rz);

    float cosRyCosRz = cosRy * cosRz;
    float sinRzCosRy = sinRz * cosRy;
    float sin_rx_sin_ry_cos_rz = sinRx * sinRy * cosRz;
    float sinRzCosRx = sinRz * cosRx;
    float sinRxSinRySinRz = sinRx * sinRy * sinRz;
    float cosRxCosRz = cosRx * cosRz;
    float sinRxCosRy = sinRx * cosRy;
    float sinRxSinRz = sinRx * sinRz;
    float sinRxCosRyCosRz = sinRx * cosRyCosRz;
    float sinRyCosRxCosRz = sinRy * cosRxCosRz;
    float sinRxCosRz = sinRx * cosRz;
    float sinRxSinRzCosRy = sinRx * sinRzCosRy;
    float sinRySinRzCosRx = sinRy * sinRzCosRx;
    float cosRxCosRy = cosRx * cosRy;
    float sinRzCosRxCosRy = sinRzCosRx * cosRy;
    float cosRxCosRyCosRz = cosRxCosRy * cosRz;
    float sinRyCosRz = sinRy * cosRz;
    float sinRySinRz = sinRy * sinRz;
    float sinRyCosRx = sinRy * cosRx;
    float sinRxSinRy = sinRx * sinRy;

    float denominator = std::sqrt(
        std::pow(-bax + itax * cosRyCosRz - itay * sinRzCosRy + itaz * sinRy + tx, 2.f) +
        std::pow(
            -bay + itax * (sin_rx_sin_ry_cos_rz + sinRzCosRx) - itay * (sinRxSinRySinRz - cosRxCosRz) -
                itaz * sinRxCosRy + ty,
            2.f) +
        std::pow(
            -baz + itax * (sinRxSinRz - sinRyCosRxCosRz) + itay * (sinRxCosRz + sinRySinRzCosRx) + itaz * cosRxCosRy +
                tz,
            2.f));

    error = -d + denominator;

    grad[0] = (-bax + itax * cosRyCosRz - itay * sinRzCosRy + itaz * sinRy + tx) / denominator;
    grad[1] = (-bay + itax * (sin_rx_sin_ry_cos_rz + sinRzCosRx) - itay * (sinRxSinRySinRz - cosRxCosRz) -
               itaz * sinRxCosRy + ty) /
              denominator;
    grad[2] = (-baz + itax * (sinRxSinRz - sinRyCosRxCosRz) + itay * (sinRxCosRz + sinRySinRzCosRx) +
               itaz * cosRxCosRy + tz) /
              denominator;

    grad[3] =
        (-(itax * (sinRxSinRz - sinRyCosRxCosRz) + itay * (sinRxCosRz + sinRySinRzCosRx) + itaz * cosRxCosRy) *
             (-bay + itax * (sin_rx_sin_ry_cos_rz + sinRzCosRx) + itay * (-sinRxSinRySinRz + cosRxCosRz) -
              itaz * sinRxCosRy + ty) +
         (itax * (sin_rx_sin_ry_cos_rz + sinRzCosRx) + itay * (-sinRxSinRySinRz + cosRxCosRz) - itaz * sinRxCosRy) *
             (-baz + itax * (sinRxSinRz - sinRyCosRxCosRz) + itay * (sinRxCosRz + sinRySinRzCosRx) + itaz * cosRxCosRy +
              tz)) /
        denominator;
    grad[4] =
        (bax * itax * sinRyCosRz - bax * itay * sinRySinRz - bax * itaz * cosRy - bay * itax * sinRxCosRyCosRz +
         bay * itay * sinRxSinRzCosRy - bay * itaz * sinRxSinRy + baz * itax * cosRxCosRyCosRz -
         baz * itay * sinRzCosRxCosRy + baz * itaz * sinRyCosRx - itax * tx * sinRyCosRz + itax * ty * sinRxCosRyCosRz -
         itax * tz * cosRxCosRyCosRz + itay * tx * sinRySinRz - itay * ty * sinRxSinRzCosRy +
         itay * tz * sinRzCosRxCosRy + itaz * tx * cosRy + itaz * ty * sinRxSinRy - itaz * tz * sinRyCosRx) /
        denominator;
    grad[5] =
        (bax * itax * sinRzCosRy + bax * itay * cosRyCosRz + bay * itax * sinRxSinRySinRz - bay * itax * cosRxCosRz +
         bay * itay * sin_rx_sin_ry_cos_rz + bay * itay * sinRzCosRx - baz * itax * sinRxCosRz -
         baz * itax * sinRySinRzCosRx + baz * itay * sinRxSinRz - baz * itay * sinRyCosRxCosRz -
         itax * tx * sinRzCosRy - itax * ty * sinRxSinRySinRz + itax * ty * cosRxCosRz + itax * tz * sinRxCosRz +
         itax * tz * sinRySinRzCosRx - itay * tx * cosRyCosRz - itay * ty * sin_rx_sin_ry_cos_rz -
         itay * ty * sinRzCosRx - itay * tz * sinRxSinRz + itay * tz * sinRyCosRx * cosRz) /
        denominator;
}

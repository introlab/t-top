#include "StewartKinematicsParameters.h"

const float STEWART_HORN_ORIENTATION_ANGLES_DATA[STEWART_SERVO_COUNT] = {
    1.5707963267948966f,
    0.5235987756005183f,
    -2.617993877989275f,
    2.617993877989275f,
    -0.5235987756005183f,
    -1.5707963267948966f
};
const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> STEWART_HORN_ORIENTATION_ANGLES(STEWART_HORN_ORIENTATION_ANGLES_DATA);

const bool STEWART_IS_HORN_ORIENTATION_REVERSED[STEWART_SERVO_COUNT] = {true, false, true, false, true, false};

const Eigen::Vector3f STEWART_TOP_ANCHORS[STEWART_SERVO_COUNT] = {
    Eigen::Vector3f(-0.03673734124f, -0.03363094156f, 0.0f),
    Eigen::Vector3f(-0.01075657913f, -0.04863094156f, 0.0f),
    Eigen::Vector3f(0.04749392037f, -0.015f, 0.0f),
    Eigen::Vector3f(0.04749392037f, 0.015f, 0.0f),
    Eigen::Vector3f(-0.01075657913f, 0.04863094156f, 0.0f),
    Eigen::Vector3f(-0.03673734124f, 0.03363094156f, 0.0f)
};

const Eigen::Vector3f STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[STEWART_SERVO_COUNT] = {
    Eigen::Vector3f(-0.12332192622999999f, -0.06f, 0.0f),
    Eigen::Vector3f(0.009699438890019778f, -0.13679992096493046f, 0.0f),
    Eigen::Vector3f(0.11362248734001977f, -0.07679992096493045f, 0.0f),
    Eigen::Vector3f(0.11362248734001977f, 0.07679992096493045f, 0.0f),
    Eigen::Vector3f(0.009699438890019778f, 0.13679992096493046f, 0.0f),
    Eigen::Vector3f(-0.12332192622999999f, 0.06f, 0.0f)
};

float STEWART_FORWARD_KINEMATICS_X0_DATA[6] = {
    8.812734942598466e-12f,
    1.1492058570273169e-17f,
    0.17029453895440808f,
    0.0f,
    -5.963274318787626e-11f,
    0.0f
};
const Eigen::Matrix<float, 6, 1> STEWART_FORWARD_KINEMATICS_X0(STEWART_FORWARD_KINEMATICS_X0_DATA);

float STEWART_FORWARD_KINEMATICS_MIN_BOUNDS_DATA[6] = {
    -0.09599999999118734f,
    -0.07200000000000004f,
    0.12629453895440804f,
    -0.8200000000000004f,
    -0.8800000000596332f,
    -1.5000000000000009f
};
const Eigen::Matrix<float, 6, 1> STEWART_FORWARD_KINEMATICS_MIN_BOUNDS(STEWART_FORWARD_KINEMATICS_MIN_BOUNDS_DATA);

float STEWART_FORWARD_KINEMATICS_MAX_BOUNDS_DATA[6] = {
    0.06600000000881277f,
    0.07200000000000005f,
    0.21429453895440811f,
    0.8200000000000004f,
    1.0199999999403677f,
    1.5000000000000009f
};
const Eigen::Matrix<float, 6, 1> STEWART_FORWARD_KINEMATICS_MAX_BOUNDS(STEWART_FORWARD_KINEMATICS_MAX_BOUNDS_DATA);
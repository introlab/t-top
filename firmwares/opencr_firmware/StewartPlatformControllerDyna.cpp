#include "StewartPlatformControllerDyna.h"

#include <Eigen/Geometry>
#include <cstring>
#include <cmath>

#define POW_2(x) ((x) * (x)) 

static const int MOTOR_INIT_DELAY_MS = 250;

static const float ROD_LENGTH = 0.18107669683;
static const float HORN_LENGTH = 0.03;
static const float TOP_INITIAL_Z = 0.16893650962222248;

static const float HORN_ORIENTATION_ANGLES[STEWART_PLATFORM_DYNAMIXEL_COUNT] = {
  1.5707963267926772,
  0.5235987755982988,
  -2.6179938779914944,
  2.6179938779937135,
  -0.5235987755960796,
  -1.5707963267971161
};

static const bool IS_HORN_ORIENTATION_REVERSED[STEWART_PLATFORM_DYNAMIXEL_COUNT] = {
  true,
  false,
  true,
  false,
  true,
  false
};

static const Eigen::Vector3f TOP_ANCHORS[STEWART_PLATFORM_DYNAMIXEL_COUNT] = {
  Eigen::Vector3f(-0.0363303810590635, -0.03292606584624272, 0.0),
  Eigen::Vector3f(-0.01034961894229692, -0.047926065851843176, 0.0),
  Eigen::Vector3f(0.04668000000229053, -0.01500000000240383, 0.0),
  Eigen::Vector3f(0.046679999999057115, 0.015000000003196621, 0.0),
  Eigen::Vector3f(-0.010349618943233418, 0.047926065848657606, 0.0),
  Eigen::Vector3f(-0.03633038105676658, 0.03292606584865761, 0.0)
};

static const Eigen::Vector3f BOTTOM_LINEAR_ACTUATOR_ANCHORS[STEWART_PLATFORM_DYNAMIXEL_COUNT] = {
  Eigen::Vector3f(-0.10145076181816247, -0.06000000000229673, 0.0),
  Eigen::Vector3f(-0.001236143317066325, -0.11785893696940442, 0.0),
  Eigen::Vector3f(0.10268690513706631, -0.05785893696940442, 0.0),
  Eigen::Vector3f(0.1026869051381366, 0.05785893696666469, 0.0),
  Eigen::Vector3f(-0.0012361433189297145, 0.117858936971746, 0.0),
  Eigen::Vector3f(-0.10145076182109615, 0.06000000000278456, 0.0)
};

StewartPlatformControllerDyna::StewartPlatformControllerDyna(DynamixelWorkbench& dynamixelWorkbench) : 
  m_dynamixelWorkbench(dynamixelWorkbench),
  m_isPoseReachable(true),
  m_position(0, 0, TOP_INITIAL_Z) {
  m_orientation << 1, 0, 0,
                   0, 1, 0,
                   0, 0, 1;
}

StewartPlatformControllerDyna::~StewartPlatformControllerDyna() {
  
}

void StewartPlatformControllerDyna::init() {
  for (int i = 0; i < STEWART_PLATFORM_DYNAMIXEL_COUNT; i++) {
    m_dynamixelWorkbench.ping(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

    m_dynamixelWorkbench.torqueOff(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
    m_dynamixelWorkbench.setNormalDirection(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
    m_dynamixelWorkbench.torqueOn(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);

    m_dynamixelWorkbench.jointMode(STEWART_PLATFORM_DYNAMIXEL_IDS[i]);
    delay(MOTOR_INIT_DELAY_MS);
  }

  for (int i = 0; i < STEWART_PLATFORM_DYNAMIXEL_COUNT; i++) {
    m_dynamixelWorkbench.goalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], 0.f);
    delay(MOTOR_INIT_DELAY_MS);
  }
}

void StewartPlatformControllerDyna::setPose(const geometry_msgs::PoseStamped& pose) {
  if (std::strcmp(pose.header.frame_id, "stewart_base") != 0) {
    m_isPoseReachable = false;
    return;
  }
  
  m_position << pose.pose.position.x, pose.pose.position.y, pose.pose.position.z;
  Eigen::Quaternionf quaternion(pose.pose.orientation.w,
    pose.pose.orientation.x,
    pose.pose.orientation.y,
    pose.pose.orientation.z);
  m_orientation = quaternion.toRotationMatrix();

  m_isPoseReachable = calculateInverseKinematicsServoAngles();
}

bool StewartPlatformControllerDyna::isPoseReachable() {
  return m_isPoseReachable;
}

void StewartPlatformControllerDyna::readCurrentPose(float servoAngles[], geometry_msgs::PoseStamped& pose) {
  for (int i = 0; i < STEWART_PLATFORM_DYNAMIXEL_COUNT; i++) {
    m_dynamixelWorkbench.getRadian(STEWART_PLATFORM_DYNAMIXEL_IDS[i], servoAngles + i);
  }

  pose.header.frame_id = "stewart_base";
  m_stewartForwardKinematicsNeuralNetwork.calculateCurrentPose(servoAngles, pose);
}

bool StewartPlatformControllerDyna::calculateInverseKinematicsServoAngles() {
  for (int i = 0; i < STEWART_PLATFORM_DYNAMIXEL_COUNT; i++) {
    Eigen::Vector3f l = m_orientation * TOP_ANCHORS[i] + m_position - BOTTOM_LINEAR_ACTUATOR_ANCHORS[i];

    float ek = 2 * HORN_LENGTH * l.z();
    float fk = 2 * HORN_LENGTH * (std::cos(HORN_ORIENTATION_ANGLES[i]) * l.x() + 
      std::sin(HORN_ORIENTATION_ANGLES[i]) * l.y());
    float gk = POW_2(l.x()) + POW_2(l.y()) + POW_2(l.z()) - POW_2(ROD_LENGTH) + POW_2(HORN_LENGTH);

    float asinParameter = gk / std::sqrt(POW_2(ek) + POW_2(fk));
    if (asinParameter < -1.f || asinParameter > 1.f) {
      return false;
    }

    m_inverseKinematicsServoAngles[i] = std::asin(asinParameter) - std::atan2(fk, ek);

    if (IS_HORN_ORIENTATION_REVERSED[i]) {
      m_inverseKinematicsServoAngles[i] = -m_inverseKinematicsServoAngles[i];  
    }
  }

  for (int i = 0; i < STEWART_PLATFORM_DYNAMIXEL_COUNT; i++) {
    m_dynamixelWorkbench.goalPosition(STEWART_PLATFORM_DYNAMIXEL_IDS[i], m_inverseKinematicsServoAngles[i]);
  }

  return true;
}

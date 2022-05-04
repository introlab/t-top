#include <Arduino.h>
#include <SPI.h>
#include <chip.h>

#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <geometry_msgs/PoseStamped.h>

#include "MPU9250.h"
#include "StewartPlatformController.h"
#include "TorsoController.h"
#include "PsuControlCommandHandler.h"
#include "PsuControlCommandSender.h"
#include "Ticker.h"

// ROS
constexpr unsigned long ROS_SERIAL_BAUD_RATE = 1000000;
static ros::NodeHandle nh;

static void setTorsoOrientation(const std_msgs::Float32& msg);
static ros::Subscriber<std_msgs::Float32> setTorsoOrientationSub("opencr/torso_orientation", &setTorsoOrientation);

static void setHeadPose(const geometry_msgs::PoseStamped& msg);
static ros::Subscriber<geometry_msgs::PoseStamped> setHeadPoseSub("opencr/head_pose", &setHeadPose);

static void setAudioPowerAmplifierVolume(const std_msgs::Int8& msg);
static ros::Subscriber<std_msgs::Int8>
    setAudioPowerAmplifierVolumeSub("opencr/audio_power_amplifier_volume", &setAudioPowerAmplifierVolume);


static std_msgs::Float32 currentTorsoOrientationMsg;
static ros::Publisher currentTorsoOrientationPub("opencr/current_torso_orientation", &currentTorsoOrientationMsg);

static std_msgs::Int32 currentTorsoServoSpeedMsg;
static ros::Publisher currentTorsoServoSpeedPub("opencr/current_torso_servo_speed", &currentTorsoServoSpeedMsg);


constexpr int CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH = 6;
static float currentHeadServoAnglesMsgData[CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH];
static std_msgs::Float32MultiArray currentHeadServoAnglesMsg;
static ros::Publisher currentHeadServoAnglesPub("opencr/current_head_servo_angles", &currentHeadServoAnglesMsg);

constexpr int CURRENT_HEAD_SERVO_SPEEDS_MSG_DATA_LENGTH = 6;
static int32_t currentHeadServoSpeedsMsgData[CURRENT_HEAD_SERVO_SPEEDS_MSG_DATA_LENGTH];
static std_msgs::Int32MultiArray currentHeadServoSpeedsMsg;
static ros::Publisher currentHeadServoSpeedsPub("opencr/current_head_servo_speeds", &currentHeadServoSpeedsMsg);

static geometry_msgs::PoseStamped currentHeadPoseMsg;
static ros::Publisher currentHeadPosePub("opencr/current_head_pose", &currentHeadPoseMsg);

static std_msgs::Bool isHeadPoseReachableMsg;
static ros::Publisher isHeadPoseReachablePub("opencr/is_head_pose_reachable", &isHeadPoseReachableMsg);


constexpr int RAW_IMU_MSG_DATA_LENGTH = 9;
static float rawImuMsgData[RAW_IMU_MSG_DATA_LENGTH];
static std_msgs::Float32MultiArray rawImuMsg;
static ros::Publisher rawImuPub("opencr/raw_imu", &rawImuMsg);

// Message format : stateOfCharge, current, voltage, isPsuConnected, isBatteryCharging
constexpr int BASE_STATUS_MSG_DATA_LENGTH = 5;
static float baseStatusMsgData[BASE_STATUS_MSG_DATA_LENGTH];
static std_msgs::Float32MultiArray baseStatusMsg;
static ros::Publisher baseStatusPub("opencr/base_status", &baseStatusMsg);

// Robot
static MPU9250 imu(SPI_IMU, BDPIN_SPI_CS_IMU);

const char* DYNAMIXEL_BUS_SERIAL = "/dev/ttyUSB0";
constexpr long DYNAMIXEL_BAUDRATE = 1000000;
static DynamixelWorkbench dynamixelWorkbench;
static StewartPlatformController stewartPlatformController(dynamixelWorkbench);
static TorsoController torsoController(dynamixelWorkbench);

constexpr unsigned long PSU_CONTROL_BAUD_RATE = 9600;
static PsuControlCommandHandler psuControlCommandHandler;
static PsuControlCommandSender psuControlCommandSender;

// Timers
constexpr long ROS_TIMER_PERIOD_MS = 1;
constexpr long SERVO_STATUS_TIMER_PERIOD_MS = 33;
constexpr long IMU_TIMER_PERIOD_MS = 10;

static void onRosTimer();
static Ticker rosTimer(onRosTimer, ROS_TIMER_PERIOD_MS);

static void onServoStatusTimer();
static Ticker servoStatusTimer(onServoStatusTimer, SERVO_STATUS_TIMER_PERIOD_MS);

static void onImuTimer();
static Ticker imuTimer(onImuTimer, IMU_TIMER_PERIOD_MS);


static void setupRos()
{
    nh.getHardware()->setBaud(ROS_SERIAL_BAUD_RATE);

    nh.initNode();
    nh.subscribe(setTorsoOrientationSub);
    nh.subscribe(setHeadPoseSub);
    nh.subscribe(setAudioPowerAmplifierVolumeSub);

    nh.advertise(currentTorsoOrientationPub);
    nh.advertise(currentTorsoServoSpeedPub);
    nh.advertise(currentHeadServoAnglesPub);
    nh.advertise(currentHeadServoSpeedsPub);
    nh.advertise(currentHeadPosePub);
    nh.advertise(isHeadPoseReachablePub);
    nh.advertise(rawImuPub);
    nh.advertise(baseStatusPub);
}

static void setupImu()
{
    imu.begin();
    imu.setAccelRange(MPU9250::ACCEL_RANGE_2G);
    imu.setGyroRange(MPU9250::GYRO_RANGE_250DPS);
    imu.setDlpfBandwidth(MPU9250::DLPF_BANDWIDTH_41HZ);
    imu.setSrd(9);  // 100 Hz update rate
}

static void setupControllers()
{
    dynamixelWorkbench.begin(DYNAMIXEL_BUS_SERIAL, DYNAMIXEL_BAUDRATE);
    stewartPlatformController.init();
    torsoController.init();
}

static void setupPsuControlCommandHandler()
{
    Serial1.begin(PSU_CONTROL_BAUD_RATE);
    psuControlCommandHandler.setStatusCommandHandler(&onStatusCommand);
}

void setup()
{
    setupRos();
    setupImu();
    setupControllers();
    setupPsuControlCommandHandler();

    rosTimer.start();
    servoStatusTimer.start();
    imuTimer.start();
}

void loop()
{
    rosTimer.update();
    servoStatusTimer.update();
    imuTimer.update();
    psuControlCommandHandler.update();
}

static void setTorsoOrientation(const std_msgs::Float32& msg)
{
    torsoController.setOrientation(msg.data);
}

static void setHeadPose(const geometry_msgs::PoseStamped& msg)
{
    stewartPlatformController.setPose(msg);
}

static void setAudioPowerAmplifierVolume(const std_msgs::Int8& msg)
{
    psuControlCommandSender.sendVolumeCommand(msg.data);
}

static void onRosTimer()
{
    nh.spinOnce();
}

static void onServoStatusTimer()
{
    isHeadPoseReachableMsg.data = stewartPlatformController.isPoseReachable();
    isHeadPoseReachablePub.publish(&isHeadPoseReachableMsg);

    stewartPlatformController.readCurrentPose(currentHeadServoAnglesMsgData, currentHeadPoseMsg);
    currentHeadServoAnglesMsg.data = currentHeadServoAnglesMsgData;
    currentHeadServoAnglesMsg.data_length = CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH;
    currentHeadServoAnglesPub.publish(&currentHeadServoAnglesMsg);

    stewartPlatformController.readServoSpeeds(currentHeadServoSpeedsMsgData);
    currentHeadServoSpeedsMsg.data = currentHeadServoSpeedsMsgData;
    currentHeadServoSpeedsMsg.data_length = CURRENT_HEAD_SERVO_SPEEDS_MSG_DATA_LENGTH;
    currentHeadServoSpeedsPub.publish(&currentHeadServoSpeedsMsg);

    currentHeadPosePub.publish(&currentHeadPoseMsg);


    currentTorsoOrientationMsg.data = torsoController.readOrientation();
    currentTorsoOrientationPub.publish(&currentTorsoOrientationMsg);

    currentTorsoServoSpeedMsg.data = torsoController.readServoSpeed();
    currentTorsoServoSpeedPub.publish(&currentTorsoServoSpeedMsg);
}

static void onImuTimer()
{
    imu.readSensor();
    rawImuMsgData[0] = imu.getAccelX_mss();
    rawImuMsgData[1] = imu.getAccelY_mss();
    rawImuMsgData[2] = imu.getAccelZ_mss();
    rawImuMsgData[3] = imu.getGyroX_rads();
    rawImuMsgData[4] = imu.getGyroY_rads();
    rawImuMsgData[5] = imu.getGyroZ_rads();
    rawImuMsgData[6] = imu.getMagX_uT();
    rawImuMsgData[7] = imu.getMagY_uT();
    rawImuMsgData[8] = imu.getMagZ_uT();

    rawImuMsg.data = rawImuMsgData;
    rawImuMsg.data_length = RAW_IMU_MSG_DATA_LENGTH;
    rawImuPub.publish(&rawImuMsg);
}

static void
    onStatusCommand(bool isPsuConnected, bool isBatteryCharging, float stateOfCharge, float current, float voltage)
{
    baseStatusMsgData[0] = stateOfCharge;
    baseStatusMsgData[1] = voltage;
    baseStatusMsgData[2] = current;
    baseStatusMsgData[3] = isPsuConnected;
    baseStatusMsgData[4] = isBatteryCharging;

    baseStatusMsg.data = baseStatusMsgData;
    baseStatusMsg.data_length = BASE_STATUS_MSG_DATA_LENGTH;
    baseStatusPub.publish(&baseStatusMsg);
}

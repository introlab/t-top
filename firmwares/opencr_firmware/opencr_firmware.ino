#include <Arduino.h>
#include <SPI.h>
#include <chip.h>

#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PoseStamped.h>

#include "MPU9250.h"
#include "StewartPlatformController.h"
#include "TorsoController.h"
#include "PsuControlCommandHandler.h"
#include "PsuControlCommandSender.h"
#include "Ticker.h"

// ROS
constexpr unsigned long ROS_SERIAL_BAUD_RATE = 1000000;
ros::NodeHandle nh;

void setTorsoOrientation(const std_msgs::Float32& msg);
ros::Subscriber<std_msgs::Float32> setTorsoOrientationSub("opencr/torso_orientation", &setTorsoOrientation);

void setHeadPose(const geometry_msgs::PoseStamped& msg);
ros::Subscriber<geometry_msgs::PoseStamped> setHeadPoseSub("opencr/head_pose", &setHeadPose);

void setAudioPowerAmplifierVolume(const std_msgs::Int8& msg);
ros::Subscriber<std_msgs::Int8>
    setAudioPowerAmplifierVolumeSub("opencr/audio_power_amplifier_volume", &setAudioPowerAmplifierVolume);


std_msgs::Float32 currentTorsoOrientationMsg;
ros::Publisher currentTorsoOrientationPub("opencr/current_torso_orientation", &currentTorsoOrientationMsg);

constexpr int CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH = 6;
float currentHeadServoAnglesData[CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH];
std_msgs::Float32MultiArray currentHeadServoAnglesMsg;
ros::Publisher currentHeadServoAnglesPub("opencr/current_head_servo_angles", &currentHeadServoAnglesMsg);

geometry_msgs::PoseStamped currentHeadPoseMsg;
ros::Publisher currentHeadPosePub("opencr/current_head_pose", &currentHeadPoseMsg);

std_msgs::Bool isHeadPoseReachableMsg;
ros::Publisher isHeadPoseReachablePub("opencr/is_head_pose_reachable", &isHeadPoseReachableMsg);

constexpr int RAW_IMU_MSG_DATA_LENGTH = 9;
float rawImuMsgData[RAW_IMU_MSG_DATA_LENGTH];
std_msgs::Float32MultiArray rawImuMsg;
ros::Publisher rawImuPub("opencr/raw_imu", &rawImuMsg);

constexpr int STATE_OF_CHARGE_VOLTAGE_CURRENT_MSG_DATA_LENGTH = 3;
float stateOfChargeVoltageCurrentMsgData[STATE_OF_CHARGE_VOLTAGE_CURRENT_MSG_DATA_LENGTH];
std_msgs::Float32MultiArray stateOfChargeVoltageCurrentMsg;
ros::Publisher
    stateOfChargeVoltageCurrentPub("opencr/state_of_charge_voltage_current", &stateOfChargeVoltageCurrentMsg);

std_msgs::Bool isBatteryChargingMsg;
ros::Publisher isBatteryChargingPub("opencr/is_battery_charging", &isBatteryChargingMsg);

// Robot
MPU9250 imu(SPI_IMU, BDPIN_SPI_CS_IMU);

const char* DYNAMIXEL_BUS_SERIAL = "/dev/ttyUSB0";
constexpr long DYNAMIXEL_BAUDRATE = 1000000;
DynamixelWorkbench dynamixelWorkbench;
StewartPlatformController stewartPlatformController(dynamixelWorkbench);
TorsoController torsoController(dynamixelWorkbench);

constexpr unsigned long PSU_CONTROL_BAUD_RATE = 9600;
PsuControlCommandHandler psuControlCommandHandler;
PsuControlCommandSender psuControlCommandSender;

// Timers
constexpr long ROS_TIMER_PERIOD_MS = 1;
constexpr long SERVO_STATUS_TIMER_PERIOD_MS = 33;
constexpr long IMU_TIMER_PERIOD_MS = 10;

void onRosTimer();
Ticker rosTimer(onRosTimer, ROS_TIMER_PERIOD_MS);

void onServoStatusTimer();
Ticker servoStatusTimer(onServoStatusTimer, SERVO_STATUS_TIMER_PERIOD_MS);

void onImuTimer();
Ticker imuTimer(onImuTimer, IMU_TIMER_PERIOD_MS);


void setupRos()
{
    // Setup ROS
    nh.getHardware()->setBaud(ROS_SERIAL_BAUD_RATE);

    nh.initNode();
    nh.subscribe(setTorsoOrientationSub);
    nh.subscribe(setHeadPoseSub);
    nh.subscribe(setAudioPowerAmplifierVolumeSub);

    nh.advertise(currentTorsoOrientationPub);
    nh.advertise(currentHeadServoAnglesPub);
    nh.advertise(currentHeadPosePub);
    nh.advertise(isHeadPoseReachablePub);
    nh.advertise(rawImuPub);
    nh.advertise(stateOfChargeVoltageCurrentPub);
    nh.advertise(isBatteryChargingPub);
}

void setupImu()
{
    imu.begin();
    imu.setAccelRange(MPU9250::ACCEL_RANGE_2G);
    imu.setGyroRange(MPU9250::GYRO_RANGE_250DPS);
    imu.setDlpfBandwidth(MPU9250::DLPF_BANDWIDTH_41HZ);
    imu.setSrd(9);  // 100 Hz update rate
}

void setupControllers()
{
    dynamixelWorkbench.begin(DYNAMIXEL_BUS_SERIAL, DYNAMIXEL_BAUDRATE);
    stewartPlatformController.init();
    torsoController.init();
}

void setupPsuControlCommandHandler()
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

void setTorsoOrientation(const std_msgs::Float32& msg)
{
    torsoController.setOrientation(msg.data);
}

void setHeadPose(const geometry_msgs::PoseStamped& msg)
{
    stewartPlatformController.setPose(msg);
}

void setAudioPowerAmplifierVolume(const std_msgs::Int8& msg)
{
    psuControlCommandSender.sendVolumeCommand(msg.data);
}

void onRosTimer()
{
    nh.spinOnce();
}

void onServoStatusTimer()
{
    isHeadPoseReachableMsg.data = stewartPlatformController.isPoseReachable();
    isHeadPoseReachablePub.publish(&isHeadPoseReachableMsg);

    stewartPlatformController.readCurrentPose(currentHeadServoAnglesData, currentHeadPoseMsg);

    currentHeadServoAnglesMsg.data = &currentHeadServoAnglesData[0];
    currentHeadServoAnglesMsg.data_length = CURRENT_HEAD_SERVO_ANGLES_MSG_DATA_LENGTH;
    currentHeadServoAnglesPub.publish(&currentHeadServoAnglesMsg);

    currentHeadPosePub.publish(&currentHeadPoseMsg);


    currentTorsoOrientationMsg.data = torsoController.readOrientation();
    currentTorsoOrientationPub.publish(&currentTorsoOrientationMsg);
}

void onImuTimer()
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

    rawImuMsg.data = &rawImuMsgData[0];
    rawImuMsg.data_length = RAW_IMU_MSG_DATA_LENGTH;
    rawImuPub.publish(&rawImuMsg);
}

void onStatusCommand(bool isBatteryCharging, float stateOfCharge, float current, float voltage)
{
    stateOfChargeVoltageCurrentMsgData[0] = stateOfCharge;
    stateOfChargeVoltageCurrentMsgData[1] = current;
    stateOfChargeVoltageCurrentMsgData[2] = voltage;

    stateOfChargeVoltageCurrentMsg.data = &stateOfChargeVoltageCurrentMsgData[0];
    stateOfChargeVoltageCurrentMsg.data_length = STATE_OF_CHARGE_VOLTAGE_CURRENT_MSG_DATA_LENGTH;
    stateOfChargeVoltageCurrentPub.publish(&stateOfChargeVoltageCurrentMsg);
}

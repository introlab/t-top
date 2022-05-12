#include <Arduino.h>

#include <ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Empty.h>

#include "IoMapping.h"
#include "TorsoController.h"
#include "ServoSpeedController.h"

#include <cmath>

// Constants
constexpr unsigned long INIT_DELAY = 10000;
constexpr unsigned long DATA_GATHERING_DELAY = 1000;
constexpr unsigned long AUDIO_RECORDING_DELAY = 2000;
constexpr unsigned long TORSO_STEP_DELAY = 10;

constexpr float TORSO_ORIENTATION_START = 0;
constexpr float TORSO_ORIENTATION_END = 2 * M_PI / 16;
constexpr float TORSO_ORIENTATION_STEP = TORSO_ORIENTATION_END / 10;

constexpr int32_t HEAD_SERVO_SPEED_START = 25;
constexpr int32_t HEAD_SERVO_SPEED_END = 265;
constexpr int32_t HEAD_SERVO_SPEED_STEP = 5;

constexpr int32_t TORSO_SERVO_SPEED_START = 5;
constexpr int32_t TORSO_SERVO_SPEED_END = TORSO_MAX_VELOCITY;
constexpr int32_t TORSO_SERVO_SPEED_STEP = 5;

// ROS
constexpr unsigned long ROS_SERIAL_BAUD_RATE = 1000000;
static ros::NodeHandle nh;

static std_msgs::Float32 currentTorsoOrientationMsg;
static ros::Publisher currentTorsoOrientationPub("opencr/current_torso_orientation", &currentTorsoOrientationMsg);

static std_msgs::Int32 movingServoIdMsg;
static ros::Publisher movingServoIdPub("opencr/moving_servo_id", &movingServoIdMsg);

static std_msgs::Int32 movingServoSpeedTargetMsg;
static ros::Publisher movingServoSpeedTargetPub("opencr/moving_servo_speed_target", &movingServoSpeedTargetMsg);

static std_msgs::Int32 movingServoSpeedMsg;
static ros::Publisher movingServoSpeedPub("opencr/moving_servo_speed", &movingServoSpeedMsg);

static std_msgs::Empty startHeadServoAudioRecordingMsg;
static ros::Publisher
    startHeadServoAudioRecordingPub("opencr/start_head_servo_audio_recording", &startHeadServoAudioRecordingMsg);

static std_msgs::Empty stopHeadServoAudioRecordingMsg;
static ros::Publisher
    stopHeadServoAudioRecordingPub("opencr/stop_head_servo_audio_recording", &stopHeadServoAudioRecordingMsg);

static std_msgs::Empty startTorsoServoAudioRecordingMsg;
static ros::Publisher
    startTorsoServoAudioRecordingPub("opencr/start_torso_servo_audio_recording", &startTorsoServoAudioRecordingMsg);

static std_msgs::Empty stopTorsoServoAudioRecordingMsg;
static ros::Publisher
    stopTorsoServoAudioRecordingPub("opencr/stop_torso_servo_audio_recording", &stopTorsoServoAudioRecordingMsg);

static std_msgs::Empty egoNoiseDataGatheringFinishedMsg;
static ros::Publisher
    egoNoiseDataGatheringFinishedPub("opencr/ego_noise_data_gathering_finished", &egoNoiseDataGatheringFinishedMsg);

static void startEgoNoiseDataGathering(const std_msgs::Empty& msg);
static ros::Subscriber<std_msgs::Empty>
    startEgoNoiseDataGatheringSub("opencr/start_ego_noise_data_gathering", &startEgoNoiseDataGathering);

const char* DYNAMIXEL_BUS_SERIAL = "/dev/ttyUSB0";
constexpr long DYNAMIXEL_BAUDRATE = 1000000;
DynamixelWorkbench dynamixelWorkbench;

static bool dataGatheringStarted = false;

static void gatherEgoNoiseData();

void setup()
{
    nh.getHardware()->setBaud(ROS_SERIAL_BAUD_RATE);

    nh.initNode();
    nh.subscribe(startEgoNoiseDataGatheringSub);

    nh.advertise(currentTorsoOrientationPub);
    nh.advertise(movingServoIdPub);
    nh.advertise(movingServoSpeedTargetPub);
    nh.advertise(movingServoSpeedPub);
    nh.advertise(startHeadServoAudioRecordingPub);
    nh.advertise(stopHeadServoAudioRecordingPub);
    nh.advertise(startTorsoServoAudioRecordingPub);
    nh.advertise(stopTorsoServoAudioRecordingPub);
    nh.advertise(egoNoiseDataGatheringFinishedPub);

    dynamixelWorkbench.begin(DYNAMIXEL_BUS_SERIAL, DYNAMIXEL_BAUDRATE);
}

void loop()
{
    if (!dataGatheringStarted)
    {
        nh.spinOnce();
    }
    else
    {
        gatherEgoNoiseData();
        dataGatheringStarted = false;
    }
}

void nhDelay(unsigned long ms)
{
    unsigned long start = millis();

    while ((millis() - start) < ms)
    {
        nh.spinOnce();
    }
}

static void startEgoNoiseDataGathering(const std_msgs::Empty& msg)
{
    dataGatheringStarted = true;
}

static void sendTorsoOrientation(TorsoController& torsoController)
{
    currentTorsoOrientationMsg.data = torsoController.readOrientation();
    currentTorsoOrientationPub.publish(&currentTorsoOrientationMsg);
}

static void sendTorsoOrientation(float orientation)
{
    currentTorsoOrientationMsg.data = orientation;
    currentTorsoOrientationPub.publish(&currentTorsoOrientationMsg);
}

static void sendMovingServoId(uint8_t id)
{
    movingServoIdMsg.data = id;
    movingServoIdPub.publish(&movingServoIdMsg);
}

static void sendMovingServoSpeedTarget(int32_t speed)
{
    movingServoSpeedTargetMsg.data = speed;
    movingServoSpeedTargetPub.publish(&movingServoSpeedTargetMsg);
}

static void sendMovingServoSpeed(int32_t speed)
{
    movingServoSpeedMsg.data = speed;
    movingServoSpeedPub.publish(&movingServoSpeedMsg);
}

static void startHeadServoAudioRecording()
{
    startHeadServoAudioRecordingPub.publish(&startHeadServoAudioRecordingMsg);
}

static void stopHeadServoAudioRecording()
{
    stopHeadServoAudioRecordingPub.publish(&stopHeadServoAudioRecordingMsg);
}

static void startTorsoServoAudioRecording()
{
    startTorsoServoAudioRecordingPub.publish(&startTorsoServoAudioRecordingMsg);
}

static void stopTorsoServoAudioRecording()
{
    stopTorsoServoAudioRecordingPub.publish(&stopTorsoServoAudioRecordingMsg);
}

static void gatherHeadServoEgoNoiseData()
{
    ServoSpeedController servoSpeedControllers[STEWART_SERVO_COUNT] = {
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[0]),
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[1]),
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[2]),
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[3]),
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[4]),
        ServoSpeedController(dynamixelWorkbench, STEWART_PLATFORM_DYNAMIXEL_IDS[5])};

    nhDelay(INIT_DELAY);

    for (size_t servoIndex = 0; servoIndex < STEWART_SERVO_COUNT; servoIndex++)
    {
        sendMovingServoId(STEWART_PLATFORM_DYNAMIXEL_IDS[servoIndex]);

        for (int32_t speed = HEAD_SERVO_SPEED_START; speed <= HEAD_SERVO_SPEED_END; speed += HEAD_SERVO_SPEED_STEP)
        {
            sendMovingServoSpeedTarget(speed);
            servoSpeedControllers[servoIndex].setSpeed(speed);
            nhDelay(DATA_GATHERING_DELAY);
            sendMovingServoSpeed(servoSpeedControllers[servoIndex].readSpeed());
            nhDelay(DATA_GATHERING_DELAY);

            startHeadServoAudioRecording();
            nhDelay(AUDIO_RECORDING_DELAY);
            stopHeadServoAudioRecording();
        }

        servoSpeedControllers[servoIndex].setSpeed(0);
        nhDelay(DATA_GATHERING_DELAY);
    }
}

static void gatherAllHeadServoEgoNoiseData()
{
    TorsoController torsoController(dynamixelWorkbench);
    torsoController.init();
    nhDelay(INIT_DELAY);

    for (float o = TORSO_ORIENTATION_START; o < TORSO_ORIENTATION_END; o += TORSO_ORIENTATION_STEP)
    {
        torsoController.setOrientation(o);
        nhDelay(DATA_GATHERING_DELAY);
        sendTorsoOrientation(torsoController);
        nhDelay(DATA_GATHERING_DELAY);

        gatherHeadServoEgoNoiseData();
    }
    torsoController.setOrientation(0);
    nhDelay(INIT_DELAY);
}

static void gatherTorsoServoEgoNoiseData(int32_t speed)
{
    sendMovingServoSpeedTarget(speed);
    {
        TorsoController torsoController(dynamixelWorkbench);
        torsoController.init();
        nhDelay(INIT_DELAY);
    }

    ServoSpeedController servoSpeedController(dynamixelWorkbench, TORSO_DYNAMIXEL_ID);

    float lastPosition = fmodRadian(servoSpeedController.readPosition());
    float orientation = 0.f;
    bool recordingStarted = false;

    sendMovingServoSpeedTarget(speed);
    servoSpeedController.setSpeed(speed);

    while (orientation < (4 * M_PI))
    {
        if (orientation > (M_PI / 2) && !recordingStarted)
        {
            startTorsoServoAudioRecording();
            recordingStarted = true;
        }

        float position = fmodRadian(servoSpeedController.readPosition());
        orientation += fmodRadian(position - lastPosition) * TORSO_GEAR_RATIO;
        lastPosition = position;

        sendTorsoOrientation(fmodRadian(orientation));
        sendMovingServoSpeed(servoSpeedController.readSpeed());
        nhDelay(TORSO_STEP_DELAY);
    }

    stopTorsoServoAudioRecording();
    nhDelay(DATA_GATHERING_DELAY);
}

static void gatherAllTorsoServoEgoNoiseData()
{
    sendMovingServoId(TORSO_DYNAMIXEL_ID);
    nhDelay(DATA_GATHERING_DELAY);

    for (int32_t speed = TORSO_SERVO_SPEED_START; speed <= TORSO_SERVO_SPEED_END; speed += TORSO_SERVO_SPEED_STEP)
    {
        gatherTorsoServoEgoNoiseData(speed);
    }
}

static void gatherEgoNoiseData()
{
    gatherAllHeadServoEgoNoiseData();
    gatherAllTorsoServoEgoNoiseData();
    egoNoiseDataGatheringFinishedPub.publish(&egoNoiseDataGatheringFinishedMsg);
}

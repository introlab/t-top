#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_NORMAL

#include "criticalError.h"
#include "mainCommon.h"
#include "actuators/StewartPlatformController.h"
#include "actuators/TorsoController.h"

#include <SerialCommunication.h>

#include <Ticker.h>

static TeensySerialPort<decltype(COMPUTER_COMMUNICATION_SERIAL)>
    computerSerialCommunicationSerialPort(COMPUTER_COMMUNICATION_SERIAL);
static SerialCommunicationManager computerSerialCommunicationManager(
    Device::DYNAMIXEL_CONTROL,
    COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
    COMMUNICATION_MAXIMUM_TRIAL_COUNT,
    computerSerialCommunicationSerialPort);

static TeensySerialPort<decltype(PSU_CONTROL_COMMUNICATION_SERIAL)>
    psuControlSerialCommunicationSerialPort(PSU_CONTROL_COMMUNICATION_SERIAL);
static SerialCommunicationManager psuControlSerialCommunicationManager(
    Device::DYNAMIXEL_CONTROL,
    COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
    COMMUNICATION_MAXIMUM_TRIAL_COUNT,
    psuControlSerialCommunicationSerialPort);

static volatile bool imuDataReady = false;

static StewartPlatformController stewartPlatformController(dynamixel);
static TorsoController torsoController(dynamixel);

static void setupSerialCommunicationManagers();
static void setupControllers();

static void updateSerialCommunicationManagers();

static void onMotorStatusTicker();
static void onImuDataReadyInterrupt();
static void sendImuData();

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload);
static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload);
static void onShutdownMessage(Device source, const ShutdownPayload& payload);
static void computerRouteCallback(Device destination, const uint8_t* data, size_t size);
static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size);
static void logInvalidRoute(const char* device, Device destination);
static void onComputerSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onPsuControlSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);

Ticker motorStatusTicker(onMotorStatusTicker, MOTOR_STATUS_TICKER_INTERVAL_MS, 0, MILLIS);

void setup()
{
    // put your setup code here, to run once:
    setupDebugSerial();
    setupWire();

    imuDataReady = false;
    setupImu(onImuDataReadyInterrupt);
    setupSerialCommunicationManagers();
    setupDynamixel();
    setupControllers();

    motorStatusTicker.start();
}

static void setupSerialCommunicationManagers()
{
    DEBUG_SERIAL.println("Setup Serial Communication Manager - Start");

    COMPUTER_COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD_RATE);
    computerSerialCommunicationManager.setSetHeadPoseHandler(onSetHeadPoseMessage);
    computerSerialCommunicationManager.setSetTorsoOrientationHandler(onSetTorsoOrientationMessage);
    computerSerialCommunicationManager.setRouteCallback(computerRouteCallback);
    computerSerialCommunicationManager.setErrorCallback(onComputerSerialCommunicationError);

    pinMode(PSU_CONTROL_COMMUNICATION_RS232_INVALID, INPUT);
    PSU_CONTROL_COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD_RATE);
    psuControlSerialCommunicationManager.setShutdownHandler(onShutdownMessage);
    psuControlSerialCommunicationManager.setRouteCallback(psuControlRouteCallback);
    psuControlSerialCommunicationManager.setErrorCallback(onPsuControlSerialCommunicationError);

    DEBUG_SERIAL.println("Setup Serial Communication Manager - End");
}

static void setupControllers()
{
    DEBUG_SERIAL.println("Setup Controllers - Start");

    if (!stewartPlatformController.begin())
    {
        CRITICAL_ERROR("stewartPlatformController.begin() failed");
    }

    torsoController.begin();

    DEBUG_SERIAL.println("Setup Controllers - End");
}

void loop()
{
    if (imuDataReady)
    {
        imuDataReady = false;
        sendImuData();
    }

    motorStatusTicker.update();

    updateSerialCommunicationManagers();
}

static void updateSerialCommunicationManagers()
{
    computerSerialCommunicationManager.update(millis());
    psuControlSerialCommunicationManager.update(millis());
}

static void onMotorStatusTicker()
{
    MotorStatusPayload motorStatusPayload;
    motorStatusPayload.torsoOrientation = torsoController.readOrientation();
    motorStatusPayload.torsoServoSpeed = torsoController.readServoSpeed();
    updateSerialCommunicationManagers();

    float headServoAngles[STEWART_SERVO_COUNT];
    HeadPose headPose;
    int16_t headServoSpeeds[STEWART_SERVO_COUNT];
    stewartPlatformController.readCurrentPose(headServoAngles, headPose);
    stewartPlatformController.readServoSpeeds(headServoSpeeds);
    updateSerialCommunicationManagers();

    motorStatusPayload.headServoAngle1 = headServoAngles[0];
    motorStatusPayload.headServoAngle2 = headServoAngles[1];
    motorStatusPayload.headServoAngle3 = headServoAngles[2];
    motorStatusPayload.headServoAngle4 = headServoAngles[3];
    motorStatusPayload.headServoAngle5 = headServoAngles[4];
    motorStatusPayload.headServoAngle6 = headServoAngles[5];
    motorStatusPayload.headServoSpeed1 = headServoSpeeds[0];
    motorStatusPayload.headServoSpeed2 = headServoSpeeds[1];
    motorStatusPayload.headServoSpeed3 = headServoSpeeds[2];
    motorStatusPayload.headServoSpeed4 = headServoSpeeds[3];
    motorStatusPayload.headServoSpeed5 = headServoSpeeds[4];
    motorStatusPayload.headServoSpeed6 = headServoSpeeds[5];
    motorStatusPayload.headPosePositionX = headPose.positionX;
    motorStatusPayload.headPosePositionY = headPose.positionY;
    motorStatusPayload.headPosePositionZ = headPose.positionZ;
    motorStatusPayload.headPoseOrientationW = headPose.orientationW;
    motorStatusPayload.headPoseOrientationX = headPose.orientationX;
    motorStatusPayload.headPoseOrientationY = headPose.orientationY;
    motorStatusPayload.headPoseOrientationZ = headPose.orientationZ;
    motorStatusPayload.isHeadPoseReachable = stewartPlatformController.isPoseReachable();

    computerSerialCommunicationManager.send(Device::COMPUTER, motorStatusPayload, millis());
}

static void onImuDataReadyInterrupt()
{
    imuDataReady = true;
}

static void sendImuData()
{
    if (!imu.readData())
    {
        return;
    }

    ImuDataPayload imuDataPayload;
    imuDataPayload.accelerationX = imu.getAccelerationXInMPerSS();
    imuDataPayload.accelerationY = imu.getAccelerationYInMPerSS();
    imuDataPayload.accelerationZ = imu.getAccelerationZInMPerSS();
    imuDataPayload.angularRateX = imu.getAngularRateXInRadPerS();
    imuDataPayload.angularRateY = imu.getAngularRateYInRadPerS();
    imuDataPayload.angularRateZ = imu.getAngularRateZInRadPerS();

    computerSerialCommunicationManager.send(Device::COMPUTER, imuDataPayload, millis());
}

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload)
{
    HeadPose pose;
    pose.positionX = payload.headPosePositionX;
    pose.positionY = payload.headPosePositionY;
    pose.positionZ = payload.headPosePositionZ;
    pose.orientationW = payload.headPoseOrientationW;
    pose.orientationX = payload.headPoseOrientationX;
    pose.orientationY = payload.headPoseOrientationY;
    pose.orientationZ = payload.headPoseOrientationZ;

    stewartPlatformController.setPose(pose);
}

static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload)
{
    torsoController.setOrientation(payload.torsoOrientation);
}

static void onShutdownMessage(Device source, const ShutdownPayload& payload)
{
    digitalWrite(DYNAMIXEL_ENABLE_PIN, false);
}

static void computerRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::PSU_CONTROL)
    {
        logInvalidRoute("Computer", destination);
        return;
    }

    psuControlSerialCommunicationManager.sendRaw(data, size);
}

static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::COMPUTER)
    {
        logInvalidRoute("PSU Control", destination);
        return;
    }

    computerSerialCommunicationManager.sendRaw(data, size);
}

static void logInvalidRoute(const char* device, Device destination)
{
    DEBUG_SERIAL.print(device);
    DEBUG_SERIAL.print("Invalid Route - destination=");
    DEBUG_SERIAL.print(static_cast<int>(destination));
    DEBUG_SERIAL.println();
}

static void onComputerSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("Computer ");
    onSerialCommunicationError(message, messageType);
}

static void onPsuControlSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("PSU Control ");
    onSerialCommunicationError(message, messageType);
}

static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType)
{
    DEBUG_SERIAL.print("Serial Communication Manager - ");
    DEBUG_SERIAL.print(message);
    if (messageType.has_value())
    {
        DEBUG_SERIAL.print(" (messageType=");
        DEBUG_SERIAL.print(static_cast<int>(*messageType));
        DEBUG_SERIAL.print(")");
    }
    DEBUG_SERIAL.println();
}

#endif

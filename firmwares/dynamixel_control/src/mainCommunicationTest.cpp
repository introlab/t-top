#include "config.h"

#if FIRMWARE_MODE == FIRMWARE_MODE_COMMUNICATION_TEST

#include "mainCommon.h"

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

static void setupSerialCommunicationManagers();

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload);
static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload);
static void onShutdownMessage(Device source, const ShutdownPayload& payload);
static void computerRouteCallback(Device destination, const uint8_t* data, size_t size);
static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size);
static void logInvalidRoute(const char* device, Device destination);
static void onComputerSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onPsuControlSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);
static void onSerialCommunicationError(const char* message, tl::optional<MessageType> messageType);

void setup()
{
    // put your setup code here, to run once:
    setupDebugSerial();
    setupWire();

    setupSerialCommunicationManagers();
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

template<class Payload>
static void sendMessageToComputer(const char* name, Payload payload)
{
    uint32_t startTime = micros();
    computerSerialCommunicationManager.send(Device::COMPUTER, payload, millis());
    uint32_t endTime = micros();

    DEBUG_SERIAL.print("Elapsed time for sending ");
    DEBUG_SERIAL.print(name);
    DEBUG_SERIAL.print(": ");
    DEBUG_SERIAL.print(endTime - startTime);
    DEBUG_SERIAL.println(" us");

    computerSerialCommunicationManager.update(millis());
}

void loop()
{
    MotorStatusPayload motorStatusPayload;
    motorStatusPayload.torsoOrientation = 1.f;
    motorStatusPayload.torsoServoSpeed = 2;
    motorStatusPayload.headServoAngle1 = 3.f;
    motorStatusPayload.headServoAngle2 = 4.f;
    motorStatusPayload.headServoAngle3 = 5.f;
    motorStatusPayload.headServoAngle4 = 6.f;
    motorStatusPayload.headServoAngle5 = 7.f;
    motorStatusPayload.headServoAngle6 = 8.f;
    motorStatusPayload.headServoSpeed1 = 9;
    motorStatusPayload.headServoSpeed2 = 10;
    motorStatusPayload.headServoSpeed3 = 11;
    motorStatusPayload.headServoSpeed4 = 12;
    motorStatusPayload.headServoSpeed5 = 13;
    motorStatusPayload.headServoSpeed6 = 14;
    motorStatusPayload.headPosePositionX = 15.f;
    motorStatusPayload.headPosePositionY = 16.f;
    motorStatusPayload.headPosePositionZ = 17.f;
    motorStatusPayload.headPoseOrientationW = 18.f;
    motorStatusPayload.headPoseOrientationX = 19.f;
    motorStatusPayload.headPoseOrientationY = 20.f;
    motorStatusPayload.headPoseOrientationZ = 21.f;
    motorStatusPayload.isHeadPoseReachable = false;
    sendMessageToComputer("motor status", motorStatusPayload);

    ImuDataPayload imuDataPayload;
    imuDataPayload.accelerationX = 1.f;
    imuDataPayload.accelerationY = 2.f;
    imuDataPayload.accelerationZ = 3.f;
    imuDataPayload.angularRateX = 4.f;
    imuDataPayload.angularRateX = 5.f;
    imuDataPayload.angularRateX = 6.f;
    sendMessageToComputer("imu data", imuDataPayload);

    uint32_t startTime = millis();
    while ((millis() - startTime) < 1000)
    {
        computerSerialCommunicationManager.update(millis());
        psuControlSerialCommunicationManager.update(millis());
    }
}

static void onSetHeadPoseMessage(Device source, const SetHeadPosePayload& payload)
{
    DEBUG_SERIAL.print("Serial Communication Manager Set Head Pose - ");
    DEBUG_SERIAL.print(" (source=");
    DEBUG_SERIAL.print(static_cast<int>(source));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();
}

static void onSetTorsoOrientationMessage(Device source, const SetTorsoOrientationPayload& payload)
{
    DEBUG_SERIAL.print("Serial Communication Manager Set Torso Orientation - ");
    DEBUG_SERIAL.print(payload.torsoOrientation);
    DEBUG_SERIAL.print(" (source=");
    DEBUG_SERIAL.print(static_cast<int>(source));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();
}

static void onShutdownMessage(Device source, const ShutdownPayload& payload)
{
    DEBUG_SERIAL.print("Serial Communication Manager Shutdown - ");
    DEBUG_SERIAL.print(" (source=");
    DEBUG_SERIAL.print(static_cast<int>(source));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();
}

static void computerRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::PSU_CONTROL)
    {
        logInvalidRoute("Computer", destination);
        return;
    }

    DEBUG_SERIAL.print("Serial Communication Manager Route - ");
    DEBUG_SERIAL.print(" (destination=");
    DEBUG_SERIAL.print(static_cast<int>(destination));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();

    psuControlSerialCommunicationManager.sendRaw(data, size);
}

static void psuControlRouteCallback(Device destination, const uint8_t* data, size_t size)
{
    if (destination != Device::COMPUTER)
    {
        logInvalidRoute("PSU Control", destination);
        return;
    }

    DEBUG_SERIAL.print("Serial Communication Manager Route - ");
    DEBUG_SERIAL.print(" (destination=");
    DEBUG_SERIAL.print(static_cast<int>(destination));
    DEBUG_SERIAL.print(")");
    DEBUG_SERIAL.println();

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